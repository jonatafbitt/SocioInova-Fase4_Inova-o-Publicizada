"""Extração e filtragem dos registros de PI da RFEPCT a partir do dump do INPI (Story 1.3).

Lê ``data/raw/patentes.zip`` (garantido e validado na Story 1.2), cruza os
membros ``PATENTES_DADOS_BIBLIOGRAFICOS.csv``, ``PATENTES_CONTEUDO.csv`` e
``PATENTES_DEPOSITANTES.csv`` em pedaços (``pandas`` ``chunksize``, engine
``python`` para contabilizar linhas rejeitadas), retém registros com **qualquer**
depositante da Rede (união, preserva co-titularidade) e grava
``data/processed/inpi_rfepct_records.csv`` (numero_inpi, data_deposito, titulo,
resumo, instituicoes) + relatório executável ``data/raw/inpi_extracao.json``.

Identificação do depositante da Rede (insensível a caixa/acento):
- primário por crosswalk de CNPJ ``config/crosswalk_cnpj.json`` (14 dígitos);
  crosswalk ausente/vazio ⇒ reservado a preenchimento futuro (só raiz nominal);
- complemento por raiz nominal ancorada em ``INSTITUTO FEDERAL DE``,
  ``CENTRO FEDERAL DE EDUCAC`` e ``COLEGIO PEDRO II`` — nunca subcorda solta.

Nenhum LLM, nenhuma rede: processamento 100% local/offline. Idempotente por
``numero_inpi`` (um registro por número, primeiro encontrado vence).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import unicodedata
import zipfile

import pandas as pd
import psutil

# O engine ``python`` do pandas usa o módulo ``csv``; campos longos (resumos de
# patentes podem passar de 128 KB) estouram o limite padrão. Alça o teto sem
# desligá-lo: valores além de 256 MB são dados patológicos e serão rejeitados.
csv.field_size_limit(256 * 1024 * 1024)

PROJETO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.dirname(os.path.abspath(__file__)) != PROJETO_RAIZ and PROJETO_RAIZ not in sys.path:
    sys.path.insert(0, PROJETO_RAIZ)

from scrapers import inpi_parser as ip  # noqa: E402

__version__ = "1.3.0"

DIR_RAW = os.path.join(PROJETO_RAIZ, "data", "raw")
DIR_PROCESSED = os.path.join(PROJETO_RAIZ, "data", "processed")
ZIP_DEFAULT = os.path.join(DIR_RAW, "patentes.zip")
CSV_DEFAULT = os.path.join(DIR_PROCESSED, "inpi_rfepct_records.csv")
REPORT_DEFAULT = os.path.join(DIR_RAW, "inpi_extracao.json")
CROSSWALK_DEFAULT = os.path.join(PROJETO_RAIZ, "config", "crosswalk_cnpj.json")
VALIDA_DEFAULT = os.path.join(DIR_RAW, "inpi_validacao.json")

MEMBRO_BIBLIOGRAFICOS = "PATENTES_DADOS_BIBLIOGRAFICOS.csv"
MEMBRO_CONTEUDO = "PATENTES_CONTEUDO.csv"
MEMBRO_DEPOSITANTES = "PATENTES_DEPOSITANTES.csv"

# Raízes nominais ancoradas (já normalizadas: sem acento, caixa alta).
# NUNCA usar a subcorda solta "INSTITUTO FEDERAL" (casaria "Fundo Instituto
# Federal..." etc.). Ancoragem metodológica/sociológica — não alterar sem HALT.
RAIZES_RFEPCT = [
    "INSTITUTO FEDERAL DE",
    "CENTRO FEDERAL DE EDUCAC",
    "COLEGIO PEDRO II",
]

CHUNKSIZE = 200000
LIMITE_MEMORIA_MB = 2048

_RE_CNPJ_14 = re.compile(r"^\d{14}$")


def _normalizar(texto: str) -> str:
    """Remove acentos e converte para caixa alta (NFKD + drop combining)."""
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto.upper()


def _cnpj_digitos(cnpj: str | None) -> str:
    return re.sub(r"\D", "", cnpj or "")


def _carregar_crosswalk(caminho: str) -> set[str]:
    """Carrega o conjunto de CNPJs válidos (14 dígitos) a partir do JSON.

    Arquivo ausente/vazio/valores sem 14 dígitos ⇒ conjunto vazio (reservado a
    preenchimento futuro). Aceita tanto topo como lista de strings quanto como
    dict — nesse caso, os valores do dict são usados.
    """
    if not caminho or not os.path.exists(caminho):
        return set()
    try:
        with open(caminho, encoding="utf-8") as f:
            dados = json.load(f)
    except Exception:  # noqa: BLE001 - crosswalk malformado ⇒ reservado
        return set()
    if isinstance(dados, dict):
        valores = [v for v in dados.values() if isinstance(v, str)]
    elif isinstance(dados, list):
        valores = [v for v in dados if isinstance(v, str)]
    else:
        valores = []
    validos = set()
    for v in valores:
        digitos = _cnpj_digitos(v)
        if _RE_CNPJ_14.match(digitos):
            validos.add(digitos)
    return validos


def classificar_depositante(depositante: str, cnpj: str | None,
                            crosswalk: set[str]) -> tuple[bool, str | None]:
    """Retorna ``(e_rede, raiz)`` para um depositante (união: CNPJ OU raiz nominal).

    ``raiz`` = primeira raiz nominal casada; ``"crosswalk_cnpj"`` quando a
    correspondência foi só por CNPJ; ``None`` quando não-Rede.
    """
    digitos = _cnpj_digitos(cnpj)
    if len(digitos) == 14 and digitos in crosswalk:
        return True, "crosswalk_cnpj"
    norm = _normalizar(depositante)
    for raiz in RAIZES_RFEPCT:
        if raiz in norm:
            return True, raiz
    return False, None


# --------------------------------------------------------------------------- #
# Leitura em pedaços + contagem de rejeitados
# --------------------------------------------------------------------------- #
def _contador_rejeitadas(rejeitadas: dict) -> callable:
    def _cb(bad_line):
        rejeitadas["n"] += 1
        return None  # descarta a linha (skip) contabilizada
    return _cb


def _abrir_membro(zf: zipfile.ZipFile, alvo: str) -> tuple[str | None, str, str, list[str]]:
    """Localiza membro e retorna (nome_real|None, encoding, delimitador, colunas)."""
    nome = ip.achar_membro(zf, alvo)
    if nome is None:
        return None, "utf-8", ",", []
    cab, _trunc = ip.ler_cabecalho(zf, nome)
    enc = ip.detectar_encoding(cab)
    delim = ip.detectar_delimitador(cab, enc)
    cols = ip._limpar_cabecalho(cab.decode(enc, errors="replace"), delim)
    return nome, enc, delim, cols


def _ler_pedacos(zf: zipfile.ZipFile, nome: str, enc: str, delim: str,
                 rejeitadas: dict):
    """Iterador de pedaços de um membro CSV (engine python + contagem)."""
    src = zf.open(nome)
    texto = io.TextIOWrapper(src, encoding=enc, errors="replace", newline="")
    reader = pd.read_csv(
        texto, sep=delim, chunksize=CHUNKSIZE, dtype=str,
        keep_default_na=False, on_bad_lines=_contador_rejeitadas(rejeitadas),
        engine="python", quotechar='"',
    )
    return reader


def _processar_depositantes(zf, info, crosswalk, registros, cobertura, rejeitadas, amostrar):
    """Acumula depositantes da Rede no ``registros`` (união). Retorna casados/descartados."""
    n_casados = 0
    n_descart = 0
    nome_dep = ip.achar_membro(zf, MEMBRO_DEPOSITANTES)
    if nome_dep is None:
        return n_casados, n_descart
    mdep = info.get(MEMBRO_DEPOSITANTES, {})
    enc_dep = mdep.get("encoding") or "utf-8"
    delim_dep = mdep.get("delimitador") or ","
    for chunk in _ler_pedacos(zf, nome_dep, enc_dep, delim_dep, rejeitadas):
        amostrar()
        if "numero_inpi" not in chunk.columns:
            continue
        tem_dep = "depositante" in chunk.columns
        tem_cnpj = "cgccpfdepositante" in chunk.columns
        dep_col = chunk["depositante"].astype(str) if tem_dep else None
        cnpj_col = chunk["cgccpfdepositante"].astype(str) if tem_cnpj else None
        num_col = chunk["numero_inpi"].astype(str).str.strip()
        for i in range(len(chunk)):
            num = num_col.iloc[i]
            cnpj = cnpj_col.iloc[i] if tem_cnpj else ""
            nome_depi = dep_col.iloc[i] if tem_dep else ""
            # Quando a coluna ``depositante`` está ausente, a linha só pode ser
            # da Rede via CNPJ (crosswalk); nunca a descarta antes de avaliá-lo.
            e_rede, raiz = classificar_depositante(nome_depi, cnpj, crosswalk)
            if not e_rede:
                n_descart += 1
                continue
            n_casados += 1
            if raiz is not None:
                cobertura[raiz] = cobertura.get(raiz, 0) + 1
            rec = registros.setdefault(num, {
                "numero_inpi": num, "data_deposito": "", "titulo": "",
                "resumo": "", "instituicoes": [],
            })
            instituicoes = rec.setdefault("instituicoes", [])
            # rótulo institucional: nome normalizado (raiz) ou, sem nome
            # disponível e casado por CNPJ, o próprio CNPJ em dígitos.
            etiqueta = _normalizar(nome_depi) if (nome_depi and raiz not in (None, "crosswalk_cnpj")) else ""
            if not etiqueta and raiz == "crosswalk_cnpj":
                etiqueta = _cnpj_digitos(cnpj)
            if etiqueta and etiqueta not in instituicoes:
                instituicoes.append(etiqueta)
        amostrar()
    return n_casados, n_descart


def _enriquecer_conteudo(zf, info, registros, rede_set, rejeitadas, amostrar):
    """Preenche título/resumo (first wins) dos registros da Rede."""
    nome_cont = ip.achar_membro(zf, MEMBRO_CONTEUDO)
    if nome_cont is None:
        return
    mc = info.get(MEMBRO_CONTEUDO, {})
    enc_c = mc.get("encoding") or "utf-8"
    delim_c = mc.get("delimitador") or ","
    for chunk in _ler_pedacos(zf, nome_cont, enc_c, delim_c, rejeitadas):
        amostrar()
        if "numero_inpi" not in chunk.columns:
            continue
        titulo = chunk["titulo"].astype(str) if "titulo" in chunk.columns else None
        resumo = chunk["resumo"].astype(str) if "resumo" in chunk.columns else None
        num_col = chunk["numero_inpi"].astype(str).str.strip()
        for i in range(len(chunk)):
            num = num_col.iloc[i]
            if num not in rede_set:
                continue
            rec = registros[num]
            if titulo is not None:
                v = titulo.iloc[i]
                if not rec.get("titulo") and v not in (None, "nan", ""):
                    rec["titulo"] = v
            if resumo is not None:
                v = resumo.iloc[i]
                if not rec.get("resumo") and v not in (None, "nan", ""):
                    rec["resumo"] = v
        amostrar()


def _enriquecer_bibliograficos(zf, info, registros, rede_set, rejeitadas, amostrar):
    """Preenche data_deposito (first wins) dos registros da Rede."""
    nome_bib = ip.achar_membro(zf, MEMBRO_BIBLIOGRAFICOS)
    if nome_bib is None:
        return
    mb = info.get(MEMBRO_BIBLIOGRAFICOS, {})
    enc_b = mb.get("encoding") or "utf-8"
    delim_b = mb.get("delimitador") or ","
    for chunk in _ler_pedacos(zf, nome_bib, enc_b, delim_b, rejeitadas):
        amostrar()
        if "numero_inpi" not in chunk.columns:
            continue
        data_col = chunk["data_deposito"].astype(str) if "data_deposito" in chunk.columns else None
        num_col = chunk["numero_inpi"].astype(str).str.strip()
        for i in range(len(chunk)):
            num = num_col.iloc[i]
            if num not in rede_set:
                continue
            rec = registros[num]
            if data_col is not None:
                v = data_col.iloc[i].strip()
                if not rec.get("data_deposito") and v not in ("nan", "None"):
                    rec["data_deposito"] = v
        amostrar()


# --------------------------------------------------------------------------- #
# Pipeline central
# --------------------------------------------------------------------------- #
def extrair_filtrar(zip_path: str, out_csv: str, out_json: str,
                    crosswalk_path: str | None = None,
                    validacao_path: str | None = None,
                    ) -> dict:
    """Extrai e filtra registros de PI da RFEPCT, grava CSV+relatório.

    Retorna o relatório (mesmo shapes por path, sem lançar por dados ausentes).
    """
    crosswalk_path = crosswalk_path or CROSSWALK_DEFAULT
    validacao_path = validacao_path or VALIDA_DEFAULT
    crosswalk = _carregar_crosswalk(crosswalk_path)

    rss_max = {"v": 0}
    rejeitadas = {"n": 0}
    membros_ausentes: list[str] = []

    def amostrar_rss() -> None:
        try:
            valor = psutil.Process().memory_info().rss
            if valor > rss_max["v"]:
                rss_max["v"] = valor
        except Exception:  # noqa: BLE001
            pass

    amostrar_rss()

    # contrato estável do relatório
    relatorio = {
        "veredito": "pendente",
        "n_registros_csv": 0,
        "n_casados_rede": 0,
        "n_descartados": 0,
        "linhas_rejeitadas": 0,
        "cobertura_por_raiz": {r: 0 for r in RAIZES_RFEPCT} | {"crosswalk_cnpj": 0},
        "membros_ausentes": [],
        "membros": {},
        "pico_memoria_mb": 0.0,
        "erro": None,
    }

    if not os.path.exists(zip_path):
        relatorio["erro"] = f"zip não encontrado: {zip_path}"
        _gravar_relatorio(relatorio, out_json)
        return relatorio

    try:
        zf = zipfile.ZipFile(zip_path)
    except Exception as exc:  # noqa: BLE001
        relatorio["erro"] = f"não foi possível abrir o zip: {exc}"
        _gravar_relatorio(relatorio, out_json)
        return relatorio

    # ---- schema / presença dos membros ------------------------------------ #
    info = {}
    for alvo in (MEMBRO_DEPOSITANTES, MEMBRO_CONTEUDO, MEMBRO_BIBLIOGRAFICOS):
        try:
            nome, enc, delim, cols = _abrir_membro(zf, alvo)
        except Exception as exc:  # noqa: BLE001 - membro corrompido não aborta
            membros_ausentes.append(alvo)
            info[alvo] = {
                "membro_encontrado": False, "colunas_reais": [], "colunas_ausentes": [],
                "encoding": None, "delimitador": None, "erro": str(exc),
            }
            continue
        if nome is None:
            membros_ausentes.append(alvo)
            info[alvo] = {
                "membro_encontrado": False, "colunas_reais": [], "colunas_ausentes": [],
                "encoding": None, "delimitador": None,
            }
        else:
            info[alvo] = {
                "membro_encontrado": True, "colunas_reais": cols,
                "colunas_ausentes": [], "encoding": enc, "delimitador": delim,
            }
            # drift por coluna ausente das essenciais para esta etapa
            essenciais = {
                MEMBRO_DEPOSITANTES: ("numero_inpi", "depositante", "cgccpfdepositante"),
                MEMBRO_CONTEUDO: ("numero_inpi", "titulo", "resumo"),
                MEMBRO_BIBLIOGRAFICOS: ("numero_inpi", "data_deposito"),
            }[alvo]
            info[alvo]["colunas_ausentes"] = [c for c in essenciais if c not in cols]
    relatorio["membros_ausentes"] = membros_ausentes
    relatorio["membros"] = info
    amostrar_rss()

    registros: dict[str, dict] = {}  # numero_inpi -> dados
    n_casados_rede = 0
    n_descartados = 0
    cobertura: dict[str, int] = {r: 0 for r in RAIZES_RFEPCT}
    cobertura["crosswalk_cnpj"] = 0
    erros: list[str] = []

    try:
        n_casados_rede, n_descartados = _processar_depositantes(
            zf, info, crosswalk, registros, cobertura, rejeitadas, amostrar_rss
        )
    except Exception as exc:  # noqa: BLE001
        erros.append(f"PATENTES_DEPOSITANTES: {exc}")

    rede_set = set(registros.keys())

    try:
        _enriquecer_conteudo(zf, info, registros, rede_set, rejeitadas, amostrar_rss)
    except Exception as exc:  # noqa: BLE001
        erros.append(f"PATENTES_CONTEUDO: {exc}")

    try:
        _enriquecer_bibliograficos(zf, info, registros, rede_set, rejeitadas, amostrar_rss)
    except Exception as exc:  # noqa: BLE001
        erros.append(f"PATENTES_DADOS_BIBLIOGRAFICOS: {exc}")

    try:
        zf.close()
    except Exception:  # noqa: BLE001
        pass

    amostrar_rss()

    # ---- consolidar CSV ---------------------------------------------------- #
    linhas_csv = []
    for num in registros:
        rec = registros[num]
        linhas_csv.append({
            "numero_inpi": rec.get("numero_inpi", ""),
            "data_deposito": rec.get("data_deposito", ""),
            "titulo": rec.get("titulo", ""),
            "resumo": rec.get("resumo", ""),
            "instituicoes": "|".join(rec.get("instituicoes", [])),
        })
    df = pd.DataFrame(linhas_csv, columns=["numero_inpi", "data_deposito",
                                           "titulo", "resumo", "instituicoes"])
    if not out_csv.startswith("nul"):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001 - falha de escrita nunca
            erros.append(f"falha ao gravar CSV: {exc}")  #   impede o relatório

    n_registros_csv = len(df)
    pico_mb = round(rss_max["v"] / (1024 * 1024), 3)

    # ---- veredito ---------------------------------------------------------- #
    fr_aceito = _fr1_aceito(validacao_path)
    if erros:
        relatorio["erro"] = "; ".join(erros)
    erro_fatal = relatorio.get("erro") is not None
    veredito = "ok" if (fr_aceito and not erro_fatal) else "pendente"

    relatorio.update({
        "veredito": veredito,
        "n_registros_csv": n_registros_csv,
        "n_casados_rede": n_casados_rede,
        "n_descartados": n_descartados,
        "linhas_rejeitadas": rejeitadas["n"],
        "cobertura_por_raiz": cobertura,
        "membros_ausentes": membros_ausentes,
        "pico_memoria_mb": pico_mb,
        "fr1_aceito": fr_aceito,
        "crosswalk_cnpj": {"presente": len(crosswalk) > 0, "n_cnpj_validos": len(crosswalk)},
        "limite_memoria_mb": LIMITE_MEMORIA_MB,
        "dentro_do_limite": pico_mb < LIMITE_MEMORIA_MB,
    })

    _gravar_relatorio(relatorio, out_json)
    return relatorio


def _fr1_aceito(validacao_path: str) -> bool:
    """Lê o gate FR-1 (Story 1.2) e devolve ``fr1.aceito``; ausente ⇒ False."""
    if not validacao_path or not os.path.exists(validacao_path):
        return False
    try:
        with open(validacao_path, encoding="utf-8") as f:
            dados = json.load(f)
        return bool((dados.get("fr1") or {}).get("aceito"))
    except Exception:  # noqa: BLE001
        return False


def _gravar_relatorio(relatorio: dict, out_json: str) -> None:
    if not out_json.startswith("nul"):
        os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(relatorio, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Extrai e filtra registros de PI da RFEPCT do dump do INPI (Story 1.3)."
    )
    ap.add_argument("--zip", default=ZIP_DEFAULT, help="caminho do dump patentes.zip")
    ap.add_argument("--csv", default=CSV_DEFAULT, help="CSV de saída (utf-8-sig)")
    ap.add_argument("--relatorio", default=REPORT_DEFAULT, help="JSON de relatório")
    ap.add_argument("--crosswalk", default=CROSSWALK_DEFAULT, help="JSON de crosswalk CNPJ")
    args = ap.parse_args(argv)

    relatorio = extrair_filtrar(
        args.zip, args.csv, args.relatorio, crosswalk_path=args.crosswalk
    )
    sys.stdout.write(
        f"veredito={relatorio['veredito']} n_registros_csv={relatorio['n_registros_csv']} "
        f"n_casados_rede={relatorio['n_casados_rede']} "
        f"pico_memoria_mb={relatorio['pico_memoria_mb']} "
        f"relatorio={args.relatorio}\n"
    )
    return 0 if relatorio["veredito"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
