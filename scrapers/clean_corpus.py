"""Estrutura, limpa e aplica a Janela temporal ao corpus da RFEPCT (Story 1.5).

Unifica as duas fontes geradas nas stories 1.3 e 1.4 num único corpus limpo em
``data/processed/corpus_rfepct.csv`` com 9 colunas (fonte, tipo, data,
instituicao, texto_límpido, na_janela, data_ausente, texto_fallback_titulo,
instituicoes_aux) e grava ``data/processed/relatorio_corpus.json`` com contagens
por fonte/tipo/instituição, cobertura da Janela 2019–2026 e trilha de auditoria
(duplicatas removidas, erros registrados).

Fontes lidas como idempotentes (merge por chave canônica): re-execução sobre o
mesmo corpus regenera o mesmo resultado, sem duplicatas.

Nenhum LLM, nenhuma rede: processamento 100% local/offline.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
from datetime import datetime

try:
    from bs4 import BeautifulSoup
    _TEM_BS4 = True
except Exception:  # noqa: BLE001 - bs4 opcional (fallback regex)
    _TEM_BS4 = False

__version__ = "1.5.0"

PROJETO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.dirname(os.path.abspath(__file__)) != PROJETO_RAIZ and PROJETO_RAIZ not in sys.path:
    sys.path.insert(0, PROJETO_RAIZ)

DIR_PROCESSED = os.path.join(PROJETO_RAIZ, "data", "processed")
DIR_RAW = os.path.join(PROJETO_RAIZ, "data", "raw")
INPI_DEFAULT = os.path.join(DIR_PROCESSED, "inpi_rfepct_records.csv")
NOTICIAS_DEFAULT = os.path.join(DIR_RAW, "noticias_rfepct.json")
OUT_CSV_DEFAULT = os.path.join(DIR_PROCESSED, "corpus_rfepct.csv")
OUT_RELATORIO_DEFAULT = os.path.join(DIR_PROCESSED, "relatorio_corpus.json")

JANELA_INICIO = 2019
JANELA_FIM = 2026

# Delimitador e encoding usados na leitura/escrita do CSV de entrada/saída.
CSV_DELIMITER = ","
CSV_ENCODING = "utf-8-sig"

# Regex de tag HTML residual para o fallback sem bs4.
_RE_TAG = re.compile(r"<[^>]+>")

# Colunas do corpus final.
COLUNAS = ["fonte", "tipo", "data", "instituicao", "texto_límpido",
           "na_janela", "data_ausente", "texto_fallback_titulo", "instituicoes_aux"]


# --------------------------------------------------------------------------- #
# Limpeza de texto
# --------------------------------------------------------------------------- #
def limpar_html(texto: str | None) -> str:
    """Remove tags HTML residuais e decodifica entidades.

    Usa ``bs4`` quando disponível (texto → texto plano, separador espaço),
    senão regex fallback. Decodifica entidades (``&nbsp;``, ``&amp;``, etc.),
    normaliza múltiplos espaços e faz trim. Nunca trunca o conteúdo (frases
    completas preservadas para o recorte social semântico do Epic 2).
    """
    if not texto:
        return ""
    if _TEM_BS4:
        soup = BeautifulSoup(texto, "html.parser")
        resultado = soup.get_text(separator=" ", strip=True)
    else:
        resultado = _RE_TAG.sub(" ", texto)
    # decodifica entidades (GetText já decodifica com bs4, mas fallback/norm extra)
    try:
        resultado = html.unescape(resultado)
    except Exception:  # noqa: BLE001
        pass
    # normaliza múltiplos espaços e trim
    resultado = re.sub(r"\s+", " ", resultado).strip()
    return resultado


def _normalizar_iso(data: str | None) -> str:
    """Normaliza uma data para ISO ``YYYY-MM-DD`` (strito), ou retorna ``""``.

    Aceita já-ISO ou datas brasileiras ``dd/mm/yyyy``. Valores vazios/nulos ou
    não-parseáveis ⇒ ``""`` (item sem data).
    """
    if not data:
        return ""
    valor = str(data).strip()
    if not valor:
        return ""
    # ISO YYYY-MM-DD (possivelmente com hora)
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", valor)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return ""
        return dt.date().isoformat()
    # brasileiro dd/mm/yyyy
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})", valor)
    if m:
        dd, mm, aaaa = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dt = datetime(aaaa, mm, dd)
        except ValueError:
            return ""
        return dt.date().isoformat()
    return ""


def _na_janela(ano: int | None) -> bool:
    """True quando o ano está dentro da Janela 2019–2026 inclusive."""
    if ano is None:
        return False
    return JANELA_INICIO <= ano <= JANELA_FIM


def _e_fronteira_temporal(data_iso: str) -> tuple[int | None, bool]:
    """Retorna ``(ano, na_janela)`` a partir de ISO; sem data ⇒ ``(None, False)``."""
    if not data_iso:
        return None, False
    try:
        ano = int(data_iso[:4])
    except ValueError:
        return None, False
    return ano, _na_janela(ano)


# --------------------------------------------------------------------------- #
# Fonte INPI
# --------------------------------------------------------------------------- #
def _processar_inpi(linha: dict, itens: dict, ordem: int) -> bool:
    """Converte uma linha INPI em item do corpus e acumula no ``itens`` (merge).

    ``ordem`` é o índice sequencial (0-based) da linha dentro da fonte, usado
    para desambiguar a chave de fallback quando ``numero_inpi`` está vazio
    (itens distintos degenerados não colidem). Retorna ``True`` quando o item
    foi adicionado, ``False`` quando foi descartado por chave já presente.
    """
    numero = (linha.get("numero_inpi") or "").strip()
    data_raw = linha.get("data_deposito") or ""
    data_iso = _normalizar_iso(data_raw)
    ano, na_janela = _e_fronteira_temporal(data_iso)

    titulo = limpar_html(linha.get("titulo") or "")
    resumo = limpar_html(linha.get("resumo") or "")
    texto = (titulo + " " + resumo).strip() if (titulo or resumo) else ""

    instituicoes_raw = linha.get("instituicoes") or ""
    partes = [p.strip() for p in instituicoes_raw.split("|") if p.strip()]
    instituicao = partes[0] if partes else ""
    instituicoes_aux = "|".join(partes[1:]) if len(partes) > 1 else ""

    item = {
        "fonte": numero,
        "tipo": "inpi",
        "data": data_iso,
        "instituicao": instituicao,
        "texto_límpido": texto,
        "na_janela": na_janela,
        "data_ausente": not data_iso,
        "texto_fallback_titulo": False,
        "instituicoes_aux": instituicoes_aux,
    }
    chave = ("inpi", numero) if numero else ("inpi_fallback", ordem, data_iso, titulo)
    if chave not in itens:
        itens[chave] = item
        return True
    return False


# --------------------------------------------------------------------------- #
# Fonte notícias (portais)
# --------------------------------------------------------------------------- #
def _processar_noticia(item: dict, itens: dict, ordem: int) -> bool:
    """Converte um item de notícia em item do corpus e acumula no ``itens``.

    ``ordem`` é o índice sequencial (0-based) do item dentro da fonte, usado
    para desambiguar a chave de fallback quando ``url`` está vazio. Retorna
    ``True`` quando adicionado, ``False`` quando descartado por chave já presente.
    """
    url = (item.get("url") or "").strip()
    data_raw = item.get("data")
    data_iso = _normalizar_iso(data_raw)
    ano, na_janela = _e_fronteira_temporal(data_iso)

    titulo = limpar_html(item.get("titulo") or "")
    texto_raw = limpar_html(item.get("texto") or "")
    fallback = False
    if not texto_raw and titulo:
        texto_raw = titulo
        fallback = True

    instituicao = (item.get("instituicao") or "").strip()

    item_out = {
        "fonte": url,
        "tipo": "noticia",
        "data": data_iso,
        "instituicao": instituicao,
        "texto_límpido": texto_raw,
        "na_janela": na_janela,
        "data_ausente": not data_iso,
        "texto_fallback_titulo": fallback,
        "instituicoes_aux": "",
    }
    chave = ("noticia", url) if url else ("noticia_fallback", ordem, data_iso, titulo)
    if chave not in itens:
        itens[chave] = item_out
        return True
    return False


# --------------------------------------------------------------------------- #
# Pipeline central
# --------------------------------------------------------------------------- #
def limpar_corpus(inpi_csv: str, noticias_json: str, out_csv: str,
                  out_relatorio: str) -> dict:
    """Unifica INPI + notícias num corpus limpo e grava CSV + relatório.

    Retorna o relatório (dict). Erros de leitura/escrita registrados no
    relatório, nunca abortam o pipeline (drifts registrados, não fatais).
    """
    itens: dict = {}
    erros: list[str] = []
    n_dup_inpi = 0
    n_dup_noticias = 0

    # ---- fonte INPI ------------------------------------------------------ #
    n_inpi = 0
    if os.path.exists(inpi_csv):
        try:
            with open(inpi_csv, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f, delimiter=CSV_DELIMITER)
                for ordem, linha in enumerate(reader):
                    if not _processar_inpi(linha, itens, ordem):
                        n_dup_inpi += 1
                    n_inpi += 1
        except Exception as exc:  # noqa: BLE001
            erros.append(f"inpi_csv: {exc}")
    else:
        erros.append(f"inpi_csv não encontrado: {inpi_csv}")

    # ---- fonte notícias -------------------------------------------------- #
    n_noticias = 0
    if os.path.exists(noticias_json):
        try:
            with open(noticias_json, encoding="utf-8") as f:
                dados = json.load(f)
            if isinstance(dados, list):
                for ordem, item in enumerate(dados):
                    if isinstance(item, dict):
                        if not _processar_noticia(item, itens, ordem):
                            n_dup_noticias += 1
                        n_noticias += 1
                    else:
                        erros.append(
                            f"noticias_json[{ordem}]: item não-dict ignorado "
                            f"(tipo={type(item).__name__})"
                        )
            else:
                erros.append("noticias_json: formato inesperado (esperado lista top-level)")
        except Exception as exc:  # noqa: BLE001
            erros.append(f"noticias_json: {exc}")
    else:
        erros.append(f"noticias_json não encontrado: {noticias_json}")

    # ---- consolidar e ordenar -------------------------------------------- #
    # ordem: fontes estáveis (INPI por numero) depois noticias por url; estável
    # e determinística para idempotência total do CSV.
    linhas = []
    for chave, item in itens.items():
        linhas.append(item)
    linhas.sort(key=lambda r: (r["tipo"], r["fonte"]))

    # ---- gravar CSV ------------------------------------------------------ #
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUNAS, delimiter=CSV_DELIMITER)
            writer.writeheader()
            for item in linhas:
                writer.writerow(item)
    except Exception as exc:  # noqa: BLE001
        erros.append(f"falha ao gravar CSV: {exc}")

    # ---- estatísticas ---------------------------------------------------- #
    n_total = len(linhas)
    n_janela = sum(1 for item in linhas if item["na_janela"])
    n_fora_janela = sum(1 for item in linhas if not item["na_janela"] and not item["data_ausente"])
    n_sem_data = sum(1 for item in linhas if item["data_ausente"])

    n_por_tipo: dict[str, int] = {}
    n_por_instituicao: dict[str, int] = {}
    for item in linhas:
        tipo = item["tipo"]
        n_por_tipo[tipo] = n_por_tipo.get(tipo, 0) + 1
        inst = item["instituicao"] or "(sem instituicao)"
        n_por_instituicao[inst] = n_por_instituicao.get(inst, 0) + 1

    n_fallback_titulo = sum(1 for item in linhas if item["texto_fallback_titulo"])
    n_multi_instituicoes = sum(1 for item in linhas if item["instituicoes_aux"])
    n_duplicatas_removidas = n_dup_inpi + n_dup_noticias

    relatorio = {
        "n_total": n_total,
        "n_janela": n_janela,
        "n_fora_janela": n_fora_janela,
        "n_sem_data": n_sem_data,
        "n_inpi": n_inpi,
        "n_noticias": n_noticias,
        "n_duplicatas_removidas": n_duplicatas_removidas,
        "n_dup_inpi": n_dup_inpi,
        "n_dup_noticias": n_dup_noticias,
        "n_por_tipo": n_por_tipo,
        "n_por_instituicao": n_por_instituicao,
        "cobertura_janela": {
            "inicio": JANELA_INICIO,
            "fim": JANELA_FIM,
            "inclusive": True,
        },
        "n_texto_fallback_titulo": n_fallback_titulo,
        "n_multi_instituicoes": n_multi_instituicoes,
        "erros": erros,
        "metadados": {
            "versao_modulo": __version__,
            "inpi_csv": inpi_csv,
            "noticias_json": noticias_json,
            "out_csv": out_csv,
            "out_relatorio": out_relatorio,
        },
    }

    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_relatorio)) or ".", exist_ok=True)
        with open(out_relatorio, "w", encoding="utf-8") as f:
            json.dump(relatorio, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        erros.append(f"falha ao gravar relatório: {exc}")
        relatorio["erros"] = erros

    return relatorio


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Estrutura, limpa e aplica a Janela 2019–2026 ao corpus da RFEPCT (Story 1.5)."
    )
    ap.add_argument("--inpi", default=INPI_DEFAULT,
                    help="CSV INPI de entrada (inpi_rfepct_records.csv)")
    ap.add_argument("--noticias", default=NOTICIAS_DEFAULT,
                    help="JSON de notícias de entrada (noticias_rfepct.json)")
    ap.add_argument("--out", default=OUT_CSV_DEFAULT,
                    help="CSV de saída (corpus_rfepct.csv)")
    ap.add_argument("--relatorio", default=OUT_RELATORIO_DEFAULT,
                    help="JSON de relatório (relatorio_corpus.json)")
    args = ap.parse_args(argv)

    rel = limpar_corpus(args.inpi, args.noticias, args.out, args.relatorio)
    sys.stdout.write(
        f"n_total={rel['n_total']} n_janela={rel['n_janela']} "
        f"n_fora_janela={rel['n_fora_janela']} n_sem_data={rel['n_sem_data']} "
        f"relatorio={args.relatorio}\n"
    )
    # exit 0 apenas quando não há erros registrados; exit 1 quando há erros
    # (contagens coerentes com os módulos irmãos inpi_extrair/inpi_parser).
    return 0 if not rel["erros"] else 1


if __name__ == "__main__":
    sys.exit(main())
