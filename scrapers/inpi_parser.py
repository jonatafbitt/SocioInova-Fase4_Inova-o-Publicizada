"""Protótipo de acesso e validação do dump de Dados Abertos do INPI (Story 1.2).

Valida o endpoint HTTPS de Dados Abertos, garante/descompacta ``patentes.zip``,
faz parsing em pedaços (``pandas`` ``chunksize`` sobre ``TextIOWrapper``) e gera
o relatório do gate FR-1 em ``data/raw/inpi_validacao.json``.

Re-derivado conforme a Story 1.2 (spec aprovado). Correções sobre a v0 reprovada:
- gate de data exige *cobertura anual* 2019–2026 e ``pct_na_janela >= LIMIAR``
  (não apenas uma linha dentro da Janela);
- ``min_bruto``/``max_bruto`` reportam o *universo* de linhas parseáveis,
  separados dos limites da Janela (``min_janela``/``max_janela``);
- leitura do cabeçalho de forma limitada (stream até ``\\n``, máx. 256 KB),
  sem ``ZipFile.read`` (que carregaria o membro inteiro);
- encoding via BOM (``utf-8-sig``) com fallback ``cp1252`` e drift reportado;
- pico de memória por RSS (``psutil``), ``tracemalloc`` como referência
  secundária;
- veredito quadrivalente: endpoint HTTPS (host + schema na URL resolvida) **E**
  schema presente **E** data utilizável na Janela (cobertura anual + pct >= 10%
  + proporcao_parseaveis >= 50%) **E** memória dentro do limite => ``aprovado``;
  qualquer falha => ``pendente`` (adiando FR-1/Story 1.3).

Nenhum LLM, nenhum dado sai da máquina: download/parse/validação mecânica.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tracemalloc
import zipfile
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import psutil
import requests

__version__ = "1.2.0"

PROJETO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RAW = os.path.join(PROJETO_RAIZ, "data", "raw")
REPORT_PATH = os.path.join(DIR_RAW, "inpi_validacao.json")

ENDPOINT = "https://dadosabertos.inpi.gov.br/"
URL_ZIP = "https://dadosabertos.inpi.gov.br/download/patentes/patentes.zip"
HOST_DADOS_ABERTOS = "dadosabertos.inpi.gov.br"

COLUNAS_DOCUMENTADAS = [
    "codigo_interno",
    "numero_inpi",
    "data_deposito",
    "data_protocolo",
    "data_publicacao",
    "numero_pct",
    "numero_wo",
    "data_publicacao_wo",
    "data_entrada_fase_nacional",
]
ENCODING_DOCUMENTADO = "utf-8"
DELIMITADOR_DOCUMENTADO = ";"

JANELA_INICIO = 2019
JANELA_FIM = 2026
LIMIAR_PCT_JANELA = 0.10
LIMIAR_PROPR_PARSEAVEIS = 0.50
ANO_MIN_PLAUSIVEL = 1800
ANO_MAX_PLAUSIVEL = 2100
LIMITE_MEMORIA_MB = 2048
CHUNKSIZE = 200000
TIMEOUT = 600
MAX_HEADER_BYTES = 256 * 1024
DELIMITADORES_CANDIDATOS = [",", ";", "|", "\t"]

_RE_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
def validar_endpoint() -> dict:
    """Confere alcançabilidade HTTPS do endpoint: host + schema na URL RESOLVIDA (pós-redirect) e status < 400."""
    try:
        resp = requests.get(ENDPOINT, timeout=60)
        url_final = resp.url
        parsed = urlparse(url_final)
        netloc = (parsed.netloc or "").lower()
        https_ok = parsed.scheme == "https"
        host_ok = netloc == HOST_DADOS_ABERTOS or netloc.endswith("." + HOST_DADOS_ABERTOS)
        status_ok = resp.status_code < 400
        alcancavel = https_ok and host_ok and status_ok
        return {
            "url_requisitada": ENDPOINT,
            "url_resolvida": url_final,
            "scheme": parsed.scheme,
            "host": parsed.netloc,
            "https_ok": https_ok,
            "host_ok": host_ok,
            "status_code": resp.status_code,
            "status_ok": status_ok,
            "alcancavel": alcancavel,
            "erro": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "url_requisitada": ENDPOINT,
            "url_resolvida": None,
            "scheme": None,
            "host": None,
            "https_ok": False,
            "host_ok": False,
            "status_code": None,
            "status_ok": False,
            "alcancavel": False,
            "erro": str(exc),
        }


# --------------------------------------------------------------------------- #
# Download atômico + integridade
# --------------------------------------------------------------------------- #
def _zip_integro(caminho: str, tamanho_esperado: int | None) -> tuple[bool, str | None]:
    """Valida integridade de um zip local: tamanho (vs Content-Length) e CRC."""
    if tamanho_esperado is not None:
        try:
            if os.path.getsize(caminho) != tamanho_esperado:
                return False, (
                    f"tamanho local {os.path.getsize(caminho)} != Content-Length "
                    f"{tamanho_esperado}"
                )
        except OSError as exc:
            return False, str(exc)
    try:
        with zipfile.ZipFile(caminho) as zf:
            if zf.testzip() is not None:
                return False, "testzip: membro com CRC inválido"
    except zipfile.BadZipFile as exc:
        return False, f"BadZipFile: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, None


def garantir_dump(dump: str, url_zip: str, dir_raw: str, progresso: bool = True) -> dict:
    """Garante o dump local, reutilizando apenas se íntegro; senão baixa atômico."""
    os.makedirs(dir_raw, exist_ok=True)
    destino = os.path.join(dir_raw, dump)
    part = destino + ".part"

    tamanho_esperado: int | None = None
    try:
        resp_head = requests.head(url_zip, timeout=60)
        cab = resp_head.headers.get("Content-Length", "")
        if cab.strip().lstrip("-").isdigit():
            tamanho_esperado = int(cab)
    except Exception:  # noqa: BLE001
        tamanho_esperado = None

    if os.path.exists(destino) and os.path.getsize(destino) > 0:
        integro, erro_int = _zip_integro(destino, tamanho_esperado)
        if integro:
            return {
                "dump": dump,
                "url": url_zip,
                "baixado": True,
                "era_local": True,
                "integro": True,
                "bytes": os.path.getsize(destino),
                "content_length_esperado": tamanho_esperado,
                "caminho": destino,
            }
        # zip local corrompido/incoerente => redownload (nunca reutilizar como prova)
    try:
        with requests.get(url_zip, stream=True, timeout=TIMEOUT) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0) or 0)
            baixados = 0
            with open(part, "wb") as f:
                for pedaco in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(pedaco)
                    baixados += len(pedaco)
                    if progresso and total:
                        sys.stderr.write(f"\rbaixando {dump}: {baixados / total:.1%}")
            if progresso:
                sys.stderr.write("\n")
        os.replace(part, destino)  # rename atômico: nunca deixa .part parcial como prova
        integro, erro_int = _zip_integro(destino, tamanho_esperado)
        if not integro:
            return {
                "dump": dump,
                "url": url_zip,
                "baixado": True,
                "era_local": False,
                "integro": False,
                "erro": erro_int,
                "bytes": os.path.getsize(destino),
                "caminho": destino,
            }
        return {
            "dump": dump,
            "url": url_zip,
            "baixado": True,
            "era_local": False,
            "integro": True,
            "bytes": os.path.getsize(destino),
            "content_length_esperado": tamanho_esperado,
            "caminho": destino,
        }
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass
        return {
            "dump": dump,
            "url": url_zip,
            "baixado": False,
            "integro": False,
            "erro": str(exc),
            "caminho": destino,
        }


# --------------------------------------------------------------------------- #
# CSV header (stream limitado) + encoding/delimitador
# --------------------------------------------------------------------------- #
def ler_cabecalho(zf: zipfile.ZipFile, nome: str, max_bytes: int = MAX_HEADER_BYTES) -> tuple[bytes, bool]:
    """Lê a primeira linha do CSV de forma limitada. Retorna (bytes, truncado).

    Stream até ``\\n`` com limite de ``max_bytes`` bytes. ``truncado=True`` quando
    o limite é atingido sem encontrar uma quebra de linha (cabeçalho estourado).
    """
    buf = bytearray()
    excedeu = False
    with zf.open(nome) as src:
        while True:
            pedaco = src.read(4096)
            if not pedaco:
                break
            buf += pedaco
            if b"\n" in pedaco:
                break
            if len(buf) >= max_bytes:
                excedeu = True
                break
    nl = buf.find(b"\n")
    if nl != -1:
        buf = buf[:nl]
    elif excedeu or len(buf) >= max_bytes:
        excedeu = True
        buf = buf[:max_bytes]
    if buf.endswith(b"\r"):
        buf = buf[:-1]
    return bytes(buf), excedeu


def detectar_encoding(cabecalho: bytes) -> str:
    """Detecta encoding pelo BOM, com fallback ``cp1252``; nunca assume utf-8 cegamente."""
    if cabecalho.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        cabecalho.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


def detectar_delimitador(cabecalho: bytes, encoding: str) -> str:
    """Heurística de delimitador pela linha do cabeçalho; fallback ao documentado (``;``) quando nenhum candidato tem contagem > 0."""
    linha = cabecalho.decode(encoding, errors="replace")
    contagens = {d: linha.count(d) for d in DELIMITADORES_CANDIDATOS}
    melhor = max(contagens, key=contagens.get)
    if contagens[melhor] == 0:
        return DELIMITADOR_DOCUMENTADO
    return melhor


def achar_membro(zf: zipfile.ZipFile, membro_alvo: str) -> str | None:
    """Localiza o membro do zip cujo basename iguala ``membro_alvo`` (case-insensitive)."""
    alvo = membro_alvo.lower()
    for nome in zf.namelist():
        if nome.lower() == alvo or os.path.basename(nome).lower() == alvo:
            return nome
    return None


def _limpar_cabecalho(linha: str, delimitador: str) -> list[str]:
    return [c.strip().strip('"').strip() for c in linha.split(delimitador)]


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def validar_schema(zf: zipfile.ZipFile, membro: str) -> tuple[dict, list[str]]:
    """Valida as colunas documentadas no cabeçalho e reporta drift (encoding/delimitador/colunas)."""
    nome = achar_membro(zf, membro)
    if nome is None:
        return {
            "membro_encontrado": False,
            "cabecalho_truncado": False,
            "colunas_esperadas": {c: "ausente" for c in COLUNAS_DOCUMENTADAS},
            "colunas_reais": [],
            "colunas_ausentes": list(COLUNAS_DOCUMENTADAS),
            "colunas_extras": [],
            "colunas_extras_drift": [],
            "delimitador_detectado": None,
            "delimitador_documentado": DELIMITADOR_DOCUMENTADO,
            "drift_delimitador": None,
            "encoding": None,
            "encoding_documentado": ENCODING_DOCUMENTADO,
            "drift_encoding": True,
            "schema_ok": False,
        }, []

    cabecalho, cabecalho_truncado = ler_cabecalho(zf, nome)
    encoding = detectar_encoding(cabecalho)
    delimitador = detectar_delimitador(cabecalho, encoding)
    linha = cabecalho.decode(encoding, errors="replace")
    colunas_reais = _limpar_cabecalho(linha, delimitador)

    presentes = {c: ("presente" if c in colunas_reais else "ausente") for c in COLUNAS_DOCUMENTADAS}
    ausentes = [c for c in COLUNAS_DOCUMENTADAS if c not in colunas_reais]
    extras = [c for c in colunas_reais if c not in COLUNAS_DOCUMENTADAS]
    drift_encoding = encoding not in ("utf-8", "utf-8-sig")

    return {
        "membro_encontrado": True,
        "cabecalho_truncado": cabecalho_truncado,
        "colunas_esperadas": presentes,
        "colunas_reais": colunas_reais,
        "colunas_ausentes": ausentes,
        "colunas_extras": extras,
        "colunas_extras_drift": extras,
        "delimitador_detectado": delimitador,
        "delimitador_documentado": DELIMITADOR_DOCUMENTADO,
        "drift_delimitador": delimitador != DELIMITADOR_DOCUMENTADO,
        "encoding": encoding,
        "encoding_documentado": ENCODING_DOCUMENTADO,
        "drift_encoding": drift_encoding,
        "schema_ok": not ausentes,
    }, colunas_reais


# --------------------------------------------------------------------------- #
# Dados (data_deposito)
# --------------------------------------------------------------------------- #
def parsear_data_iso(celula: str) -> tuple[datetime | None, bool]:
    """Parsing estrito de ``data_deposito`` em ISO ``YYYY-MM-DD``.

    Retorna ``(datetime|None, plausivel)``. Só é parseável se a célula (strip)
    for ISO ``YYYY-MM-DD`` (len >= 10; posição 10 vazia/``T``/espaço + hora
    opcional) via regex+``strptime``. ``plausivel`` = ano em [1800, 2100].
    """
    valor = celula.strip()
    if len(valor) < 10:
        return None, False
    m = _RE_ISO.match(valor)
    if not m:
        return None, False
    fim = valor[10:11] if len(valor) > 10 else ""
    if fim not in ("", "T", " "):
        return None, False
    try:
        dt = datetime.strptime(valor[:10], "%Y-%m-%d")
    except ValueError:
        return None, False
    ano = int(m.group(1))
    return dt, (ANO_MIN_PLAUSIVEL <= ano <= ANO_MAX_PLAUSIVEL)


def parsear_e_validar_data(
    zf: zipfile.ZipFile, membro: str, delimitador: str,
    encoding: str = "utf-8", amostrar=None,
) -> dict:
    """Parsing em pedaços de ``data_deposito``: cobertura anual, Janela e universo."""
    nome = achar_membro(zf, membro)
    if nome is None:
        return {**_data_default(), "erro": "membro não encontrado"}

    info = zf.getinfo(nome)
    n_linhas = 0
    n_com_data = 0
    n_parseaveis = 0
    n_na_janela = 0
    n_nao_plausiveis = 0
    cobertura: dict[int, int] = {}
    min_b = max_b = min_jan = max_jan = None

    with zf.open(nome) as src:
        texto = io.TextIOWrapper(src, encoding=encoding, errors="replace", newline="")
        reader = pd.read_csv(
            texto,
            sep=delimitador,
            chunksize=CHUNKSIZE,
            dtype=str,
            keep_default_na=False,
            on_bad_lines="warn",
            quotechar='"',
            low_memory=False,
        )
        for pedaco in reader:
            n_linhas += len(pedaco)
            if "data_deposito" not in pedaco.columns:
                return {
                    **_data_default(),
                    "erro": "coluna ausente no parsing",
                    "n_linhas_amostradas": n_linhas,
                    "bytes_membro": info.file_size,
                }
            col = pedaco["data_deposito"].astype(str).str.strip()
            n_com_data += int(col.ne("").sum())
            passo = max(1, len(col) // 10)
            for i, valor in enumerate(col):
                if amostrar is not None and (i % passo == 0 or i == len(col) - 1):
                    amostrar()
                if not valor:
                    continue
                dt, plausivel = parsear_data_iso(valor)
                if dt is None:
                    continue
                n_parseaveis += 1
                if not plausivel:
                    n_nao_plausiveis += 1
                if min_b is None or dt < min_b:
                    min_b = dt
                if max_b is None or dt > max_b:
                    max_b = dt
                cobertura[dt.year] = cobertura.get(dt.year, 0) + 1
                if JANELA_INICIO <= dt.year <= JANELA_FIM:
                    n_na_janela += 1
                    if min_jan is None or dt < min_jan:
                        min_jan = dt
                    if max_jan is None or dt > max_jan:
                        max_jan = dt
            if amostrar is not None:
                amostrar()

    for ano in range(JANELA_INICIO, JANELA_FIM + 1):
        cobertura.setdefault(ano, 0)

    pct = (n_na_janela / n_linhas) if n_linhas else 0.0
    proporcao_parseaveis = (n_parseaveis / n_linhas) if n_linhas else 0.0
    anos_janela_presentes = all(cobertura.get(ano, 0) > 0 for ano in range(JANELA_INICIO, JANELA_FIM + 1))
    representatividade_ok = (
        anos_janela_presentes
        and pct >= LIMIAR_PCT_JANELA
        and proporcao_parseaveis >= LIMIAR_PROPR_PARSEAVEIS
    )

    return {
        "campo": "data_deposito",
        "parseavel": n_parseaveis > 0,
        "erro": None,
        "disponibilidade": round((n_com_data / n_linhas), 6) if n_linhas else 0.0,
        "proporcao_parseaveis": round(proporcao_parseaveis, 6),
        "n_linhas_amostradas": n_linhas,
        "n_com_data": n_com_data,
        "n_parseaveis": n_parseaveis,
        "n_na_janela": n_na_janela,
        "n_nao_plausiveis": n_nao_plausiveis,
        "limiar_proporcao_parseaveis": LIMIAR_PROPR_PARSEAVEIS,
        "pct_na_janela": round(pct, 6),
        "cobertura_por_ano": {str(ano): cobertura[ano] for ano in sorted(cobertura)},
        "min_bruto": min_b.date().isoformat() if min_b else None,
        "max_bruto": max_b.date().isoformat() if max_b else None,
        "min_janela": min_jan.date().isoformat() if min_jan else None,
        "max_janela": max_jan.date().isoformat() if max_jan else None,
        "janela_2019_2026": bool(anos_janela_presentes),
        "representatividade_ok": bool(representatividade_ok),
        "qualidade": "usavel" if representatividade_ok else "ruim",
        "bytes_membro": info.file_size,
    }


# --------------------------------------------------------------------------- #
# Depositantes (crosswalk 1.3)
# --------------------------------------------------------------------------- #
def validar_depositantes(zf: zipfile.ZipFile, membro: str) -> dict:
    """Valida por presença de coluna o membro de depositantes (crosswalk 1.3)."""
    nome = achar_membro(zf, membro)
    if nome is None:
        return {"membro_encontrado": False, "coluna_cgccpfdepositante_presente": False, "colunas": []}
    cabecalho, _ = ler_cabecalho(zf, nome)
    encoding = detectar_encoding(cabecalho)
    delimitador = detectar_delimitador(cabecalho, encoding)
    linha = cabecalho.decode(encoding, errors="replace")
    colunas = _limpar_cabecalho(linha, delimitador)
    return {
        "membro_encontrado": True,
        "coluna_cgccpfdepositante_presente": "cgccpfdepositante" in colunas,
        "colunas": colunas,
    }


# --------------------------------------------------------------------------- #
# Veredito quadrivalente
# --------------------------------------------------------------------------- #
def computar_veredito(endpoint_ok: bool, schema_ok: bool, data_ok: bool,
                      memoria_ok: bool = True) -> str:
    """Aprovado SOMENTE quando endpoint E schema E data E memória forem verdadeiros."""
    return "aprovado" if (endpoint_ok and schema_ok and data_ok and memoria_ok) else "pendente"


def _schema_default() -> dict:
    return {
        "membro_encontrado": False,
        "cabecalho_truncado": False,
        "colunas_esperadas": {c: "ausente" for c in COLUNAS_DOCUMENTADAS},
        "colunas_reais": [],
        "colunas_ausentes": list(COLUNAS_DOCUMENTADAS),
        "colunas_extras": [],
        "colunas_extras_drift": [],
        "delimitador_detectado": None,
        "delimitador_documentado": DELIMITADOR_DOCUMENTADO,
        "drift_delimitador": None,
        "encoding": None,
        "encoding_documentado": ENCODING_DOCUMENTADO,
        "drift_encoding": True,
        "schema_ok": False,
    }


def _data_default() -> dict:
    """Contrato B6 da seção ``data``: MESMO conjunto de chaves em todo path.

    Lista fixa (successo e erro): ``campo``, ``parseavel``, ``erro``,
    ``disponibilidade``, ``proporcao_parseaveis``, ``limiar_proporcao_parseaveis``,
    ``n_linhas_amostradas``, ``n_com_data``, ``n_parseaveis``, ``n_na_janela``,
    ``n_nao_plausiveis``, ``pct_na_janela``, ``cobertura_por_ano``, ``min_bruto``,
    ``max_bruto``, ``min_janela``, ``max_janela``, ``janela_2019_2026``,
    ``representatividade_ok``, ``qualidade``, ``bytes_membro``.
    """
    return {
        "campo": "data_deposito",
        "parseavel": False,
        "erro": None,
        "disponibilidade": 0.0,
        "proporcao_parseaveis": 0.0,
        "limiar_proporcao_parseaveis": LIMIAR_PROPR_PARSEAVEIS,
        "n_linhas_amostradas": 0,
        "n_com_data": 0,
        "n_parseaveis": 0,
        "n_na_janela": 0,
        "n_nao_plausiveis": 0,
        "pct_na_janela": 0.0,
        "cobertura_por_ano": {},
        "min_bruto": None,
        "max_bruto": None,
        "min_janela": None,
        "max_janela": None,
        "janela_2019_2026": False,
        "representatividade_ok": False,
        "qualidade": "ruim",
        "bytes_membro": None,
    }


def _dep_default() -> dict:
    return {"membro_encontrado": False, "coluna_cgccpfdepositante_presente": False, "colunas": []}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Protótipo de validação do dump de Dados Abertos do INPI (FR-1)."
    )
    ap.add_argument("--dump", default="patentes.zip", help="nome do zip em data/raw")
    ap.add_argument("--membro", default="PATENTES_DADOS_BIBLIOGRAFICOS.csv",
                    help="membro CSV a validar (contém data_deposito)")
    ap.add_argument("--membro-depositantes", default="PATENTES_DEPOSITANTES.csv",
                    help="membro CSV de depositantes a validar por coluna")
    ap.add_argument("--url", default=URL_ZIP, help="URL do dump de Dados Abertos")
    ap.add_argument("--no-progresso", action="store_true",
                    help="suprime barra de progresso do download")
    args = ap.parse_args(argv)

    rss_max = {"v": 0}

    def amostrar_rss() -> None:
        try:
            valor = psutil.Process().memory_info().rss
            if valor > rss_max["v"]:
                rss_max["v"] = valor
        except Exception:  # noqa: BLE001
            pass

    amostrar_rss()
    tracemalloc.start()

    rel_endpoint = validar_endpoint()
    amostrar_rss()

    rel_download = garantir_dump(args.dump, args.url, DIR_RAW, progresso=not args.no_progresso)
    amostrar_rss()

    rel_schema = _schema_default()
    rel_data = _data_default()
    rel_dep = _dep_default()
    membros_zip: list[str] = []
    descompactado = False

    caminho = rel_download.get("caminho")
    if rel_download.get("baixado") and rel_download.get("integro") and caminho \
            and os.path.exists(caminho):
        try:
            with zipfile.ZipFile(caminho) as zf:
                membros_zip = [i.filename for i in zf.infolist() if not i.is_dir()]
                try:
                    rel_schema, _colunas = validar_schema(zf, args.membro)
                except Exception as exc:  # noqa: BLE001
                    rel_schema = {**rel_schema, "schema_ok": False, "erro": str(exc)}
                amostrar_rss()
                delimitador = rel_schema.get("delimitador_detectado") or ","
                encoding = rel_schema.get("encoding") or "utf-8"
                try:
                    rel_data = parsear_e_validar_data(
                        zf, args.membro, delimitador, encoding=encoding, amostrar=amostrar_rss
                    )
                except Exception as exc:  # noqa: BLE001
                    rel_data = {**_data_default(), "erro": str(exc)}
                try:
                    rel_dep = validar_depositantes(zf, args.membro_depositantes)
                except Exception as exc:  # noqa: BLE001
                    rel_dep = {**_dep_default(), "erro": str(exc)}
                descompactado = True
        except Exception as exc:  # noqa: BLE001
            rel_download = {**rel_download, "erro": str(exc)}
            descompactado = False
    amostrar_rss()

    _, tracemalloc_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    pico_mb = round(rss_max["v"] / (1024 * 1024), 3)
    tracemalloc_mb = round(tracemalloc_bytes / (1024 * 1024), 3)

    dentro_do_limite = pico_mb < LIMITE_MEMORIA_MB
    endpoint_ok = bool(rel_endpoint.get("alcancavel"))
    schema_ok = bool(rel_schema.get("schema_ok"))
    data_ok = bool(rel_data.get("representatividade_ok"))
    memoria_ok = bool(dentro_do_limite)
    veredito = computar_veredito(endpoint_ok, schema_ok, data_ok, memoria_ok)

    caminho_relativo = os.path.basename(rel_download.get("caminho") or "")
    rel_report_download = {**rel_download, "caminho": caminho_relativo,
                           "descompactado": descompactado, "membros_no_zip": membros_zip}

    relatorio = {
        "endpoint": rel_endpoint,
        "download": rel_report_download,
        "schema": rel_schema,
        "data": rel_data,
        "depositantes": rel_dep,
        "memoria": {
            "pico_memoria_mb": pico_mb,
            "max_rss_mb": pico_mb,
            "tracemalloc_mb": tracemalloc_mb,
            "limite_mb": LIMITE_MEMORIA_MB,
            "dentro_do_limite": dentro_do_limite,
        },
        "janela": {"inicio": JANELA_INICIO, "fim": JANELA_FIM,
                   "limiar_pct_min": LIMIAR_PCT_JANELA,
                   "limiar_proporcao_parseaveis": LIMIAR_PROPR_PARSEAVEIS},
        "veredito": veredito,
        "fr1": {
            "status": veredito,
            "aceito": veredito == "aprovado",
            "endpoint_https": endpoint_ok,
            "schema": schema_ok,
            "data_janela_2019_2026": data_ok,
            "memoria": memoria_ok,
        },
        "metadados": {
            "args": vars(args),
            "versao_modulo": __version__,
            "gerado_em": datetime.now().isoformat(timespec="seconds"),
        },
    }

    os.makedirs(DIR_RAW, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    sys.stdout.write(
        f"veredito={veredito} pico_memoria_mb={pico_mb} relatorio={REPORT_PATH}\n"
    )
    return 0 if veredito == "aprovado" else 1


if __name__ == "__main__":
    sys.exit(main())
