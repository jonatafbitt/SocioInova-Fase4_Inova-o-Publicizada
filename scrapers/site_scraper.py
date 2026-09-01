"""Raspador config-driven de portais da RFEPCT (Story 1.4).

Lê ``config/urls_rede_federal.json`` (config-driven), raspa os portais da
Rede via ``requests`` + ``beautifulsoup4``, respeitando ``robots.txt`` e
atraso de polidez, captura data/título/texto de cada item e grava
``data/raw/noticias_rfepct.json`` (idempotente por URL canônica) e
``data/raw/relatorio_raspagem.json``.

Nenhum LLM, nenhuma rede: processamento 100% local/offline.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

__version__ = "1.4.0"

PROJETO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.dirname(os.path.abspath(__file__)) != PROJETO_RAIZ and PROJETO_RAIZ not in sys.path:
    sys.path.insert(0, PROJETO_RAIZ)

DIR_RAW = os.path.join(PROJETO_RAIZ, "data", "raw")
CONFIG_DEFAULT = os.path.join(PROJETO_RAIZ, "config", "urls_rede_federal.json")
OUTPUT_DEFAULT = os.path.join(DIR_RAW, "noticias_rfepct.json")
REPORT_DEFAULT = os.path.join(DIR_RAW, "relatorio_raspagem.json")

ATRASO_POLIDEZ = 1.0
TIMEOUT = 30
USER_AGENT = "RFEPCT-ResearchBot/1.4 (tese de doutorado em Ciencias Sociais; educacao-profissional)"

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def carregar_config(caminho: str) -> dict:
    """Carrega ``urls_rede_federal.json`` e retorna o dict completo.

    Em caso de arquivo ausente ou JSON inválido, registra um erro claro no log
    e encerra com ``sys.exit(2)`` (sem traceback não tratado).
    """
    try:
        with open(caminho, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log.error("Config não encontrado: %s", caminho)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        log.error("Config inválido (JSON malformado) em %s: %s", caminho, exc)
        sys.exit(2)


def _extrair_url_base(valor) -> str | None:
    """Extrai a URL-base de uma entrada do config.

    Aceita ``str`` (URL direta), ``dict`` com chave ``portal``, ou
    ``None``/tipos inesperados.
    """
    if isinstance(valor, str):
        return valor.rstrip("/")
    if isinstance(valor, dict):
        portal = valor.get("portal")
        if isinstance(portal, str) and portal:
            return portal.rstrip("/")
    return None


def flattener_portais(config: dict) -> list[dict]:
    """Achata o dict hierárquico do config em lista plana de portais.

    Cada item: ``{"instituicao": str, "url_base": str}`` onde ``instituicao``
    é o nome (chave-folha) da instituição. Pulando ``meta``, ``INPI``
    (tratado em 1.3), e entradas sem URL.
    """
    portais: list[dict] = []

    def _iterar(valor):
        if isinstance(valor, dict):
            for chave, sub in valor.items():
                if chave in ("meta", "INPI"):
                    continue
                if isinstance(sub, dict):
                    _iterar(sub)
                elif isinstance(sub, str):
                    url = _extrair_url_base(sub)
                    if url:
                        portais.append({"instituicao": chave, "url_base": url})

    for chave in config:
        if chave == "meta":
            continue
        _iterar(config[chave])

    return portais


# --------------------------------------------------------------------------- #
# Robots.txt
# --------------------------------------------------------------------------- #
def _robots_url(url_base: str) -> str:
    parsed = urlparse(url_base)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


def checar_robots(url_base: str, session: requests.Session | None = None) -> bool:
    """Verifica se o ``robots.txt`` proibe a URL-alvo.

    Retorna ``True`` quando a coleta é permitida (ou quando o robots.txt
    não pode ser lido); ``False`` quando explicitamente proibido.
    """
    sess = session or requests.Session()
    robots_url = _robots_url(url_base)
    try:
        resp = sess.get(robots_url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
        if resp.status_code >= 400:
            return True  # sem robots.txt => permitido
        texto = resp.text.lower()
        parsed = urlparse(url_base)
        # Busca seção User-agent: * (ou genérica)
        linhas = texto.splitlines()
        secao_aplicavel = False
        for linha in linhas:
            linha = linha.strip()
            if linha.startswith("user-agent:"):
                agente = linha.split(":", 1)[1].strip()
                secao_aplicavel = agente in ("*", "")
            elif linha.startswith("disallow:") and secao_aplicavel:
                caminho = linha.split(":", 1)[1].strip()
                if not caminho:
                    continue
                try:
                    # converte o padrão Disallow em regex, suportando wildcards '*'
                    pattern = re.escape(caminho).replace(r"\*", ".*")
                    regex = re.compile("^" + pattern)
                except re.error:  # noqa: BLE001
                    continue
                if caminho == "/":
                    return False  # proíbe tudo
                if regex.search(parsed.path):
                    return False
        return True
    except Exception:  # noqa: BLE001
        return True  # erro de conexão => permitir (será tratado no loop)


# --------------------------------------------------------------------------- #
# URL normalization (idempotency)
# --------------------------------------------------------------------------- #
def normalizar_url(url: str) -> str:
    """Normaliza URL para idempotência: sem trailing slash, lowercase scheme/host.

    Unifica ``http``->``https`` no mesmo host (design: ``https preferido``) para
    que o mesmo recurso, obtido via http e via https, produza a mesma chave.
    ``https`` permanece intacto: ``https://x`` nao vira outra coisa.
    """
    if not url:
        return ""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    if scheme == "http":
        scheme = "https"
    netloc = (parsed.netloc or "").lower()
    path = parsed.path.rstrip("/")
    query = parsed.query
    fragment = parsed.fragment
    normalized = f"{scheme}://{netloc}{path}"
    if query:
        normalized += f"?{query}"
    if fragment:
        normalized += f"#{fragment}"
    return normalized


def _chave_item(item: dict) -> str:
    """Chave de idempotência estável para um item raspado.

    Itens com ``url`` canônica usam a URL. Itens **sem** URL (feed/listagem que
    não expõe link) usam uma chave sintética derivada de ``titulo`` +
    ``instituicao`` + ``data``, para que nunca sejam re-anexados a cada run.
    Sempre prefixa a origem para evitar colisão acidental entre os dois espaços.
    """
    url = item.get("url") or ""
    if url:
        return "url:" + normalizar_url(url)
    titulo = (item.get("titulo") or "").strip().lower()
    inst = (item.get("instituicao") or "").strip().lower()
    data = (item.get("data") or "").strip()
    return f"semurl:{titulo}|{inst}|{data}"


# --------------------------------------------------------------------------- #
# RSS parsing
# --------------------------------------------------------------------------- #
def _local_name(tag: str) -> str:
    """Remove o namespace de uma tag ElementTree (``{ns}local`` -> ``local``)."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _text_no_ns(el: ET.Element, nome: str) -> str:
    """Extrai texto do primeiro filho com o nome local ``nome`` (sem namespace).

    Usa ``itertext()`` para capturar texto de markup aninhado (ex. ``<b>``
    dentro do ``<title>``), em vez de apenas ``.text``.
    """
    for child in el:
        if _local_name(child.tag) == nome:
            return "".join(child.itertext()).strip()
    return ""


def _texto_html_no_ns(el: ET.Element, nomes: tuple) -> str:
    """Extrai o primeiro corpo HTML (descrição/texto) dos filhos pelo nome local."""
    for child in el:
        if _local_name(child.tag) in nomes:
            texto = (child.text or "") + "".join(
                et.text or "" for et in child.iter() if et is not child and et.text
            )
            if texto.strip():
                return texto
    return ""


def _extrair_url_do_html(html: str, url_base: str) -> str:
    """Extrai a primeira URL ``href`` interna de um bloco HTML (feed sem ``<link>``)."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href and href != "#" and not href.startswith("javascript:"):
            return normalizar_url(urljoin(url_base, href))
    return ""


def _parsear_rss(xml_text: str, url_base: str) -> list[dict]:
    """Parseia um feed RSS/Atom e retorna itens normalizados.

    Suporta RSS 2.0 (``<item>``), RSS 1.0/RDF (``<item rdf:about>``,
    ``<dc:date>``) e Atom (``<entry>``), de forma robusta a namespaces.
    Quando o item não tem ``<link>`` (feeds Plone customizados), a URL é
    derivada do atributo ``rdf:about`` ou do primeiro ``href`` do corpo HTML.
    """
    itens: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return itens

    ns_atom = {"atom": "http://www.w3.org/2005/Atom"}
    corpo_tags = ("description", "texto", "summary", "content", "encoded")

    # Detecta se é um feed RSS (rss/rdf) ou Atom
    root_local = _local_name(root.tag)
    entries = [e for e in root.iter() if _local_name(e.tag) == "entry"]

    if root_local in ("rss", "RDF", "rdf") or _local_name(root.tag).endswith("RDF"):
        for item in root.iter():
            if _local_name(item.tag) != "item":
                continue
            titulo = _text_no_ns(item, "title")
            link = _text_no_ns(item, "link")
            # URL: <link>, <guid>, atributo rdf:about, ou href do corpo
            url = link
            if not url:
                guid = _text_no_ns(item, "guid")
                url = guid
            if not url:
                url = item.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about", "")
            # data: <pubDate> (RSS2) ou <dc:date> (RSS1/RDF)
            pubdate = None
            for child in item:
                ln = _local_name(child.tag)
                if ln in ("pubDate", "date") and (child.text or "").strip():
                    pubdate = child.text.strip()
                    break
            corpo = _texto_html_no_ns(item, corpo_tags)
            url_canon = normalizar_url(urljoin(url_base, url)) if url else ""
            if not url_canon:
                url_canon = _extrair_url_do_html(corpo, url_base)
            itens.append({
                "titulo": titulo or "",
                "data": _parsear_pubdate(pubdate),
                "url": url_canon,
                "texto": _limpar_html(corpo),
            })

    # Atom
    for entry in entries:
        titulo = _text(entry, "atom:title", ns_atom) or _text_no_ns(entry, "title")
        link_el = entry.find("atom:link", ns_atom)
        link = link_el.get("href", "") if link_el is not None else ""
        updated = (_text(entry, "atom:updated", ns_atom)
                   or _text(entry, "atom:published", ns_atom))
        resumo = (_text(entry, "atom:summary", ns_atom)
                  or _text(entry, "atom:content", ns_atom))
        url_canon = normalizar_url(urljoin(url_base, link)) if link else ""
        if not url_canon:
            url_canon = _extrair_url_do_html(resumo, url_base)
        itens.append({
            "titulo": titulo or "",
            "data": _parsear_pubdate(updated),
            "url": url_canon,
            "texto": _limpar_html(resumo or ""),
        })

    return itens


def _text(el: ET.Element, tag: str, ns: dict | None = None) -> str:
    """Extrai texto de um subelemento, retornando '' se ausente."""
    child = el.find(tag, ns) if ns else el.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _parsear_pubdate(s: str | None) -> str | None:
    """Tenta parsear datas RSS/Atom e datas brasileiras. Retorna ISO date ou None.

    Datas RFC822 em inglês (``%a, %d %b %Y``) só parseiam sob locale ``C``.
    Força ``LC_TIME`` para ``C`` ao redor dessas chamadas ``strptime`` e
    restaura o locale original em seguida.
    """
    if not s:
        return None
    s = s.strip()
    formatos = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        # brasileiro: dd/mm/yyyy e dd/mm/yyyy hh:mm
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
    ]
    locale_alterado = False
    locale_original = None
    try:
        import locale
        locale_original = locale.setlocale(locale.LC_TIME)
        if locale_original and locale_original != "C":
            locale.setlocale(locale.LC_TIME, "C")
            locale_alterado = True
    except Exception:  # noqa: BLE001 -- locale indisponível/gate falhou
        pass
    try:
        for fmt in formatos:
            try:
                dt = datetime.strptime(s, fmt)
                return dt.date().isoformat()
            except ValueError:
                continue
        # fallback: regex para YYYY-MM-DD ou DD/MM/YYYY
        m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
        if m:
            return m.group(1)
        m = re.match(r"(\d{2})/(\d{2})/(\d{4})", s)
        if m:
            dd, mm, aaaa = m.groups()
            try:
                return datetime(int(aaaa), int(mm), int(dd)).date().isoformat()
            except ValueError:
                return None
        return None
    finally:
        if locale_alterado and locale_original:
            try:
                locale.setlocale(locale.LC_TIME, locale_original)
            except Exception:  # noqa: BLE001
                pass


def _limpar_html(html: str) -> str:
    """Remove tags HTML e retorna texto plano."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def _texto_resposta(resp) -> str:
    """Texto decodificado de forma robusta, tratando charset declarado incorretamente.

    Alguns portais declaram ``charset=utf-8`` mas servem bytes latin-1/cp1252,
    o que faz ``resp.text`` conter caracteres de substituição (U+FFFD). Nesse
    caso, re-decodifica ``resp.content`` como latin-1 (cp1252).
    """
    texto = resp.text
    if "\ufffd" not in texto:
        return texto
    content = getattr(resp, "content", None)
    if content is None:
        return texto
    recon = content.decode("cp1252", errors="replace")
    if "\ufffd" not in recon:
        return recon
    return texto


# --------------------------------------------------------------------------- #
# HTML listing parsing (fallback)
# --------------------------------------------------------------------------- #
def _parsear_listagem_html(html_text: str, url_base: str) -> list[dict]:
    """Parseia listagem HTML genérica (Plone, WordPress, etc.) em busca de itens.

    Tenta seletores comuns de Plone:
    - ``div.newsItem`` / ``div.entry`` / ``div.item``
    - ``h2 a`` ou ``h3 a`` como título/link
    - ``span.date`` / ``span.pubDate`` / ``time`` como data
    """
    soup = BeautifulSoup(html_text, "html.parser")
    itens: list[dict] = []
    vistos: set[tuple] = set()  # dedupe por (url, titulo) entre seletores sobrepostos

    # seletores candidatos (Plone variants + Joomla/gov tile listings)
    seletores_container = [
        "div.tileItem", "div.newsItem", "div.news-item", "div.entry",
        "div.results-item", "article", "li.newsItem", "li.item",
        "div.item",
    ]

    for sel in seletores_container:
        containers = soup.select(sel)
        if not containers:
            continue
        # título + link
        titulo_sel = "h2 a, h3 a, h4 a, a.title, p a"
        for container in containers:
            titulo_el = container.select_one(titulo_sel)
            titulo = titulo_el.get_text(strip=True) if titulo_el else ""
            href = titulo_el.get("href", "") if titulo_el else ""
            url = normalizar_url(urljoin(url_base, href)) if href else ""

            chave = (url, titulo)
            if chave in vistos:
                continue  # mesmo item casado por seletor sobreposto (ex.: div.item>div.newsItem)
            vistos.add(chave)

            # data: elemento dedicado, ou <p> irmão que só tem data
            data = None
            for data_sel in ["time", "span.date", "span.pubDate", "span.newsDate",
                             "span.entry-date", ".effective", ".modified"]:
                data_el = container.select_one(data_sel)
                if data_el:
                    raw_date = data_el.get("datetime") or data_el.get_text(strip=True)
                    data = _parsear_pubdate(raw_date)
                    if data:
                        break
            if not data:
                # <p> irmão do link que contém apenas a data (ex.: "31/08/2026 16h14")
                for p in container.find_all("p"):
                    ptext = p.get_text(strip=True)
                    if ptext and re.match(r"^\d{1,2}/\d{1,2}/\d{4}", ptext):
                        data = _parsear_pubdate(ptext)
                        if data:
                            break

            # texto
            texto_el = container.select_one("div.description, p, div.text, span.description")
            texto = texto_el.get_text(separator=" ", strip=True) if texto_el else ""

            if titulo or url:
                itens.append({
                    "titulo": titulo,
                    "data": data,
                    "url": url,
                    "texto": texto,
                })
        if len(itens) >= 2:
            break  # seletor produziu itens reais (>= 2); não usar genérico demais

    # fallback: se nenhum container, busca links genéricos com data
    if not itens:
        for a_tag in soup.select("a[href]"):
            href = a_tag.get("href", "")
            titulo = a_tag.get_text(strip=True)
            if not titulo or not href or href.startswith("#") or href.startswith("javascript:"):
                continue
            # procura data em element-irmão ou pai
            data = None
            parent = a_tag.parent
            if parent:
                for sib in parent.find_all(["span", "time", "em"]):
                    raw = sib.get("datetime") or sib.get_text(strip=True)
                    data = _parsear_pubdate(raw)
                    if data:
                        break
            url = normalizar_url(urljoin(url_base, href))
            chave = (url, titulo)
            if chave in vistos:
                continue
            vistos.add(chave)
            itens.append({
                "titulo": titulo,
                "data": data,
                "url": url,
                "texto": "",
            })

    return itens


# --------------------------------------------------------------------------- #
# Core scraping
# --------------------------------------------------------------------------- #
def rasparr_portal(
    url_base: str,
    instituicao: str,
    session: requests.Session,
    atraso: float = ATRASO_POLIDEZ,
    urls_visitadas: set[str] | None = None,
) -> tuple[list[dict], dict]:
    """Raspa um portal: tenta RSS primeiro, fallback HTML.

    Retorna ``(itens, meta)`` onde ``meta`` contém ``n_itens``, ``fonte``
    (``"rss"``|``"html"``|``"nenhum"``), ``erro``, ``robots_ok``.
    """
    if urls_visitadas is None:
        urls_visitadas = set()

    meta: dict = {
        "url_base": url_base,
        "instituicao": instituicao,
        "n_itens": 0,
        "fonte": "nenhum",
        "erro": None,
        "robots_ok": True,
    }

    # robots.txt
    if not checar_robots(url_base, session):
        meta["robots_ok"] = False
        meta["erro"] = "robots.txt proibe o caminho-alvo"
        log.warning("robots.txt proibe: %s", url_base)
        return [], meta

    headers = {"User-Agent": USER_AGENT}
    itens: list[dict] = []

    # tentar RSS
    for rss_path in ("/RSS", "/rss", "/RSS.xml", "/rss.xml"):
        rss_url = url_base.rstrip("/") + rss_path
        try:
            time.sleep(atraso)
            resp = session.get(rss_url, timeout=TIMEOUT, headers=headers)
            if resp.status_code < 400:
                itens_rss = _parsear_rss(_texto_resposta(resp), url_base)
                if itens_rss:
                    itens = itens_rss
                    meta["fonte"] = "rss"
                    log.info("RSS OK: %s (%d itens)", rss_url, len(itens))
                    break
        except Exception:  # noqa: BLE001
            continue

    # fallback: HTML listing
    if not itens:
        try:
            time.sleep(atraso)
            resp = session.get(url_base, timeout=TIMEOUT, headers=headers)
            resp.raise_for_status()
            itens_html = _parsear_listagem_html(_texto_resposta(resp), url_base)
            if itens_html:
                itens = itens_html
                meta["fonte"] = "html"
                log.info("HTML OK: %s (%d itens)", url_base, len(itens))
        except Exception as exc:  # noqa: BLE001
            meta["erro"] = f"erro de conexão/parse: {exc}"
            log.error("Erro ao raspar %s: %s", url_base, exc)
            return [], meta

    # enriquecer com portal + normalizar URL
    for item in itens:
        item["portal"] = url_base
        item["instituicao"] = instituicao
        if item.get("url"):
            item["url"] = normalizar_url(item["url"])

    meta["n_itens"] = len(itens)
    return itens, meta


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def raspar(
    config_path: str = CONFIG_DEFAULT,
    output_path: str = OUTPUT_DEFAULT,
    report_path: str = REPORT_DEFAULT,
    portal_alvo: str | None = None,
    atraso: float = ATRASO_POLIDEZ,
) -> dict:
    """Pipeline principal: lê config, raspa portais, grava JSON + relatório.

    Retorna o relatório (dict).
    """
    config = carregar_config(config_path)
    portais = flattener_portais(config)

    if portal_alvo:
        portais = [p for p in portais if portal_alvo.lower() in p["instituicao"].lower()]
        if not portais:
            log.warning("Portal '%s' não encontrado no config", portal_alvo)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or DIR_RAW, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)) or DIR_RAW, exist_ok=True)

    # carregar dados existentes para merge idempotente
    itens_existentes: dict[str, dict] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                dados = json.load(f)
            if isinstance(dados, list):
                for item in dados:
                    itens_existentes[_chave_item(item)] = item
        except Exception:  # noqa: BLE001
            pass

    session = requests.Session()
    urls_visitadas: set[str] = set()
    todos_itens: list[dict] = []
    relatorio: dict = {
        "n_portais_total": len(portais),
        "n_portais_raspados": 0,
        "n_itens_total": 0,
        "n_itens_novos": 0,
        "n_itens_ja_existentes": 0,
        "portais_com_aviso": [],
        "portais_raspados": [],
        "erros_por_portal": {},
        "metadados": {
            "versao_modulo": __version__,
            "gerado_em": datetime.now().isoformat(timespec="seconds"),
            "config": config_path,
            "portal_filtro": portal_alvo,
        },
    }

    for portal in portais:
        inst = portal["instituicao"]
        url = portal["url_base"]

        log.info("Raspando: %s (%s)", inst, url)

        itens, meta = rasparr_portal(url, inst, session, atraso=atraso,
                                     urls_visitadas=urls_visitadas)

        if meta.get("erro"):
            relatorio["erros_por_portal"][url] = meta["erro"]

        if not meta.get("robots_ok"):
            relatorio["portais_com_aviso"].append({
                "instituicao": inst, "url": url, "motivo": "robots.txt"
            })
        elif meta["n_itens"] == 0 and not meta.get("erro"):
            relatorio["portais_com_aviso"].append({
                "instituicao": inst, "url": url,
                "motivo": "nenhum item encontrado (RSS+HTML)"
            })
        elif meta["n_itens"] > 0:
            relatorio["portais_raspados"].append({
                "instituicao": inst, "url": url,
                "n_itens": meta["n_itens"], "fonte": meta["fonte"],
            })
            relatorio["n_portais_raspados"] += 1

        # merge idempotente
        novos = 0
        existentes = 0
        for item in itens:
            chave = _chave_item(item)
            if chave in itens_existentes:
                existentes += 1
                continue
            itens_existentes[chave] = item
            todos_itens.append(item)
            novos += 1

        relatorio["n_itens_ja_existentes"] += existentes

    # consolidar: itens existentes que não foram re-raspados + novos
    itens_finais = list(itens_existentes.values())
    relatorio["n_itens_total"] = len(itens_finais)
    relatorio["n_itens_novos"] = len(todos_itens)

    # gravar JSON de saída
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or DIR_RAW, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(itens_finais, f, ensure_ascii=False, indent=2)

    # gravar relatório
    os.makedirs(os.path.dirname(os.path.abspath(report_path)) or DIR_RAW, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    sys.stdout.write(
        f"n_portais_raspados={relatorio['n_portais_raspados']} "
        f"n_itens_total={relatorio['n_itens_total']} "
        f"relatorio={report_path}\n"
    )

    return relatorio


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Raspador config-driven de portais da RFEPCT (Story 1.4)."
    )
    ap.add_argument("--config", default=CONFIG_DEFAULT,
                    help="caminho do urls_rede_federal.json")
    ap.add_argument("--output", default=OUTPUT_DEFAULT,
                    help="JSON de saída (noticias_rfepct.json)")
    ap.add_argument("--report", default=REPORT_DEFAULT,
                    help="JSON de relatório (relatorio_raspagem.json)")
    ap.add_argument("--portal", default=None,
                    help="filtro: substring do nome da instituição (ex.: IFSP)")
    ap.add_argument("--atraso", type=float, default=ATRASO_POLIDEZ,
                    help=f"atraso de polidez entre requests (s), mínimo {ATRASO_POLIDEZ}s. "
                         f"Default: {ATRASO_POLIDEZ}")
    ap.add_argument("--verbose", action="store_true",
                    help="habilita logging verboso")
    args = ap.parse_args(argv)

    # invariante: atraso de polidez nunca abaixo do mínimo (1 s)
    if args.atraso < ATRASO_POLIDEZ:
        log.warning(
            "--atraso=%s ignorado (mínimo %ss). Usando %ss.",
            args.atraso, ATRASO_POLIDEZ, ATRASO_POLIDEZ,
        )
        args.atraso = ATRASO_POLIDEZ

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    relatorio = raspar(
        config_path=args.config,
        output_path=args.output,
        report_path=args.report,
        portal_alvo=args.portal,
        atraso=args.atraso,
    )

    n_erros = len(relatorio.get("erros_por_portal", {}))
    return 0 if n_erros == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
