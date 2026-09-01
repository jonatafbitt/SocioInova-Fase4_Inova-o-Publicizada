"""Suite pytest (Story 1.4) com HTML fake de portal Plone (RSS + listagem).

Cobre os critérios de aceitação da Story 1.4: extração de itens (título, data,
url, portal, instituição), seletor ausente => aviso (não exceção), robots.txt
proibindo => portal ignorado, atraso de polidez >= 1s, idempotência por URL,
item sem data retido com data:null, e CLI (exit code/JSON). Sem rede (requests
monkeypatchado).
"""
import json
from types import SimpleNamespace

import pytest

import scrapers.site_scraper as ss


# --------------------------------------------------------------------------- #
# HTML fake fixtures
# --------------------------------------------------------------------------- #
RSS_PLONE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>IFSP Notícias</title>
    <link>https://ifsp.edu.br</link>
    <description>Notícias do IFSP</description>
    <item>
      <title>Inovação no campus</title>
      <link>https://ifsp.edu.br/noticia-inovacao</link>
      <description><![CDATA[<p>Texto sobre a inovação publicizada.</p>]]></description>
      <pubDate>Mon, 12 Aug 2024 10:00:00 -0300</pubDate>
    </item>
    <item>
      <title>Sem data explícita</title>
      <link>https://ifsp.edu.br/noticia-sem-data</link>
      <description>Sem pubDate</description>
    </item>
  </channel>
</rss>
"""

HTML_PLONE = """<!DOCTYPE html>
<html><body>
  <div class="newsItem">
    <h2><a href="https://ifsp.edu.br/noticia-html">Título HTML 1</a></h2>
    <span class="date">2024-08-12</span>
    <div class="description">Descrição do item HTML 1</div>
  </div>
  <div class="newsItem">
    <h2><a href="https://ifsp.edu.br/noticia-html-2">Título HTML 2</a></h2>
    <span class="date">2024-09-01</span>
    <div class="description">Descrição do item HTML 2</div>
  </div>
</body></html>
"""

CONFIG_IFSP = {
    "institutos_federais": {"Sudeste": {"IFSP": "https://ifsp.edu.br"}}
}


def _fazer_resp(texto, status_code=200):
    resp = SimpleNamespace()
    resp.text = texto
    resp.status_code = status_code
    resp.url = "https://ifsp.edu.br/"

    def raise_for_status():
        if status_code >= 400:
            raise Exception(f"HTTP {status_code}")

    resp.raise_for_status = raise_for_status
    return resp


def _fake_session(rotas):
    sess = SimpleNamespace()
    sess.chamadas = []

    def get(url, **kwargs):
        sess.chamadas.append(url)
        if "robots.txt" in url:
            return rotas.get("robots", _fazer_resp("User-agent: *\nDisallow:", 200))
        # variações de trailing slash devem casar a mesma rota (ifsp.edu.br vs ifsp.edu.br/)
        url_key = url.rstrip("/")
        rotas_flex = {}
        for k, v in rotas.items():
            rotas_flex[k.rstrip("/")] = v
        if url_key in rotas_flex:
            return rotas_flex[url_key]
        return rotas.get(url, _fazer_resp("", 404))

    sess.get = get
    return sess


def _escrever_config(tmp_path, config=CONFIG_IFSP):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    return str(cfg)


# --------------------------------------------------------------------------- #
# AC: extração de itens (RSS)
# --------------------------------------------------------------------------- #
def test_ac_extrai_itens_rss():
    sess = _fake_session({
        "https://ifsp.edu.br/RSS": _fazer_resp(RSS_PLONE),
    })
    itens, meta = ss.rasparr_portal("https://ifsp.edu.br", "IFSP", sess)
    assert meta["fonte"] == "rss"
    assert meta["n_itens"] >= 2
    item0 = itens[0]
    assert item0["titulo"] == "Inovação no campus"
    assert item0["data"] == "2024-08-12"
    assert "noticia-inovacao" in item0["url"]
    assert item0["portal"] == "https://ifsp.edu.br"
    assert item0["instituicao"] == "IFSP"
    sem_data = [i for i in itens if "sem-data" in i["url"]]
    assert sem_data and sem_data[0]["data"] is None


# --------------------------------------------------------------------------- #
# AC: fallback HTML quando RSS ausente
# --------------------------------------------------------------------------- #
def test_ac_fallback_html_quando_rss_ausente():
    sess = _fake_session({
        "https://ifsp.edu.br/": _fazer_resp(HTML_PLONE),
    })
    itens, meta = ss.rasparr_portal("https://ifsp.edu.br", "IFSP", sess)
    assert meta["fonte"] == "html"
    assert meta["n_itens"] >= 2
    assert itens[0]["titulo"] == "Título HTML 1"
    assert itens[0]["data"] == "2024-08-12"
    assert itens[0]["instituicao"] == "IFSP"


# --------------------------------------------------------------------------- #
# AC: seletor ausente => aviso no relatório, portal ignorado, sem exceção
# --------------------------------------------------------------------------- #
def test_ac_seletor_ausente_registra_aviso(tmp_path, monkeypatch):
    # URL existe, mas nem RSS nem HTML produzem itens (sem seletor válido)
    sess = _fake_session({
        "https://ifsp.edu.br/": _fazer_resp("<html><body>conteúdo sem itens</body></html>"),
    })
    monkeypatch.setattr(ss.requests, "Session", lambda: sess)
    out = tmp_path / "noticias.json"
    rel = tmp_path / "relatorio.json"
    cfg = _escrever_config(tmp_path)

    ss.raspar(config_path=cfg, output_path=str(out), report_path=str(rel))

    rel_data = json.loads(rel.read_text(encoding="utf-8"))
    avisos = [a for a in rel_data["portais_com_aviso"] if a["instituicao"] == "IFSP"]
    assert avisos


# --------------------------------------------------------------------------- #
# AC: robots.txt proibe o caminho-alvo => portal ignorado + relatório
# --------------------------------------------------------------------------- #
def test_ac_robots_proibe_ignora_portal():
    sess = _fake_session({
        "robots": _fazer_resp("User-agent: *\nDisallow: /"),
    })
    itens, meta = ss.rasparr_portal("https://ifsp.edu.br", "IFSP", sess)
    assert itens == []
    assert meta["robots_ok"] is False
    assert "robots" in (meta["erro"] or "")


def test_ac_robots_proibe_registra_no_relatorio(tmp_path, monkeypatch):
    sess = _fake_session({
        "robots": _fazer_resp("User-agent: *\nDisallow: /"),
    })
    monkeypatch.setattr(ss.requests, "Session", lambda: sess)
    out = tmp_path / "noticias.json"
    rel = tmp_path / "relatorio.json"
    cfg = _escrever_config(tmp_path)

    ss.raspar(config_path=cfg, output_path=str(out), report_path=str(rel))

    rel_data = json.loads(rel.read_text(encoding="utf-8"))
    avisos = [a for a in rel_data["portais_com_aviso"]
              if a["instituicao"] == "IFSP" and "robots" in a["motivo"]]
    assert avisos


# --------------------------------------------------------------------------- #
# AC: atraso de polidez >= 1s (configurável) entre requests
# --------------------------------------------------------------------------- #
def test_ac_atraso_polidez(monkeypatch):
    intervalos = []

    def fake_sleep(seg):
        intervalos.append(seg)

    monkeypatch.setattr(ss.time, "sleep", fake_sleep)
    sess = _fake_session({
        "https://ifsp.edu.br/RSS": _fazer_resp(RSS_PLONE),
    })
    itens, meta = ss.rasparr_portal("https://ifsp.edu.br", "IFSP", sess, atraso=2.0)
    assert itens
    assert any(seg >= 1.0 for seg in intervalos)
    assert all(seg == 2.0 for seg in intervalos)


# --------------------------------------------------------------------------- #
# AC: idempotência por URL canônica (re-execução sem duplicatas)
# --------------------------------------------------------------------------- #
def test_ac_idempotencia(tmp_path, monkeypatch):
    sess = _fake_session({
        "https://ifsp.edu.br/RSS": _fazer_resp(RSS_PLONE),
    })
    monkeypatch.setattr(ss.requests, "Session", lambda: sess)
    out = tmp_path / "noticias.json"
    rel = tmp_path / "relatorio.json"
    cfg = _escrever_config(tmp_path)

    ss.raspar(config_path=cfg, output_path=str(out), report_path=str(rel))
    dado1 = json.loads(out.read_text(encoding="utf-8"))
    n1 = len(dado1)
    urls1 = {item["url"] for item in dado1}
    # mesma execução (re-raspa os mesmos itens) => merge por URL, sem duplicatas
    ss.raspar(config_path=cfg, output_path=str(out), report_path=str(rel))
    dado2 = json.loads(out.read_text(encoding="utf-8"))
    urls2 = {item["url"] for item in dado2}
    assert len(urls2) == n1  # mais raspar não adiciona (mesmos dados simulados)
    assert len(urls2) == len(dado2)  # sem duplicatas
    assert urls1 == urls2


# --------------------------------------------------------------------------- #
# AC: CLI com --portal e exit code 0 + JSON coerente
# --------------------------------------------------------------------------- #
def test_cli_gera_json(tmp_path, monkeypatch):
    sess = _fake_session({
        "https://ifsp.edu.br/RSS": _fazer_resp(RSS_PLONE),
    })
    monkeypatch.setattr(ss.requests, "Session", lambda: sess)
    out = tmp_path / "noticias.json"
    rel = tmp_path / "relatorio.json"
    cfg = _escrever_config(tmp_path)

    codigo = ss.main(["--config", str(cfg), "--output", str(out),
                      "--report", str(rel), "--portal", "IFSP"])
    assert codigo == 0
    dados = json.loads(out.read_text(encoding="utf-8"))
    assert len(dados) >= 2
    assert all(item.get("titulo") is not None for item in dados)
    assert all(item.get("instituicao") == "IFSP" for item in dados)
    rel_data = json.loads(rel.read_text(encoding="utf-8"))
    assert rel_data["n_portais_raspados"] >= 1


# --------------------------------------------------------------------------- #
# AC: flattener do config — INPI é dict, instituições são strings
# --------------------------------------------------------------------------- #
def test_flattener_config_heterogeneo(tmp_path):
    config = {
        "meta": {"x": 1},
        "orgaos_centrais": {
            "MEC": "https://www.gov.br/mec/pt-br",
            "INPI": {"portal": "https://www.gov.br/inpi/pt-br", "dados_abertos": {}},
        },
        "centros_federais_e_colegio_pedro_ii": {
            "CEFET-MG": "https://www.cefetmg.br",
        },
        "institutos_federais": {"Sudeste": {"IFSP": "https://ifsp.edu.br"}},
    }
    portais = ss.flattener_portais(config)
    # INPI (dict) e meta pulados; MEC, CEFET-MG, IFSP extraídos como strings
    siglas = [p["instituicao"] for p in portais]
    assert "MEC" in siglas
    assert "CEFET-MG" in siglas
    assert "IFSP" in siglas
    urls = {p["url_base"] for p in portais}
    assert "https://ifsp.edu.br" in urls
    assert "https://www.cefetmg.br" in urls
    assert all("inpi" not in p["url_base"] for p in portais)


def test_normalizar_url_trailing_slash_e_scheme():
    assert ss.normalizar_url("https://IFSP.edu.br/") == "https://ifsp.edu.br"
    # http é unificado para https no mesmo host (P2 — idempotência)
    assert ss.normalizar_url("HTTP://ifsp.edu.br/noticia/") == "https://ifsp.edu.br/noticia"


# --------------------------------------------------------------------------- #
# Data brasileira (dd/mm/aaaa) e encoding latin-1-ofuscado
# --------------------------------------------------------------------------- #
def test_parsear_data_brasileira():
    assert ss._parsear_pubdate("31/08/2026 16h14") == "2026-08-31"
    assert ss._parsear_pubdate("31/08/2026") == "2026-08-31"
    assert ss._parsear_pubdate("2024-08-12") == "2024-08-12"
    assert ss._parsear_pubdate("") is None
    assert ss._parsear_pubdate(None) is None
    assert ss._parsear_pubdate("lixo") is None


def test_texto_resposta_trata_charset_incorreto():
    # página declara utf-8 mas serve cp1252 => resp.text tem U+FFFD
    import requests as _requests
    resp = _requests.Response()
    resp._content = "Para além do diagnóstico".encode("cp1252")
    resp.encoding = "utf-8"
    resp.status_code = 200
    texto = ss._texto_resposta(resp)
    assert "além" in texto
    assert "\ufffd" not in texto


# --------------------------------------------------------------------------- #
# RSS 1.0 / RDF (Plone) — namespace handling é essencial neste formato
# --------------------------------------------------------------------------- #
RSS_1_0 = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:dc="http://purl.org/dc/elements/1.1/"
         xmlns="http://purl.org/rss/1.0/">
  <channel rdf:about="https://ifal.edu.br/RSS">
    <title>IFAL</title>
    <link>https://ifal.edu.br</link>
    <items>
      <rdf:Seq>
        <rdf:li rdf:resource="https://ifal.edu.br/noticia-1" />
        <rdf:li rdf:resource="https://ifal.edu.br/noticia-2" />
      </rdf:Seq>
    </items>
  </channel>
  <item rdf:about="https://ifal.edu.br/noticia-1">
    <title>Notícia Um</title>
    <link>https://ifal.edu.br/noticia-1</link>
    <dc:date>2026-08-31T21:31:00Z</dc:date>
  </item>
  <item rdf:about="https://ifal.edu.br/noticia-2">
    <title>Notícia Dois</title>
    <link>https://ifal.edu.br/noticia-2</link>
    <dc:date>2026-09-01T10:00:00Z</dc:date>
  </item>
</rdf:RDF>
"""


def test_parsear_rss_1_0_rdf_com_namespace():
    itens = ss._parsear_rss(RSS_1_0, "https://ifal.edu.br")
    assert len(itens) == 2
    urls = {i["url"] for i in itens}
    assert urls == {"https://ifal.edu.br/noticia-1", "https://ifal.edu.br/noticia-2"}
    datas = sorted(i["data"] for i in itens)
    assert datas == ["2026-08-31", "2026-09-01"]
    tits = {i["titulo"] for i in itens}
    assert "Notícia Um" in tits


def test_parsear_rss_2_0_sem_link_usa_guid():
    rss = """<?xml version="1.0"?>
<rss version="2.0"><channel><item>
  <title>Item</title>
  <guid>https://ifal.edu.br/item-1</guid>
  <pubDate>Mon, 12 Aug 2024 10:00:00 -0300</pubDate>
</item></channel></rss>"""
    itens = ss._parsear_rss(rss, "https://ifal.edu.br")
    assert len(itens) == 1
    assert itens[0]["url"] == "https://ifal.edu.br/item-1"
    assert itens[0]["data"] == "2024-08-12"


# --------------------------------------------------------------------------- #
# REVIEW P1: idempotência para itens SEM url (link/guid/rdf:about/body-href)
# --------------------------------------------------------------------------- #
RSS_SEM_URL = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>IFSP</title>
  <link>https://ifsp.edu.br</link>
  <item>
    <title>Item sem qualquer link</title>
    <description>Sem href algum</description>
    <pubDate>Mon, 12 Aug 2024 10:00:00 -0300</pubDate>
  </item>
</channel></rss>
"""


def test_p1_item_sem_url_idempotente(tmp_path, monkeypatch):
    # feed cujo item não tem <link>, <guid>, rdf:about nem href no corpo
    sess = _fake_session({
        "https://ifsp.edu.br/RSS": _fazer_resp(RSS_SEM_URL),
    })
    monkeypatch.setattr(ss.requests, "Session", lambda: sess)
    out = tmp_path / "noticias.json"
    rel = tmp_path / "relatorio.json"
    cfg = _escrever_config(tmp_path)

    ss.raspar(config_path=cfg, output_path=str(out), report_path=str(rel))
    dado1 = json.loads(out.read_text(encoding="utf-8"))
    n1 = len(dado1)
    # o item sem url NÃO deve ser descartado
    assert n1 == 1
    assert dado1[0]["url"] == ""

    # re-raspagem: item sem url não pode ser re-anexado (mesma chave sintética)
    ss.raspar(config_path=cfg, output_path=str(out), report_path=str(rel))
    dado2 = json.loads(out.read_text(encoding="utf-8"))
    assert len(dado2) == n1  # idêntico entre execuções
    assert len(dado2) == 1  # sem duplicatas


def test_p1_dois_itens_sem_url_distintos_nao_colidem(tmp_path, monkeypatch):
    # dois itens sem url, com títulos/descrições diferentes => chaves distintas
    rss = """<?xml version="1.0"?><rss version="2.0"><channel><item>
  <title>Primeiro</title><description>a</description>
  <pubDate>Mon, 12 Aug 2024 10:00:00 -0300</pubDate>
</item><item>
  <title>Segundo</title><description>b</description>
  <pubDate>Mon, 12 Aug 2024 10:00:00 -0300</pubDate>
</item></channel></rss>"""
    sess = _fake_session({"https://ifsp.edu.br/RSS": _fazer_resp(rss)})
    monkeypatch.setattr(ss.requests, "Session", lambda: sess)
    out = tmp_path / "noticias.json"
    rel = tmp_path / "relatorio.json"
    cfg = _escrever_config(tmp_path)
    ss.raspar(config_path=cfg, output_path=str(out), report_path=str(rel))
    dado = json.loads(out.read_text(encoding="utf-8"))
    assert len(dado) == 2


def test_normalizar_url_vazia_nao_colapsa():
    assert ss.normalizar_url("") == ""
    # dois itens sem url NÃO colapsam num único "https://" via normalizar
    assert ss.normalizar_url("") == ss.normalizar_url("") == ""


# --------------------------------------------------------------------------- #
# REVIEW P2: unificação http -> https (mesmo host = mesma chave)
# --------------------------------------------------------------------------- #
def test_p2_normaliza_http_para_https():
    assert ss.normalizar_url("http://ifsp.edu.br/x") == "https://ifsp.edu.br/x"
    assert ss.normalizar_url("https://ifsp.edu.br/x") == "https://ifsp.edu.br/x"
    assert ss.normalizar_url("http://ifsp.edu.br/x") == ss.normalizar_url("https://ifsp.edu.br/x")


# --------------------------------------------------------------------------- #
# REVIEW P3: robots.txt com wildcard '*'
# --------------------------------------------------------------------------- #
def test_p3_robots_wildcard_bloqueia_area():
    sess = _fake_session({
        "robots": _fazer_resp("User-agent: *\nDisallow: /news/*"),
    })
    # dentro da área bloqueada
    itens, meta = ss.rasparr_portal("https://ifsp.edu.br/news/2026", "IFSP", sess)
    assert itens == []
    assert meta["robots_ok"] is False


def test_p3_robots_wildcard_permite_fora_da_area():
    sess = _fake_session({
        "robots": _fazer_resp("User-agent: *\nDisallow: /news/*"),
    })
    # fora do padrão => robots permite (mesmo que a coleta depois falhe/erro)
    itens, meta = ss.rasparr_portal("https://ifsp.edu.br/sobre", "IFSP", sess)
    assert meta["robots_ok"] is True


# --------------------------------------------------------------------------- #
# REVIEW P4: --atraso clamp para mínimo 1s
# --------------------------------------------------------------------------- #
def test_p4_atraso_zero_clampado(monkeypatch, tmp_path):
    slept = []

    def fake_sleep(seg):
        slept.append(seg)

    monkeypatch.setattr(ss.time, "sleep", fake_sleep)
    sess = _fake_session({"https://ifsp.edu.br/RSS": _fazer_resp(RSS_PLONE)})
    monkeypatch.setattr(ss.requests, "Session", lambda: sess)
    out = tmp_path / "n.json"
    rel = tmp_path / "r.json"
    cfg = _escrever_config(tmp_path)
    codigo = ss.main(["--config", str(cfg), "--output", str(out),
                      "--report", str(rel), "--portal", "IFSP", "--atraso", "0"])
    assert codigo == 0
    # atraso 0 foi clampado para 1s; nenhuma chamada abaixo de 1s
    assert slept and all(seg >= 1.0 for seg in slept)


def test_p4_atraso_negativo_clampado(monkeypatch, tmp_path):
    slept = []

    def fake_sleep(seg):
        slept.append(seg)

    monkeypatch.setattr(ss.time, "sleep", fake_sleep)
    sess = _fake_session({"https://ifsp.edu.br/RSS": _fazer_resp(RSS_PLONE)})
    monkeypatch.setattr(ss.requests, "Session", lambda: sess)
    out = tmp_path / "n.json"
    rel = tmp_path / "r.json"
    cfg = _escrever_config(tmp_path)
    ss.main(["--config", str(cfg), "--output", str(out), "--report", str(rel),
             "--portal", "IFSP", "--atraso", "-3"])
    assert slept and all(seg >= 1.0 for seg in slept)


# --------------------------------------------------------------------------- #
# REVIEW P5: dedupe na listagem HTML entre seletores sobrepostos
# --------------------------------------------------------------------------- #
def test_p5_html_dedupe_seletores_sobrepostos():
    # div.item aninha div.newsItem com o mesmo artigo => mesmo item 2x
    html = """<html><body>
  <div class="item">
    <div class="newsItem">
      <h2><a href="https://ifsp.edu.br/a">Mesmo artigo</a></h2>
      <span class="date">2024-08-12</span>
    </div>
  </div>
  <div class="item">
    <div class="newsItem">
      <h2><a href="https://ifsp.edu.br/b">Outro artigo</a></h2>
      <span class="date">2024-09-01</span>
    </div>
  </div>
</body></html>"""
    itens = ss._parsear_listagem_html(html, "https://ifsp.edu.br")
    urls = [i["url"] for i in itens]
    # cada (url, titulo) único contado uma vez, apesar dos seletores sobrepostos
    assert len(urls) == 2
    assert len(set(urls)) == len(urls)


# --------------------------------------------------------------------------- #
# REVIEW P6: data RFC822 em inglês sob qualquer locale
# --------------------------------------------------------------------------- #
def test_p6_pubdate_ingles_sob_locale_nao_c():
    import os
    os.environ["LC_ALL"] = "pt_BR.UTF-8"
    os.environ["LANG"] = "pt_BR.UTF-8"
    try:
        import locale
        locale.setlocale(locale.LC_ALL, "")
    except Exception:  # noqa: BLE001 -- locale pt_BR pode não existir no host
        pass
    # se o host suportar pt_BR, força o caminho; senão cai em C (ainda ok)
    datas = ss._parsear_pubdate("Mon, 12 Aug 2024 10:00:00 -0300")
    assert datas is not None
    assert datas == "2024-08-12"


# --------------------------------------------------------------------------- #
# REVIEW P7: título com markup aninhado (ex.: <b> dentro de <title>)
# --------------------------------------------------------------------------- #
def test_p7_titulo_com_markup_aninhado():
    rss = """<?xml version="1.0"?><rss version="2.0"><channel><item>
  <title>Projeto <![CDATA[<b>Inovação</b>]]> social</title>
  <link>https://ifal.edu.br/x</link>
  <pubDate>Mon, 12 Aug 2024 10:00:00 -0300</pubDate>
</item></channel></rss>"""
    itens = ss._parsear_rss(rss, "https://ifal.edu.br")
    assert len(itens) == 1
    assert "Inovação" in itens[0]["titulo"]


# --------------------------------------------------------------------------- #
# REVIEW P11: RSS item sem pubDate E sem description => data:null retido
# --------------------------------------------------------------------------- #
def test_p11_item_sem_data_e_sem_descricao():
    rss = """<?xml version="1.0"?><rss version="2.0"><channel><item>
  <title>Somente título</title>
  <link>https://ifal.edu.br/y</link>
</item></channel></rss>"""
    itens = ss._parsear_rss(rss, "https://ifal.edu.br")
    assert len(itens) == 1
    assert itens[0]["data"] is None
    assert itens[0]["titulo"] == "Somente título"  # item retido


# --------------------------------------------------------------------------- #
# REVIEW P12: URL derivada do primeiro href do corpo (item sem <link>)
# --------------------------------------------------------------------------- #
def test_p12_url_derivada_do_href_do_corpo():
    rss = """<?xml version="1.0"?><rss version="2.0"><channel><item>
  <title>Sem link direto</title>
  <description><![CDATA[<p>Veja <a href="https://ifal.edu.br/noticias/abc">aqui</a>.</p>]]></description>
  <pubDate>Mon, 12 Aug 2024 10:00:00 -0300</pubDate>
</item></channel></rss>"""
    itens = ss._parsear_rss(rss, "https://ifal.edu.br")
    assert len(itens) == 1
    assert itens[0]["url"] == "https://ifal.edu.br/noticias/abc"
