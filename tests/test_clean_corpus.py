"""Suite pytest (Story 1.5) com CSV/JSON de entrada sintéticos em tmp_path.

Cobre os critérios de aceitação da Story 1.5: unificação INPI + notícias,
Janela 2019–2026 sem perda, item sem data retido e marcado, limpeza de HTML,
fallback título, multi-instituições (primeira + auxiliares), idempotência por
chave canônica e CLI (exit code/relatório coerente). Sem rede e sem dados reais.
"""
import csv
import json
import os
import pathlib
import tempfile

import pytest

import scrapers.clean_corpus as cc


def _escrever_inpi(tmp_path, linhas):
    p = tmp_path / "inpi.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["numero_inpi", "data_deposito", "titulo", "resumo", "instituicoes"])
        for linha in linhas:
            writer.writerow(linha)
    return str(p)


def _escrever_noticias(tmp_path, itens):
    p = tmp_path / "noticias.json"
    p.write_text(json.dumps(itens, ensure_ascii=False), encoding="utf-8")
    return str(p)


def _ler_csv(caminho):
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _cache_out(tmp_path, inpi_linhas, noticias):
    """Roda limpar_corpus com inputs sintéticos e retorna (rel, linhas_csv)."""
    inpi = _escrever_inpi(tmp_path, inpi_linhas)
    noti = _escrever_noticias(tmp_path, noticias)
    out = str(tmp_path / "corpus.csv")
    rel_path = str(tmp_path / "relatorio.json")
    rel = cc.limpar_corpus(inpi, noti, out, rel_path)
    return rel, _ler_csv(out)


# --------------------------------------------------------------------------- #
# AC: uma linha por item com colunas mínimas
# --------------------------------------------------------------------------- #
def test_ac_linha_por_item_com_colunas(tmp_path):
    inpi = [
        ("BR1", "2023-05-01", "Titulo INPI", "Resumo INPI", "INSTITUTO FEDERAL DE X"),
    ]
    noticias = [
        {"titulo": "Notícia A", "data": "2026-08-31", "url": "https://ifal.edu.br/a",
         "texto": "Texto da notícia", "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
    ]
    rel, linhas = _cache_out(tmp_path, inpi, noticias)

    assert len(linhas) == 2
    colunas = set(linhas[0].keys())
    assert {"fonte", "tipo", "data", "instituicao", "texto_límpido",
            "na_janela", "data_ausente"} <= colunas
    inpi_row = [r for r in linhas if r["tipo"] == "inpi"][0]
    assert inpi_row["fonte"] == "BR1"
    assert inpi_row["data"] == "2023-05-01"
    assert inpi_row["na_janela"] == "True"
    noti_row = [r for r in linhas if r["tipo"] == "noticia"][0]
    assert noti_row["fonte"] == "https://ifal.edu.br/a"
    assert noti_row["data"] == "2026-08-31"
    assert noti_row["na_janela"] == "True"


# --------------------------------------------------------------------------- #
# AC: itens fora da Janela (2015) retidos sem perda
# --------------------------------------------------------------------------- #
def test_ac_fora_da_janela_retido(tmp_path):
    inpi = [
        ("BR1", "2015-08-05", "Titulo antigo", "Resumo antigo", "INSTITUTO FEDERAL DE X"),
    ]
    rel, linhas = _cache_out(tmp_path, inpi, [])
    assert len(linhas) == 1
    row = linhas[0]
    assert row["data"] == "2015-08-05"
    assert row["na_janela"] == "False"
    assert row["data_ausente"] == "False"
    assert rel["n_fora_janela"] == 1
    assert rel["n_janela"] == 0


# --------------------------------------------------------------------------- #
# AC: item sem data retido, data_ausente true, não contado na Janela
# --------------------------------------------------------------------------- #
def test_ac_item_sem_data_retido(tmp_path):
    noticias = [
        {"titulo": "Sem data", "data": None, "url": "https://ifal.edu.br/sem-data",
         "texto": "texto", "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
    ]
    rel, linhas = _cache_out(tmp_path, [], noticias)
    assert len(linhas) == 1
    row = linhas[0]
    assert row["data"] == ""
    assert row["data_ausente"] == "True"
    assert row["na_janela"] == "False"
    assert rel["n_sem_data"] == 1
    assert rel["n_janela"] == 0


# --------------------------------------------------------------------------- #
# AC: item dentro da Janela => na_janela true
# --------------------------------------------------------------------------- #
def test_ac_dentro_da_janela(tmp_path):
    inpi = [
        ("BR1", "2023-05-01", "T", "R", "INSTITUTO FEDERAL DE X"),
    ]
    rel, linhas = _cache_out(tmp_path, inpi, [])
    assert linhas[0]["na_janela"] == "True"
    assert rel["n_janela"] == 1


# --------------------------------------------------------------------------- #
# AC: texto vazio usa título (fallback)
# --------------------------------------------------------------------------- #
def test_ac_texto_vazio_usar_titulo(tmp_path):
    noticias = [
        {"titulo": "Título da matéria", "data": "2026-08-31",
         "url": "https://ifal.edu.br/x", "texto": "", "portal": "https://ifal.edu.br",
         "instituicao": "IFAL"},
    ]
    rel, linhas = _cache_out(tmp_path, [], noticias)
    row = linhas[0]
    assert row["texto_límpido"] == "Título da matéria"
    assert row["texto_fallback_titulo"] == "True"
    assert rel["n_texto_fallback_titulo"] == 1


def test_ac_texto_presente_nao_faz_fallback(tmp_path):
    noticias = [
        {"titulo": "Título", "data": "2026-08-31", "url": "https://ifal.edu.br/y",
         "texto": "Corpo integral", "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
    ]
    rel, linhas = _cache_out(tmp_path, [], noticias)
    row = linhas[0]
    assert row["texto_límpido"] == "Corpo integral"
    assert row["texto_fallback_titulo"] == "False"


# --------------------------------------------------------------------------- #
# AC: tags HTML residuais removidas e entidades decodificadas
# --------------------------------------------------------------------------- #
def test_ac_html_residual_removido(tmp_path):
    inpi = [
        ("BR1", "2023-05-01", "<p>Titulo</p>&nbsp;<b>com</b>", "Resumo &amp; mais",
         "INSTITUTO FEDERAL DE X"),
    ]
    _, linhas = _cache_out(tmp_path, inpi, [])
    row = linhas[0]
    # tags removidas, &nbsp; -> espaço, &amp; -> &
    assert "<p>" not in row["texto_límpido"]
    assert "<b>" not in row["texto_límpido"]
    assert "&nbsp;" not in row["texto_límpido"]
    assert "&amp;" not in row["texto_límpido"]
    assert "Titulo com" in row["texto_límpido"]
    assert "Resumo & mais" in row["texto_límpido"]


def test_limpar_html_entidades_e_tags():
    assert cc.limpar_html("<p>Olá&nbsp;mundo</p>") == "Olá mundo"
    assert cc.limpar_html("<b>negrito &amp; sublinhado</b>") == "negrito & sublinhado"
    assert cc.limpar_html("") == ""
    assert cc.limpar_html(None) == ""
    assert cc.limpar_html("   múltiplos    espaços   ") == "múltiplos espaços"


# --------------------------------------------------------------------------- #
# AC: multi-instituições INPI => primeira vira instituicao, demais em aux
# --------------------------------------------------------------------------- #
def test_ac_inpi_multi_instituicoes(tmp_path):
    inpi = [
        ("BR1", "2023-05-01", "T", "R",
         "INSTITUTO FEDERAL DE X|INSTITUTO FEDERAL DE Y|COLEGIO PEDRO II"),
    ]
    rel, linhas = _cache_out(tmp_path, inpi, [])
    row = linhas[0]
    assert row["instituicao"] == "INSTITUTO FEDERAL DE X"
    assert row["instituicoes_aux"] == "INSTITUTO FEDERAL DE Y|COLEGIO PEDRO II"
    assert rel["n_multi_instituicoes"] == 1


# --------------------------------------------------------------------------- #
# AC: idempotência por chave canônica (re-execução sem duplicatas)
# --------------------------------------------------------------------------- #
def test_ac_idempotencia(tmp_path):
    inpi = [("BR1", "2023-05-01", "T", "R", "INSTITUTO FEDERAL DE X")]
    noticias = [
        {"titulo": "N", "data": "2026-08-31", "url": "https://ifal.edu.br/a",
         "texto": "t", "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
    ]
    inpi_p = _escrever_inpi(tmp_path, inpi)
    noti_p = _escrever_noticias(tmp_path, noticias)
    out = str(tmp_path / "corpus.csv")
    rel_path = str(tmp_path / "relatorio.json")

    cc.limpar_corpus(inpi_p, noti_p, out, rel_path)
    n1 = len(_ler_csv(out))
    cc.limpar_corpus(inpi_p, noti_p, out, rel_path)
    n2 = len(_ler_csv(out))

    assert n1 == n2 == 2  # sem duplicatas após re-execução


def test_ac_idempotencia_portal_mesma_url(tmp_path):
    # mesmo URL duas vezes no JSON de entrada => uma linha (merge por URL)
    noticias = [
        {"titulo": "N1", "data": "2026-08-31", "url": "https://ifal.edu.br/a",
         "texto": "t1", "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
        {"titulo": "N1 duplicado", "data": "2026-08-31", "url": "https://ifal.edu.br/a",
         "texto": "t2", "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
    ]
    _, linhas = _cache_out(tmp_path, [], noticias)
    assert len(linhas) == 1


# --------------------------------------------------------------------------- #
# AC: data brasileira e ISO normalizadas
# --------------------------------------------------------------------------- #
def test_normalizar_iso():
    assert cc._normalizar_iso("2023-05-01") == "2023-05-01"
    assert cc._normalizar_iso("31/08/2026") == "2026-08-31"
    assert cc._normalizar_iso("2023-05-01T10:00:00Z") == "2023-05-01"
    assert cc._normalizar_iso("") == ""
    assert cc._normalizar_iso(None) == ""
    assert cc._normalizar_iso("lixo") == ""
    assert cc._normalizar_iso("2023-13-40") == ""  # data inválida


# --------------------------------------------------------------------------- #
# CLI: exit code 0 e relatório coerente
# --------------------------------------------------------------------------- #
def test_cli_exit_0_e_relatorio_coerente(tmp_path):
    inpi = [
        ("BR1", "2023-05-01", "T1", "R1", "INSTITUTO FEDERAL DE X"),
        ("BR2", "2015-08-05", "T2", "R2", "INSTITUTO FEDERAL DE Y"),
        ("BR3", "", "T3", "R3", "INSTITUTO FEDERAL DE Z"),
    ]
    noticias = [
        {"titulo": "N1", "data": "2026-08-31", "url": "https://ifal.edu.br/a",
         "texto": "", "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
    ]
    inpi_p = _escrever_inpi(tmp_path, inpi)
    noti_p = _escrever_noticias(tmp_path, noticias)
    out = str(tmp_path / "corpus.csv")
    rel_path = str(tmp_path / "relatorio.json")

    codigo = cc.main(["--inpi", inpi_p, "--noticias", noti_p,
                      "--out", out, "--relatorio", rel_path])
    assert codigo == 0

    rel = json.load(open(rel_path, encoding="utf-8"))
    linhas = _ler_csv(out)
    assert rel["n_total"] == len(linhas) == 4
    assert rel["n_janela"] == 2      # BR1 (2023) + notícia (2026)
    assert rel["n_fora_janela"] == 1  # BR2 (2015)
    assert rel["n_sem_data"] == 1     # BR3
    assert rel["n_por_tipo"] == {"inpi": 3, "noticia": 1}
    assert rel["n_por_instituicao"]["INSTITUTO FEDERAL DE X"] == 1
    assert rel["n_por_instituicao"]["IFAL"] == 1
    # trilha de auditoria: leituras == itens únicos + duplicatas removidas
    assert rel["n_duplicatas_removidas"] == 0
    assert rel["n_inpi"] - rel["n_dup_inpi"] == rel["n_por_tipo"]["inpi"]
    assert rel["n_noticias"] - rel["n_dup_noticias"] == rel["n_por_tipo"]["noticia"]


# --------------------------------------------------------------------------- #
# PATCH 3: duplicatas removidas são registradas (trilha de auditoria)
# --------------------------------------------------------------------------- #
def test_duplicatas_removidas_registradas(tmp_path):
    # mesmo numero INPI duas vezes => 1 lida como duplicata
    inpi = [
        ("BR1", "2023-05-01", "T", "R", "INSTITUTO FEDERAL DE X"),
        ("BR1", "2023-05-01", "T", "R", "INSTITUTO FEDERAL DE X"),
    ]
    # mesma url de notícia duas vezes => 1 duplicata
    noticias = [
        {"titulo": "N", "data": "2026-08-31", "url": "https://ifal.edu.br/a",
         "texto": "t", "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
        {"titulo": "N dup", "data": "2026-08-31", "url": "https://ifal.edu.br/a",
         "texto": "t2", "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
    ]
    rel, linhas = _cache_out(tmp_path, inpi, noticias)
    # only 2 unique lines (1 INPI + 1 notícia)
    assert len(linhas) == 2
    assert rel["n_duplicatas_removidas"] == 2
    assert rel["n_dup_inpi"] == 1
    assert rel["n_dup_noticias"] == 1
    assert rel["n_inpi"] - rel["n_dup_inpi"] == rel["n_por_tipo"]["inpi"]
    assert rel["n_noticias"] - rel["n_dup_noticias"] == rel["n_por_tipo"]["noticia"]


# --------------------------------------------------------------------------- #
# PATCH 4: itens degenerados sem url/numero não colidem silenciosamente
# --------------------------------------------------------------------------- #
def test_fallback_chave_desambiguada_noticias():
    # dois itens sem url, mesma data e título => devem coexistir (sem colisão)
    noticias = [
        {"titulo": "Mesmo título", "data": "2026-08-31", "url": "",
         "texto": "um", "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
        {"titulo": "Mesmo título", "data": "2026-08-31", "url": "",
         "texto": "dois", "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
    ]
    import tempfile, pathlib  # noqa: F401
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        noti = _escrever_noticias(d, noticias)
        out = str(d / "corpus.csv")
        rel_path = str(d / "relatorio.json")
        rel = cc.limpar_corpus(str(d / "nope.csv"), noti, out, rel_path)
        linhas = _ler_csv(out)
        assert len(linhas) == 2  # ambos retidos
        textos = {r["texto_límpido"] for r in linhas}
        assert textos == {"um", "dois"}


def test_fallback_chave_desambiguada_inpi(tmp_path):
    # dois itens sem numero, mesma data e título => coexistir
    inpi = [
        ("", "2023-05-01", "T", "R", "INSTITUTO FEDERAL DE X"),
        ("", "2023-05-01", "T", "R", "INSTITUTO FEDERAL DE X"),
    ]
    rel, linhas = _cache_out(tmp_path, inpi, [])
    assert len(linhas) == 2
    assert rel["n_dup_inpi"] == 0


# --------------------------------------------------------------------------- #
# PATCH 5: itens não-dict no JSON são registrados nos erros
# --------------------------------------------------------------------------- #
def test_noticias_item_nao_dict_registrado(tmp_path):
    noticias = [
        {"titulo": "ok", "data": "2026-08-31", "url": "https://ifal.edu.br/a",
         "texto": "t", "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
        "isto não é um dict",
        42,
    ]
    rel, linhas = _cache_out(tmp_path, [], noticias)
    assert len(linhas) == 1  # só o item dict retido
    assert any("não-dict" in e for e in rel["erros"])
    assert any("[1]" in e and "str" in e for e in rel["erros"])
    assert any("[2]" in e and "int" in e for e in rel["erros"])


# --------------------------------------------------------------------------- #
# PATCH 7: roundtrip CSV preserva vírgulas, aspas e quebras de linha
# --------------------------------------------------------------------------- #
def test_roundtrip_csv_preserva_texto_complexo(tmp_path):
    # observe: limpar_html normaliza quebras de linha para espaço; commas/aspas
    # sobrevivem intactos à escrita/leitura do CSV.
    texto = "Linha 1, com vírgula e \"aspas duplas\"\nSegunda linha & caracteres: çãõ"
    noticias = [
        {"titulo": "Complexo", "data": "2026-08-31", "url": "https://ifal.edu.br/cplx",
         "texto": texto, "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
    ]
    _, linhas = _cache_out(tmp_path, [], noticias)
    row = linhas[0]
    assert '"' in row["texto_límpido"]
    assert "vírgula" in row["texto_límpido"]
    assert "e \"aspas duplas\" Segunda" in row["texto_límpido"]
    assert "çãõ" in row["texto_límpido"]


def test_roundtrip_csv_camada_cita_linhas_e_aspas(tmp_path):
    # teste direto da camada csv (DictWriter/DictReader) com valor contendo
    # vírgula, aspas duplas e quebra de linha literal — deve sobreviver intacto.
    valor = "a,b\"c\nd"
    p = tmp_path / "t.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["col"])
        writer.writeheader()
        writer.writerow({"col": valor})
    with open(p, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f))
    assert linhas[0]["col"] == valor


# --------------------------------------------------------------------------- #
# PATCH 8: ambas as fontes ausentes => CSV só com cabeçalho, erros registrados
# --------------------------------------------------------------------------- #
def test_ambas_fontes_ausentes(tmp_path):
    out = str(tmp_path / "corpus.csv")
    rel_path = str(tmp_path / "relatorio.json")
    rel = cc.limpar_corpus(str(tmp_path / "nao-inpi.csv"),
                           str(tmp_path / "nao-noticias.json"), out, rel_path)
    linhas = _ler_csv(out)
    assert linhas == []  # apenas cabeçalho
    assert rel["n_total"] == 0
    assert rel["n_inpi"] == 0
    assert rel["n_noticias"] == 0
    assert any("inpi_csv não encontrado" in e for e in rel["erros"])
    assert any("noticias_json não encontrado" in e for e in rel["erros"])


# --------------------------------------------------------------------------- #
# PATCH 9: exit 1 quando há erro registrado (fonte ausente)
# --------------------------------------------------------------------------- #
def test_main_exit_1_com_erro(tmp_path):
    noti_p = _escrever_noticias(tmp_path, [])
    out = str(tmp_path / "corpus.csv")
    rel_path = str(tmp_path / "relatorio.json")
    codigo = cc.main(["--inpi", str(tmp_path / "nao-existe.csv"),
                      "--noticias", noti_p, "--out", out, "--relatorio", rel_path])
    assert codigo == 1
    rel = json.load(open(rel_path, encoding="utf-8"))
    assert any("inpi_csv não encontrado" in e for e in rel["erros"])


# --------------------------------------------------------------------------- #
# Edges: fonte INPI ausente / malformado não aborta
# --------------------------------------------------------------------------- #
def test_edge_inpi_ausente_relatorio_erro(tmp_path):
    noti_p = _escrever_noticias(tmp_path, [
        {"titulo": "N", "data": "2026-08-31", "url": "https://ifal.edu.br/a",
         "texto": "t", "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
    ])
    out = str(tmp_path / "corpus.csv")
    rel_path = str(tmp_path / "relatorio.json")
    rel = cc.limpar_corpus(str(tmp_path / "nao-existe.csv"), noti_p, out, rel_path)
    assert os.path.basename(rel["metadados"]["out_csv"]) == "corpus.csv"
    assert "inpi_csv não encontrado" in ". ".join(rel["erros"])
    # mesmo sem INPI, notícias ainda são processadas
    assert rel["n_noticias"] == 1


# --------------------------------------------------------------------------- #
# Edges: noticias_json malformado não aborta
# --------------------------------------------------------------------------- #
def test_edge_noticias_malformado(tmp_path):
    inpi_p = _escrever_inpi(tmp_path, [("BR1", "2023-05-01", "T", "R", "INST")])
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    out = str(tmp_path / "corpus.csv")
    rel_path = str(tmp_path / "relatorio.json")
    rel = cc.limpar_corpus(inpi_p, str(p), out, rel_path)
    assert any("noticias_json" in e for e in rel["erros"])
    assert rel["n_inpi"] == 1


# --------------------------------------------------------------------------- #
# AC: texto conservado integralmente (frases completas)
# --------------------------------------------------------------------------- #
def test_ac_texto_conservado_integralmente(tmp_path):
    noticias = [
        {"titulo": "Título", "data": "2026-08-31", "url": "https://ifal.edu.br/z",
         "texto": "A inovação   tecnológica e o desenvolvimento regional são   centrais. "
                  "Frase completa preservada.",
         "portal": "https://ifal.edu.br", "instituicao": "IFAL"},
    ]
    _, linhas = _cache_out(tmp_path, [], noticias)
    row = linhas[0]
    # múltiplos espaços normalizados, mas frases completas preservadas
    assert "A inovação tecnológica e o desenvolvimento regional são centrais." in row["texto_límpido"]
    assert "Frase completa preservada." in row["texto_límpido"]
