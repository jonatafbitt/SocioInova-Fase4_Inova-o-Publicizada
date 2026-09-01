"""Suite pytest (Story 1.3) com zips sintéticos em disco (tmp_path).

Cobre os critérios de aceitação da Story 1.3: filtro por união (raiz nominal e/ou
crosswalk de CNPJ), case/accent-insensitivity, idempotência por ``numero_inpi``,
membro ausente sem exceção, idempotência de execução e CLI (exit code/relatório).
Sem rede e sem zip real (478 MB).
"""
import io
import json
import zipfile

import pandas as pd
import pytest

import scrapers.inpi_extrair as ie

BIB = "PATENTES_DADOS_BIBLIOGRAFICOS.csv"
CONT = "PATENTES_CONTEUDO.csv"
DEP = "PATENTES_DEPOSITANTES.csv"

HEAD_BIB = "codigo_interno,numero_inpi,data_deposito"
HEAD_CONT = "codigo_interno,numero_inpi,titulo,resumo"
HEAD_DEP = "codigo_interno,numero_inpi,ordem,depositante,cgccpfdepositante"


def _fazer_zip(tmp_path, bib="", cont="", dep="", nome="dump.zip"):
    """Grava um zip sintético com os membros fornecidos (CSV em utf-8)."""
    arq = tmp_path / nome
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if bib is not None:
            zf.writestr(BIB, (bib + "\n").encode("utf-8"))
        if cont is not None:
            zf.writestr(CONT, (cont + "\n").encode("utf-8"))
        if dep is not None:
            zf.writestr(DEP, (dep + "\n").encode("utf-8"))
    arq.write_bytes(buf.getvalue())
    return str(arq)


def _validacao_aceita(tmp_path):
    p = tmp_path / "inpi_validacao.json"
    p.write_text(json.dumps({"fr1": {"aceito": True}}), encoding="utf-8")
    return str(p)


def _validacao_pendente(tmp_path):
    p = tmp_path / "inpi_validacao.json"
    p.write_text(json.dumps({"fr1": {"aceito": False}}), encoding="utf-8")
    return str(p)


def _ler_csv(caminho):
    return pd.read_csv(caminho, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _assemble_cont(bib=(), cont=(), dep=()):
    """Constrói os textos CSV a partir de tuplas, ou usa strings direto."""
    texto = lambda x: "\n".join(x) if isinstance(x, tuple) else x
    return texto(bib), texto(cont), texto(dep)


# --------------------------------------------------------------------------- #
# AC1: filtro por união, raiz nominal, um registro por número
# --------------------------------------------------------------------------- #
def test_ac1_retem_raizes_e_um_registro_por_numero(tmp_path):
    bib = (
        HEAD_BIB,
        "b1,BR10001,2020-01-01",
        "b2,BR10002,2020-01-02",
        "b3,BR10003,2020-01-03",
        "b4,BR10004,2020-01-04",
        "b5,BR10005,2020-01-05",
    )
    cont = (
        HEAD_CONT,
        "c1,BR10001,Título A,Resumo A",
        "c2,BR10002,Título B,Resumo B",
        "c3,BR10003,Título C,Resumo C",
        "c4,BR10004,Título D,Resumo D",
        "c5,BR10005,Título E,Resumo E",
    )
    dep = (
        HEAD_DEP,
        "1,BR10001,1,INSTITUTO FEDERAL DE ALAGOAS,",
        "2,BR10002,1,UNIVERSIDADE FEDERAL X,",
        "3,BR10003,1,Centro Federal de Educação Tecnológica de MG,",
        "4,BR10004,1,Colégio Pedro II,",
        "5,BR10005,1,CENTRO FEDERAL DE EDUCACAO TECNOLOGICA MINAS GERAIS,",
    )
    zippath = _fazer_zip(tmp_path, * _assemble_cont(bib, cont, dep))
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    r = ie.extrair_filtrar(zippath, str(csv), str(rel), validacao_path=_validacao_aceita(tmp_path))

    df = _ler_csv(csv)
    nums = set(df["numero_inpi"])
    # retidos: IF, 2x Centro, Colégio; UNIVERSIDADE FEDERAL rejeitada (não-Rede)
    assert nums == {"BR10001", "BR10003", "BR10004", "BR10005"}
    assert r["n_registros_csv"] == 4
    assert r["n_descartados"] == 1
    assert r["n_casados_rede"] == 4
    assert df["numero_inpi"].duplicated().sum() == 0
    assert r["cobertura_por_raiz"]["INSTITUTO FEDERAL DE"] == 1
    assert r["cobertura_por_raiz"]["CENTRO FEDERAL DE EDUCAC"] == 2
    assert r["cobertura_por_raiz"]["COLEGIO PEDRO II"] == 1


# --------------------------------------------------------------------------- #
# AC2: co-titularidade — um da Rede, um não ⇒ retém só o da Rede
# --------------------------------------------------------------------------- #
def test_ac2_cotitularidade_lista_só_rede(tmp_path):
    bib = (HEAD_BIB, "b1,BR20001,2020-05-05")
    cont = (HEAD_CONT, "c1,BR20001,Titulo,Resumo")
    dep = (
        HEAD_DEP,
        "1,BR20001,1,INSTITUTO FEDERAL DE SAO PAULO,",
        "1,BR20001,2,EMPRESA COMERCIAL X,",
    )
    zippath = _fazer_zip(tmp_path, * _assemble_cont(bib, cont, dep))
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    r = ie.extrair_filtrar(zippath, str(csv), str(rel), validacao_path=_validacao_aceita(tmp_path))

    df = _ler_csv(csv)
    row = df[df["numero_inpi"] == "BR20001"].iloc[0]
    assert row["instituicoes"] == "INSTITUTO FEDERAL DE SAO PAULO"
    assert "EMPRESA COMERCIAL" not in row["instituicoes"]
    assert len(df) == 1
    # o depositante não-Rede foi descartado, o da Rede casado
    assert r["n_descartados"] == 1
    assert r["n_casados_rede"] == 1


# --------------------------------------------------------------------------- #
# AC3: codigo_interno duplicado no bibliográfico ⇒ numero_inpi único (first wins)
# --------------------------------------------------------------------------- #
def test_ac3_codigo_interno_duplicado_uma_vez(tmp_path):
    bib = (
        HEAD_BIB,
        "dup,BR30001,2020-01-01",
        "dup,BR30001,2020-02-02",
    )
    cont = (HEAD_CONT, "c1,BR30001,Titulo,Resumo")
    dep = (HEAD_DEP, "1,BR30001,1,INSTITUTO FEDERAL DE GOIAS,")
    zippath = _fazer_zip(tmp_path, * _assemble_cont(bib, cont, dep))
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    ie.extrair_filtrar(zippath, str(csv), str(rel), validacao_path=_validacao_aceita(tmp_path))

    df = _ler_csv(csv)
    assert len(df) == 1
    # primeiro ganha: data 2020-01-01
    assert df.iloc[0]["data_deposito"] == "2020-01-01"


# --------------------------------------------------------------------------- #
# AC4: membro CONTEUDO ausente ⇒ título/resumo vazios + relatório, sem exceção
# --------------------------------------------------------------------------- #
def test_ac4_conteudo_ausente_sem_excecao(tmp_path):
    bib = (HEAD_BIB, "b1,BR40001,2020-01-01")
    cont = None  # membro ausente
    dep = (HEAD_DEP, "1,BR40001,1,INSTITUTO FEDERAL DE RORAIMA,")
    zippath = _fazer_zip(tmp_path, bib="\n".join(bib), cont=None, dep="\n".join(dep))
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    r = ie.extrair_filtrar(zippath, str(csv), str(rel), validacao_path=_validacao_aceita(tmp_path))

    df = _ler_csv(csv)
    assert len(df) == 1
    assert df.iloc[0]["titulo"] == ""
    assert df.iloc[0]["resumo"] == ""
    assert "PATENTES_CONTEUDO.csv" in r["membros_ausentes"]


# --------------------------------------------------------------------------- #
# AC5: caixa mista e acentos ⇒ casada case/accent-insensitive
# --------------------------------------------------------------------------- #
def test_ac5_case_e_acento_insensivel(tmp_path):
    bib = (
        HEAD_BIB,
        "b1,BR50001,2020-01-01",
        "b2,BR50002,2020-01-02",
    )
    cont = (HEAD_CONT, "c1,BR50001,T1,R1", "c2,BR50002,T2,R2")
    dep = (
        HEAD_DEP,
        "1,BR50001,1,iNsTiTuTo FeDeRaL De GoIaS,",
        "2,BR50002,1,Centro FeDeRaL De EdUcAÇÃO TeCnOlÓgIcA,",
    )
    zippath = _fazer_zip(tmp_path, * _assemble_cont(bib, cont, dep))
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    r = ie.extrair_filtrar(zippath, str(csv), str(rel), validacao_path=_validacao_aceita(tmp_path))

    df = _ler_csv(csv)
    assert set(df["numero_inpi"]) == {"BR50001", "BR50002"}
    assert r["cobertura_por_raiz"]["INSTITUTO FEDERAL DE"] == 1
    assert r["cobertura_por_raiz"]["CENTRO FEDERAL DE EDUCAC"] == 1


# --------------------------------------------------------------------------- #
# Crosswalk de CNPJ: correspondência primária quando presente
# --------------------------------------------------------------------------- #
def test_crosswalk_cnpj_retem_sem_raiz_nominal(tmp_path):
    bib = (HEAD_BIB, "b1,BR60001,2020-01-01")
    cont = (HEAD_CONT, "c1,BR60001,T,R")
    dep = (HEAD_DEP, "1,BR60001,1,FUNDACAO DE APOIO XYZ,12.345.678/0001-90")
    zippath = _fazer_zip(tmp_path, * _assemble_cont(bib, cont, dep))
    cw = tmp_path / "crosswalk.json"
    cw.write_text(json.dumps(["12.345.678/0001-90"]), encoding="utf-8")
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    r = ie.extrair_filtrar(zippath, str(csv), str(rel), crosswalk_path=str(cw),
                           validacao_path=_validacao_aceita(tmp_path))
    df = _ler_csv(csv)
    assert set(df["numero_inpi"]) == {"BR60001"}
    assert r["cobertura_por_raiz"]["crosswalk_cnpj"] == 1


# --------------------------------------------------------------------------- #
# AC6: idempotência de execução — mesma saída, contagem igual
# --------------------------------------------------------------------------- #
def test_ac6_idempotencia_execucao(tmp_path):
    bib = (HEAD_BIB, "b1,BR70001,2020-01-01")
    cont = (HEAD_CONT, "c1,BR70001,T,R")
    dep = (HEAD_DEP, "1,BR70001,1,INSTITUTO FEDERAL DE MINAS GERAIS,")
    zippath = _fazer_zip(tmp_path, * _assemble_cont(bib, cont, dep))
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    vf = _validacao_aceita(tmp_path)

    r1 = ie.extrair_filtrar(zippath, str(csv), str(rel), validacao_path=vf)
    n1 = len(_ler_csv(csv))
    r2 = ie.extrair_filtrar(zippath, str(csv), str(rel), validacao_path=vf)
    n2 = len(_ler_csv(csv))

    assert n1 == n2 == 1
    assert r1["n_registros_csv"] == r2["n_registros_csv"]
    assert r1["n_casados_rede"] == r2["n_casados_rede"]


# --------------------------------------------------------------------------- #
# AC7: CLI + relatório coerente
# --------------------------------------------------------------------------- #
def test_cli_exit_0_e_relatorio_coerente(tmp_path, monkeypatch, capsys):
    bib = (
        HEAD_BIB,
        "b1,BR80001,2020-01-01",
        "b2,BR80002,2020-01-02",
        "b3,BR80003,2020-01-03",
    )
    cont = (HEAD_CONT, "c1,BR80001,T1,R1", "c2,BR80002,T2,R2", "c3,BR80003,T3,R3")
    dep = (
        HEAD_DEP,
        "1,BR80001,1,INSTITUTO FEDERAL DE ALAGOAS,",
        "2,BR80002,1,UNIVERSIDADE FEDERAL Z,",
        "3,BR80003,1,COLEGIO PEDRO II,",
    )
    zippath = _fazer_zip(tmp_path, * _assemble_cont(bib, cont, dep))
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    vf = _validacao_aceita(tmp_path)
    monkeypatch.setattr(ie, "VALIDA_DEFAULT", vf)

    codigo = ie.main(["--zip", zippath, "--csv", str(csv), "--relatorio", str(rel)])
    assert codigo == 0

    data = json.loads(rel.read_text(encoding="utf-8"))
    assert data["veredito"] == "ok"
    assert data["n_registros_csv"] == 2
    assert data["n_casados_rede"] == 2
    out = capsys.readouterr().out
    assert "veredito=ok" in out
    assert "n_registros_csv=2" in out


def test_cli_exit_1_quando_fr1_nao_aceito(tmp_path, monkeypatch):
    bib = (HEAD_BIB, "b1,BR90001,2020-01-01")
    cont = (HEAD_CONT, "c1,BR90001,T,R")
    dep = (HEAD_DEP, "1,BR90001,1,INSTITUTO FEDERAL DE ALAGOAS,")
    zippath = _fazer_zip(tmp_path, * _assemble_cont(bib, cont, dep))
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    monkeypatch.setattr(ie, "VALIDA_DEFAULT", _validacao_pendente(tmp_path))

    codigo = ie.main(["--zip", zippath, "--csv", str(csv), "--relatorio", str(rel)])
    assert codigo == 1
    data = json.loads(rel.read_text(encoding="utf-8"))
    assert data["veredito"] == "pendente"
    assert data["fr1_aceito"] is False


# =========================================================================== #
# P2 — membro corrompido: schema-discovery não aborta; erro visível no relatório
# =========================================================================== #
def test_p2_membro_corrompido_nao_aborta(tmp_path, monkeypatch):
    bib = (HEAD_BIB, "b1,BR10005,2020-01-01")
    cont = (HEAD_CONT, "c1,BR10005,T,R")
    dep = (HEAD_DEP, "1,BR10005,1,INSTITUTO FEDERAL DE ALAGOAS,")
    zippath = _fazer_zip(tmp_path, * _assemble_cont(bib, cont, dep))

    real_abrir = ie._abrir_membro

    def corrupta(zf, alvo):
        if alvo == CONT:
            raise RuntimeError("cabeçalho corrompido / head indecodável")
        return real_abrir(zf, alvo)

    monkeypatch.setattr(ie, "_abrir_membro", corrupta)
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    r = ie.extrair_filtrar(zippath, str(csv), str(rel), validacao_path=_validacao_aceita(tmp_path))

    # a extração COMPLETA (report escrito e CSV dos demais membros)
    assert rel.exists()
    assert "PATENTES_CONTEUDO.csv" in r["membros_ausentes"]
    assert r["membros"][CONT]["erro"] == "cabeçalho corrompido / head indecodável"
    df = _ler_csv(csv)
    assert set(df["numero_inpi"]) == {"BR10005"}  # demais membros processados


# =========================================================================== #
# P1 — múltiplos depositantes da Rede ⇒ instituicoes com TODOS os nomes (|)
# =========================================================================== #
def test_p1_dois_depositantes_rede_diferentes_raizes(tmp_path):
    bib = (HEAD_BIB, "b1,BR10010,2020-01-01")
    cont = (HEAD_CONT, "c1,BR10010,T,R")
    dep = (
        HEAD_DEP,
        "1,BR10010,1,INSTITUTO FEDERAL DE SAO PAULO,",
        "1,BR10010,2,Centro Federal de Educação Tecnológica,",
    )
    zippath = _fazer_zip(tmp_path, * _assemble_cont(bib, cont, dep))
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    ie.extrair_filtrar(zippath, str(csv), str(rel), validacao_path=_validacao_aceita(tmp_path))

    df = _ler_csv(csv)
    row = df[df["numero_inpi"] == "BR10010"].iloc[0]
    # ambos normalizados, junção por |, ordem de aparição (não-Rede nunca incluído)
    assert row["instituicoes"] == "INSTITUTO FEDERAL DE SAO PAULO|CENTRO FEDERAL DE EDUCACAO TECNOLOGICA"
    assert "BR10010" not in row["instituicoes"]


def test_p1_tres_depositantes_rede(tmp_path):
    bib = (HEAD_BIB, "b1,BR10011,2020-01-01")
    cont = (HEAD_CONT, "c1,BR10011,T,R")
    dep = (
        HEAD_DEP,
        "1,BR10011,1,INSTITUTO FEDERAL DE SAO PAULO,",
        "1,BR10011,2,Centro Federal de Educação Tecnológica de MG,",
        "1,BR10011,3,COLEGIO PEDRO II,",
    )
    zippath = _fazer_zip(tmp_path, * _assemble_cont(bib, cont, dep))
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    ie.extrair_filtrar(zippath, str(csv), str(rel), validacao_path=_validacao_aceita(tmp_path))

    df = _ler_csv(csv)
    row = df[df["numero_inpi"] == "BR10011"].iloc[0]
    instituicoes = row["instituicoes"].split("|")
    assert len(instituicoes) == 3
    assert instituicoes == [
        "INSTITUTO FEDERAL DE SAO PAULO",
        "CENTRO FEDERAL DE EDUCACAO TECNOLOGICA DE MG",
        "COLEGIO PEDRO II",
    ]


# =========================================================================== #
# P4 — membro DEPOSITANTES sem coluna ``depositante``, mas CNPJ no crosswalk
# =========================================================================== #
def test_p4_sem_coluna_depositante_mas_cnpj_no_crosswalk(tmp_path):
    bib = (HEAD_BIB, "b1,BR10020,2020-01-01")
    cont = (HEAD_CONT, "c1,BR10020,T,R")
    # membro de depositantes SEM a coluna ``depositante``
    head_dep_sem_nome = "codigo_interno,numero_inpi,ordem,cgccpfdepositante"
    dep = (
        head_dep_sem_nome,
        "1,BR10020,1,12.345.678/0001-90",
    )
    zippath = _fazer_zip(tmp_path, * _assemble_cont(bib, cont, dep))
    cw = tmp_path / "crosswalk.json"
    cw.write_text(json.dumps(["12.345.678/0001-90"]), encoding="utf-8")
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    r = ie.extrair_filtrar(zippath, str(csv), str(rel), crosswalk_path=str(cw),
                           validacao_path=_validacao_aceita(tmp_path))

    df = _ler_csv(csv)
    assert set(df["numero_inpi"]) == {"BR10020"}
    # sem nome disponível ⇒ rótulo = CNPJ em dígitos (normalizado)
    assert df.iloc[0]["instituicoes"] == "12345678000190"
    assert r["n_casados_rede"] == 1
    assert r["n_descartados"] == 0
    assert r["cobertura_por_raiz"]["crosswalk_cnpj"] == 1


def test_p4_sem_coluna_depositante_sem_cnpj_no_crosswalk_descarta(tmp_path):
    bib = (HEAD_BIB, "b1,BR10021,2020-01-01")
    cont = (HEAD_CONT, "c1,BR10021,T,R")
    head_dep_sem_nome = "codigo_interno,numero_inpi,ordem,cgccpfdepositante"
    dep = (head_dep_sem_nome, "1,BR10021,1,99.999.999/0001-99")
    zippath = _fazer_zip(tmp_path, * _assemble_cont(bib, cont, dep))
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    r = ie.extrair_filtrar(zippath, str(csv), str(rel), validacao_path=_validacao_aceita(tmp_path))

    df = _ler_csv(csv)
    # CNPJ não casa no (ausente) crosswalk e não há nome ⇒ descartado
    assert len(df) == 0
    assert r["n_casados_rede"] == 0
    assert r["n_descartados"] == 1


# =========================================================================== #
# P5 — linhas malformadas contabilizadas e excluídas
# =========================================================================== #
def test_p5_linha_malformada_contabilizada_e_ausente(tmp_path):
    bib = (
        HEAD_BIB,
        "b1,BR10030,2020-01-01",
        # linha malformada: número de campos maior que o cabeçalho ⇒ rejeitada
        "b2,BR10031,2020-01-02,EXTRAPOLOU,EXTRA",
        "b3,BR10032,2020-01-03",
    )
    cont = (HEAD_CONT, "c1,BR10030,T1,R1", "c3,BR10032,T3,R3")
    dep = (
        HEAD_DEP,
        "1,BR10030,1,INSTITUTO FEDERAL DE ALAGOAS,",
        "3,BR10032,1,INSTITUTO FEDERAL DE GOIAS,",
    )
    zippath = _fazer_zip(tmp_path, * _assemble_cont(bib, cont, dep))
    csv = tmp_path / "out.csv"
    rel = tmp_path / "ext.json"
    r = ie.extrair_filtrar(zippath, str(csv), str(rel), validacao_path=_validacao_aceita(tmp_path))

    assert r["linhas_rejeitadas"] >= 1
    df = _ler_csv(csv)
    assert set(df["numero_inpi"]) == {"BR10030", "BR10032"}
    # a linha malformada (p/ BR10031) não aparece no CSV
    assert "BR10031" not in set(df["numero_inpi"])


# =========================================================================== #
# P6 — freeze metodológico + crosswalk dict
# =========================================================================== #
def test_p6_raizes_rfepct_congeladas():
    assert ie.RAIZES_RFEPCT == [
        "INSTITUTO FEDERAL DE",
        "CENTRO FEDERAL DE EDUCAC",
        "COLEGIO PEDRO II",
    ]


def test_p6_crosswalk_forma_dict(tmp_path):
    cw = tmp_path / "crosswalk.json"
    cw.write_text(json.dumps({"ifsp": "12.345.678/0001-90", "outro": "00.000.000/0001-00"}),
                   encoding="utf-8")
    s = ie._carregar_crosswalk(str(cw))
    assert s == {"12345678000190", "00000000000100"}  # só 14 dígitos, separadores removidos

