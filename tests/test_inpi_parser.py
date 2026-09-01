"""Suite pytest (Story 1.2) com zips sintéticos em memória.

Cobre o veredito tri-condicional positivo e o caminho negativo (cada falha
isolada => ``pendente``) sem depender de rede ou do zip real de 478 MB.
Verifica o COMPORTAMENTO do módulo — não um snapshot JSON em disco.
"""
import io
import json
import zipfile

import pandas as pd
import pytest

import scrapers.inpi_parser as ip

COLS = list(ip.COLUNAS_DOCUMENTADAS)
IDX_DATA = COLS.index("data_deposito")
MEMBRO = "PATENTES_DADOS_BIBLIOGRAFICOS.csv"
MEMBRO_DEP = "PATENTES_DEPOSITANTES.csv"


def _linha(data: str | None) -> str:
    vals = ["x"] * len(COLS)
    if data is not None:
        vals[IDX_DATA] = data
    return ",".join(vals)


def fazer_zip(cabecalho: str, linhas: list[str], nome: str = MEMBRO) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(nome, (cabecalho + "\n" + "\n".join(linhas) + "\n").encode("utf-8"))
    buf.seek(0)
    return zipfile.ZipFile(buf)


def cabecalho_documentado() -> str:
    return ",".join(COLS)


def _zip_janela_boa() -> zipfile.ZipFile:
    """Universo com todos os anos 2019-2026 e pct_na_janela >= 10%."""
    linhas = [_linha(f"{ano}-06-01") for ano in range(2019, 2027) for _ in range(5)]
    linhas += [_linha("1999-01-01") for _ in range(200)]  # fora da Janela (universo)
    linhas += [_linha("2030-01-01")]  # futuro (universo)
    return fazer_zip(cabecalho_documentado(), linhas)


def _zip_janela_boa_mb() -> bytes:
    """Versão do zip em memória (bytes) para os testes de ``main()``."""
    linhas = [_linha(f"{ano}-06-01") for ano in range(2019, 2027) for _ in range(5)]
    linhas += [_linha("1999-01-01") for _ in range(200)]
    linhas += [_linha("2030-01-01")]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            MEMBRO,
            (cabecalho_documentado() + "\n" + "\n".join(linhas) + "\n").encode("utf-8"),
        )
        zf.writestr(MEMBRO_DEP, "codigo_interno,numero_inpi,depositante,cgccpfdepositante\n1,2,x,123\n".encode("utf-8"))
    return buf.getvalue()


class _FakePsutil:
    """Substituto de ``psutil.Process()`` para os testes de ``main()``."""
    def __init__(self):
        self._mem = 1_000_000

    class _Proc:
        def __init__(self, mem):
            self._mem = mem
        def memory_info(self):
            return type("MI", (), {"rss": self._mem})()

    def Process(self):
        return self._Proc(self._mem)


class _FakeResp:
    def __init__(self, url="", status_code=200, headers=None):
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}


# --------------------------------------------------------------------------- #
# Veredito tri-condicional
# --------------------------------------------------------------------------- #
def test_veredito_aprovado_somente_com_todos_verdadeiros():
    assert ip.computar_veredito(True, True, True) == "aprovado"


def test_veredito_pendente_cada_falha_isolada():
    casos = [
        (False, True, True),   # endpoint falha
        (True, False, True),   # schema falha
        (True, True, False),   # data falha
        (False, False, True),  # endpoint+schema
        (False, True, False),  # endpoint+data
        (True, False, False),  # schema+data
        (False, False, False), # tudo falha
    ]
    for c in casos:
        assert ip.computar_veredito(*c) == "pendente"


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
def test_endpoint_https_ok(monkeypatch):
    class Resp:
        url = "https://dadosabertos.inpi.gov.br/"
        status_code = 200

    monkeypatch.setattr(ip.requests, "get", lambda *a, **k: Resp())
    r = ip.validar_endpoint()
    assert r["https_ok"] is True
    assert r["alcancavel"] is True


def test_endpoint_nao_https(monkeypatch):
    class Resp:
        url = "http://dadosabertos.inpi.gov.br/"
        status_code = 200

    monkeypatch.setattr(ip.requests, "get", lambda *a, **k: Resp())
    assert ip.validar_endpoint()["https_ok"] is False


def test_endpoint_erro(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("sem rede")

    monkeypatch.setattr(ip.requests, "get", boom)
    r = ip.validar_endpoint()
    assert r["https_ok"] is False
    assert r["erro"]


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_schema_ok_com_colunas_documentadas():
    zf = fazer_zip(cabecalho_documentado(), _linha("2019-01-01"))
    schema, colunas = ip.validar_schema(zf, MEMBRO)
    assert schema["schema_ok"] is True
    assert schema["colunas_ausentes"] == []
    assert all(v == "presente" for v in schema["colunas_esperadas"].values())
    assert "data_deposito" in colunas


def test_schema_drift_delimitador_e_coluna_extra_reportados_mas_nao_falham():
    header = cabecalho_documentado() + ",sigilo"
    zf = fazer_zip(header, _linha("2019-01-01"))
    schema, colunas = ip.validar_schema(zf, MEMBRO)
    assert schema["schema_ok"] is True  # falta de coluna não é drift bloqueante
    assert schema["drift_delimitador"] is True  # real ",", documentado ";"
    assert schema["delimitador_detectado"] == ","
    assert "sigilo" in schema["colunas_extras_drift"]


def test_schema_coluna_ausente_falha():
    header = ",".join(c for c in COLS if c != "data_deposito")
    zf = fazer_zip(header, _linha(None))
    schema, _ = ip.validar_schema(zf, MEMBRO)
    assert schema["schema_ok"] is False
    assert "data_deposito" in schema["colunas_ausentes"]


def test_schema_membro_nao_encontrado():
    zf = fazer_zip(cabecalho_documentado(), _linha("2019-01-01"))
    schema, _ = ip.validar_schema(zf, "NAO_EXISTE.csv")
    assert schema["schema_ok"] is False
    assert schema["membro_encontrado"] is False


# --------------------------------------------------------------------------- #
# Dados (data_deposito)
# --------------------------------------------------------------------------- #
def test_data_representatividade_ok_universo_separado_da_janela():
    zf = _zip_janela_boa()
    data = ip.parsear_e_validar_data(zf, MEMBRO, ",")
    assert data["parseavel"] is True
    assert data["representatividade_ok"] is True
    assert data["qualidade"] == "usavel"
    assert data["janela_2019_2026"] is True
    assert data["campo"] == "data_deposito"
    # cobertura anual completa 2019-2026
    for ano in range(2019, 2027):
        assert str(ano) in data["cobertura_por_ano"]
        assert data["cobertura_por_ano"][str(ano)] >= 1
    assert data["pct_na_janela"] >= 0.10
    # universo (bruto) <> limites da Janela — B1 fix
    assert data["min_bruto"] == "1999-01-01"
    assert data["max_bruto"] == "2030-01-01"
    assert data["min_janela"] == "2019-06-01"
    assert data["max_janela"] == "2026-06-01"
    assert data["min_bruto"] != data["min_janela"]


def test_data_ano_da_janela_ausente_pendente():
    linhas = [_linha(f"{ano}-06-01") for ano in range(2019, 2026) for _ in range(5)]  # falta 2026
    linhas += [_linha("1999-01-01") for _ in range(100)]
    zf = fazer_zip(cabecalho_documentado(), linhas)
    data = ip.parsear_e_validar_data(zf, MEMBRO, ",")
    assert data["representatividade_ok"] is False
    assert data["janela_2019_2026"] is False
    assert data["qualidade"] == "ruim"
    assert data["cobertura_por_ano"]["2026"] == 0


def test_data_pct_na_janela_abaixo_limiar_pendente():
    linhas = [_linha(f"{ano}-06-01") for ano in range(2019, 2027)]  # 8 na janela
    linhas += [_linha("1999-01-01") for _ in range(1000)]  # dilui pct
    zf = fazer_zip(cabecalho_documentado(), linhas)
    data = ip.parsear_e_validar_data(zf, MEMBRO, ",")
    assert data["representatividade_ok"] is False
    assert data["qualidade"] == "ruim"
    assert data["pct_na_janela"] < 0.10


def test_data_membro_ausente_nao_parseavel():
    zf = fazer_zip(cabecalho_documentado(), _linha("2019-01-01"))
    data = ip.parsear_e_validar_data(zf, "NAO_EXISTE.csv", ",")
    assert data["parseavel"] is False
    assert data["representatividade_ok"] is False


# --------------------------------------------------------------------------- #
# Depositantes (crosswalk)
# --------------------------------------------------------------------------- #
def test_depositantes_coluna_presente():
    header = "codigo_interno,numero_inpi,depositante,cgccpfdepositante"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MEMBRO_DEP, (header + "\n1,2,x,12345\n").encode("utf-8"))
    buf.seek(0)
    zf = zipfile.ZipFile(buf)
    dep = ip.validar_depositantes(zf, MEMBRO_DEP)
    assert dep["membro_encontrado"] is True
    assert dep["coluna_cgccpfdepositante_presente"] is True


def test_depositantes_coluna_ausente():
    header = "codigo_interno,numero_inpi,depositante"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MEMBRO_DEP, (header + "\n1,2,x\n").encode("utf-8"))
    buf.seek(0)
    zf = zipfile.ZipFile(buf)
    dep = ip.validar_depositantes(zf, MEMBRO_DEP)
    assert dep["coluna_cgccpfdepositante_presente"] is False


# --------------------------------------------------------------------------- #
# Fluxo de veredito integrado (sem rede/zip real)
# --------------------------------------------------------------------------- #
def test_fluxo_aprovado_com_zip_sintetico():
    zf = _zip_janela_boa()
    schema, colunas = ip.validar_schema(zf, MEMBRO)
    data = ip.parsear_e_validar_data(zf, MEMBRO, schema["delimitador_detectado"])
    veredito = ip.computar_veredito(True, schema["schema_ok"], data["representatividade_ok"])
    assert veredito == "aprovado"


def test_fluxo_pendente_quando_data_falha():
    zf = fazer_zip(cabecalho_documentado(), [_linha("1999-01-01")])
    schema, _ = ip.validar_schema(zf, MEMBRO)
    data = ip.parsear_e_validar_data(zf, MEMBRO, schema["delimitador_detectado"])
    assert data["representatividade_ok"] is False
    veredito = ip.computar_veredito(True, schema["schema_ok"], data["representatividade_ok"])
    assert veredito == "pendente"


def test_fluxo_pendente_quando_schema_falha():
    header = ",".join(c for c in COLS if c != "data_deposito")
    zf = fazer_zip(header, [_linha(None)])
    schema, _ = ip.validar_schema(zf, MEMBRO)
    assert schema["schema_ok"] is False
    veredito = ip.computar_veredito(True, schema["schema_ok"], True)
    assert veredito == "pendente"


# =========================================================================== #
# Loopback 2 — data estrita ISO + plausibilidade (n_nao_plausiveis)
# =========================================================================== #
def test_data_iso_estrita_rejeita_sufixo_lixo():
    # B3: "2019-06-01junk" antes passava por valor[:10]; agora é rejeitado
    linhas = [_linha("2019-06-01junk")]
    zf = fazer_zip(cabecalho_documentado(), linhas)
    data = ip.parsear_e_validar_data(zf, MEMBRO, ",")
    assert data["n_parseaveis"] == 0
    assert data["proporcao_parseaveis"] == 0.0
    assert data["representatividade_ok"] is False


def test_data_conta_nao_plausiveis_anos_extremos():
    # anos 1 e 1111 (artefatos de serial do dump) são ISO-parseáveis mas não-plausíveis
    linhas = [_linha("0001-01-01"), _linha("1111-01-01"), _linha("2019-06-01")]
    zf = fazer_zip(cabecalho_documentado(), linhas)
    data = ip.parsear_e_validar_data(zf, MEMBRO, ",")
    assert data["n_parseaveis"] == 3
    assert data["n_nao_plausiveis"] == 2
    assert data["min_bruto"] == "0001-01-01"


def test_data_min_bruto_separa_universo_da_janela_com_extremos():
    linhas = [_linha("0001-01-01"), _linha("2200-01-01")]
    linhas += [_linha(f"{ano}-06-01") for ano in range(2019, 2027)]
    zf = fazer_zip(cabecalho_documentado(), linhas)
    data = ip.parsear_e_validar_data(zf, MEMBRO, ",")
    assert data["n_nao_plausiveis"] == 2
    assert data["min_bruto"] == "0001-01-01"
    assert data["max_bruto"] == "2200-01-01"
    assert data["min_janela"] == "2019-06-01"
    assert data["max_janela"] == "2026-06-01"


# =========================================================================== #
# Loopback 2 — proporcao_parseaveis >= 50% alimenta representatividade
# =========================================================================== #
def test_100_pct_na_janela_mas_maioria_nao_parseavel_pendente():
    # B4: 100% das datas VÁLIDAS estão na Janela, mas maioria das linhas não parseia
    linhas = [_linha("2019-06-01"), _linha("2020-06-01"), _linha("2021-06-01"),
              _linha("2022-06-01"), _linha("2023-06-01"),
              _linha("data-invalida"), _linha("data-invalida")]
    zf = fazer_zip(cabecalho_documentado(), linhas)
    data = ip.parsear_e_validar_data(zf, MEMBRO, ",")
    # 5/7 parseáveis (~71%) no caso acima ainda >= 50%; diluir mais para < 50%
    linhas = [_linha("2019-06-01"), _linha("2020-06-01"),
              _linha("lixo1"), _linha("lixo2"), _linha("lixo3")]
    zf = fazer_zip(cabecalho_documentado(), linhas)
    data = ip.parsear_e_validar_data(zf, MEMBRO, ",")
    assert data["proporcao_parseaveis"] < ip.LIMIAR_PROPR_PARSEAVEIS
    # todas as linhas parseáveis na Janela => pct_na_janela = 1.0, mas proporcao baixa
    assert data["pct_na_janela"] >= 0.10
    assert data["representatividade_ok"] is False


def test_parsear_data_iso_aceita_hora_opcional():
    dt, plausivel = ip.parsear_data_iso("2019-06-01T00:00:00")
    assert dt is not None and dt.year == 2019
    assert plausivel is True
    dt2, p2 = ip.parsear_data_iso("2020-01-02 10:30")
    assert dt2 is not None and p2 is True
    dt3, _ = ip.parsear_data_iso(" 2019-06-01 ")
    assert dt3 is not None


def test_parsear_data_iso_plausibilidade_extremos():
    _, p_low = ip.parsear_data_iso("1799-01-01")
    _, p_high = ip.parsear_data_iso("2101-01-01")
    assert p_low is False and p_high is False
    _, p_ok = ip.parsear_data_iso("1800-01-01")
    assert p_ok is True


def test_fluxo_pendente_quando_memoria_excede_limite():
    # quadrivalente: memoria fora do limite => pendente mesmo com resto ok
    veredito = ip.computar_veredito(True, True, True, memoria_ok=False)
    assert veredito == "pendente"


def test_veredito_quadrivalente_exige_memoria():
    assert ip.computar_veredito(True, True, True, True) == "aprovado"
    assert ip.computar_veredito(True, True, True, False) == "pendente"


# =========================================================================== #
# Loopback 2 — main() / contrato do relatório
# =========================================================================== #
def test_main_exit_0_somente_quando_aprovado(monkeypatch, tmp_path, capsys):
    zf_bytes = _zip_janela_boa_mb()
    arq_zip = tmp_path / "dump.zip"
    arq_zip.write_bytes(zf_bytes)

    chamado = {"val_endpoint": False, "dump": False}

    def fake_endpoint():
        chamado["val_endpoint"] = True
        return {"https_ok": True, "host_ok": True, "status_ok": True, "alcancavel": True}

    def fake_garantir_dump(dump, url, dir_raw, progresso=True):
        chamado["dump"] = True
        return {"dump": dump, "baixado": True, "era_local": True, "integro": True,
                "bytes": len(zf_bytes), "caminho": str(arq_zip), "url": url}

    monkeypatch.setattr(ip, "validar_endpoint", fake_endpoint)
    monkeypatch.setattr(ip, "garantir_dump", fake_garantir_dump)
    monkeypatch.setattr(ip, "DIR_RAW", str(tmp_path))
    monkeypatch.setattr(ip, "REPORT_PATH", str(tmp_path / "inpi_validacao.json"))
    monkeypatch.setattr(ip, "psutil", _FakePsutil())

    codigo = ip.main(["--dump", "dump.zip", "--membro", MEMBRO, "--no-progresso"])
    assert codigo == 0
    out = capsys.readouterr().out
    assert "veredito=aprovado" in out
    # relatório regenerado
    arq = tmp_path / "inpi_validacao.json"
    assert arq.exists()
    rel = json.loads(arq.read_text(encoding="utf-8"))
    assert rel["veredito"] == "aprovado"
    assert rel["fr1"]["status"] == "aprovado"
    assert rel["fr1"]["memoria"] is True
    assert rel["memoria"]["dentro_do_limite"] is True


def test_main_exit_1_e_contrato_quando_nao_aprovado(monkeypatch, tmp_path, capsys):
    zf_bytes = _zip_janela_boa_mb()
    arq_zip = tmp_path / "dump.zip"
    arq_zip.write_bytes(zf_bytes)

    def fake_endpoint():
        return {"https_ok": False, "host_ok": False, "status_ok": False, "alcancavel": False}

    def fake_garantir_dump(dump, url, dir_raw, progresso=True):
        return {"dump": dump, "baixado": True, "era_local": True, "integro": True,
                "bytes": len(zf_bytes), "caminho": str(arq_zip), "url": url}

    monkeypatch.setattr(ip, "validar_endpoint", fake_endpoint)
    monkeypatch.setattr(ip, "garantir_dump", fake_garantir_dump)
    monkeypatch.setattr(ip, "DIR_RAW", str(tmp_path))
    monkeypatch.setattr(ip, "REPORT_PATH", str(tmp_path / "inpi_validacao.json"))
    monkeypatch.setattr(ip, "psutil", _FakePsutil())

    codigo = ip.main(["--dump", "dump.zip", "--membro", MEMBRO])
    assert codigo == 1
    rel = json.loads((tmp_path / "inpi_validacao.json").read_text(encoding="utf-8"))
    assert rel["veredito"] == "pendente"
    assert rel["fr1"]["status"] == "pendente"
    assert rel["fr1"]["memoria"] == rel["memoria"]["dentro_do_limite"]
    # contrato estável: chaves de seção presentes mesmo em erro
    assert set(("endpoint", "schema", "data", "download", "memoria", "depositantes",
                "metadados")).issubset(rel)


def test_main_relatorio_caminho_relativo_sem_drive(monkeypatch, tmp_path):
    zf_bytes = _zip_janela_boa_mb()
    arq_zip = tmp_path / "dump.zip"
    arq_zip.write_bytes(zf_bytes)

    def fake_endpoint():
        return {"https_ok": True, "host_ok": True, "status_ok": True, "alcancavel": True}

    def fake_garantir_dump(dump, url, dir_raw, progresso=True):
        return {"dump": dump, "baixado": True, "era_local": True, "integro": True,
                "bytes": len(zf_bytes), "caminho": str(arq_zip), "url": url}

    monkeypatch.setattr(ip, "validar_endpoint", fake_endpoint)
    monkeypatch.setattr(ip, "garantir_dump", fake_garantir_dump)
    monkeypatch.setattr(ip, "DIR_RAW", str(tmp_path))
    monkeypatch.setattr(ip, "REPORT_PATH", str(tmp_path / "inpi_validacao.json"))
    monkeypatch.setattr(ip, "psutil", _FakePsutil())

    ip.main(["--dump", "dump.zip", "--membro", MEMBRO, "--no-progresso"])
    rel = json.loads((tmp_path / "inpi_validacao.json").read_text(encoding="utf-8"))
    # caminho relativo: apenas o nome, sem drive absoluto (:\\)
    assert rel["download"]["caminho"] == "dump.zip"
    assert ":\\" not in rel["download"]["caminho"]


def test_main_erro_nao_abre_zip_entra_no_contrato(monkeypatch, tmp_path):
    def fake_endpoint():
        return {"https_ok": True, "host_ok": True, "status_ok": True, "alcancavel": True}

    def fake_garantir_dump(dump, url, dir_raw, progresso=True):
        return {"dump": dump, "baixado": True, "era_local": False, "integro": True,
                "bytes": 1, "caminho": str(tmp_path / "nao_existe.zip"), "url": url}

    monkeypatch.setattr(ip, "validar_endpoint", fake_endpoint)
    monkeypatch.setattr(ip, "garantir_dump", fake_garantir_dump)
    monkeypatch.setattr(ip, "DIR_RAW", str(tmp_path))
    monkeypatch.setattr(ip, "REPORT_PATH", str(tmp_path / "inpi_validacao.json"))
    monkeypatch.setattr(ip, "psutil", _FakePsutil())

    codigo = ip.main(["--dump", "nao_existe.zip", "--membro", MEMBRO, "--no-progresso"])
    assert codigo == 1
    rel = json.loads((tmp_path / "inpi_validacao.json").read_text(encoding="utf-8"))
    assert rel["veredito"] == "pendente"
    assert rel["fr1"]["status"] == "pendente"
    # contrato estável: chaves de seção presentes mesmo quando o zip não abre
    assert set(("endpoint", "schema", "data", "download", "memoria", "depositantes",
                "metadados")).issubset(rel)
    assert set(("veredito", "fr1", "janela")).issubset(rel)


# =========================================================================== #
# Loopback 2 — garantir_dump / _zip_integro (offline/sintético)
# =========================================================================== #
def test_zip_integro_recusa_tamanho_divergente(monkeypatch, tmp_path):
    arq = tmp_path / "falso.zip"
    arq.write_bytes(b"\x00" * 100)
    entregue = {"conteudo_esperado": 12345}
    monkeypatch.setattr(ip.os.path, "getsize", lambda p: 100)
    integro, erro = ip._zip_integro(str(arq), 12345)
    assert integro is False
    assert "tamanho" in erro.lower() or "12345" in erro


def test_zip_integro_recusa_zip_truncado(tmp_path):
    arq = tmp_path / "truncado.zip"
    arq.write_bytes(b"PK\x03\x04 not a complete zip")
    integro, erro = ip._zip_integro(str(arq), None)
    assert integro is False
    assert erro


def test_zip_integro_recusa_crc_invalido(tmp_path):
    # zip ZIP_STORED com payload distinto; corrompe um byte da região de dados
    payload = b"A" * 2000
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("x.csv", payload)
    dados = bytearray(buf.getvalue())
    namelen = int.from_bytes(dados[26:28], "little")
    extralen = int.from_bytes(dados[28:30], "little")
    data_off = 30 + namelen + extralen
    dados[data_off] = 0x42  # corrompe o conteúdo armazenado (quebra CRC)
    arq = tmp_path / "crc.zip"
    arq.write_bytes(bytes(dados))
    integro, erro = ip._zip_integro(str(arq), None)
    assert integro is False
    assert erro


def test_garantir_dump_reusa_local_integro(monkeypatch, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MEMBRO, b"a,b\n1,2\n")
    dados = buf.getvalue()
    destino = tmp_path / "dump.zip"
    destino.write_bytes(dados)

    cab = str(len(dados))  # Content-Length correto
    monkeypatch.setattr(ip.requests, "head",
                        lambda *a, **k: _FakeResp(url="", status_code=200, headers={"Content-Length": cab}))
    baixado = {"sim": False}

    def fake_get(*a, **k):
        baixado["sim"] = True
        raise AssertionError("não deveria baixar de novo")

    monkeypatch.setattr(ip.requests, "get", fake_get)
    r = ip.garantir_dump("dump.zip", "http://x", str(tmp_path), progresso=False)
    assert r["integro"] is True
    assert r["era_local"] is True
    assert baixado["sim"] is False


def test_garantir_dump_redownload_quando_integro_falha(monkeypatch, tmp_path):
    destino = tmp_path / "dump.zip"
    destino.write_bytes(b"PK\x03\x04 corrompido")  # inválido, tamanho divergente
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MEMBRO, b"a,b\n1,2\n")
    dados = buf.getvalue()

    monkeypatch.setattr(ip.requests, "head",
                        lambda *a, **k: _FakeResp(url="", status_code=200,
                                                  headers={"Content-Length": str(len(dados))}))
    baixado = {"sim": False}

    class Resp:
        status_code = 200
        headers = {"Content-Length": str(len(dados))}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_content(self, chunk_size=1024):
            yield dados

    def fake_get(*a, **k):
        baixado["sim"] = True
        return Resp()

    monkeypatch.setattr(ip.requests, "get", fake_get)
    r = ip.garantir_dump("dump.zip", "http://x", str(tmp_path), progresso=False)
    assert baixado["sim"] is True
    assert r["integro"] is True
    assert r["era_local"] is False
    assert (tmp_path / "dump.zip").exists()


# =========================================================================== #
# Loopback 2 — encoding / header
# =========================================================================== #
def test_detectar_encoding_bom_utf8_sig():
    assert ip.detectar_encoding(b"\xef\xbb\xbf" + b"a,b") == "utf-8-sig"


def test_detectar_encoding_fallback_cp1252():
    # "olá" em cp1252: 0xE1 não é utf-8 válido sozinho
    assert ip.detectar_encoding(b"ol\xe1") == "cp1252"


def test_ler_cabecalho_sem_nova_linha():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MEMBRO, b"a,b,c")
    buf.seek(0)
    zf = zipfile.ZipFile(buf)
    cab, truncado = ip.ler_cabecalho(zf, MEMBRO)
    assert cab == b"a,b,c"
    assert truncado is False


def test_ler_cabecalho_flag_truncado_quando_estoura():
    linha = b"a," * (200 * 1024)  # sem \n, acima de 256KB no total
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MEMBRO, linha)
    buf.seek(0)
    zf = zipfile.ZipFile(buf)
    cab, truncado = ip.ler_cabecalho(zf, MEMBRO, max_bytes=8 * 1024)
    assert truncado is True
    assert len(cab) <= 8 * 1024


def test_schema_reporta_cabecalho_truncado():
    linha = b"a," * (200 * 1024)  # sem \n, estoura 256KB
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MEMBRO, linha)
    buf.seek(0)
    zf = zipfile.ZipFile(buf)
    schema, _ = ip.validar_schema(zf, MEMBRO)
    assert "cabecalho_truncado" in schema
    assert schema["cabecalho_truncado"] is True


# =========================================================================== #
# Loopback 2 — delimitador degenerado
# =========================================================================== #
def test_detectar_delimitador_cai_para_documentado_quando_zero():
    # nenhum candidato com contagem > 0 => fallback ao documentado ";" (B: ;)
    assert ip.detectar_delimitador(b"UMACOLUNA", "utf-8") == ip.DELIMITADOR_DOCUMENTADO
    assert ip.detectar_delimitador(b"", "utf-8") == ip.DELIMITADOR_DOCUMENTADO


def test_detectar_delimitador_normal():
    assert ip.detectar_delimitador(b"a,b,c", "utf-8") == ","
    assert ip.detectar_delimitador(b"a;b;c", "utf-8") == ";"


# =========================================================================== #
# Loopback 2 — endpoint status >= 400 e host/schema na URL resolvida
# =========================================================================== #
def test_endpoint_status_400_falha(monkeypatch):
    class Resp:
        url = "https://dadosabertos.inpi.gov.br/"
        status_code = 500

    monkeypatch.setattr(ip.requests, "get", lambda *a, **k: Resp())
    r = ip.validar_endpoint()
    assert r["status_ok"] is False
    assert r["alcancavel"] is False
    assert r["https_ok"] is True  # mas status falhou


def test_endpoint_checa_host_na_url_resolvida_apos_redirect(monkeypatch):
    # redirect final não é mais o host esperado
    class Resp:
        url = "https://outro.host.gov.br/x"
        status_code = 200

    monkeypatch.setattr(ip.requests, "get", lambda *a, **k: Resp())
    r = ip.validar_endpoint()
    assert r["host_ok"] is False
    assert r["alcancavel"] is False


def test_endpoint_schema_https_na_url_resolvida(monkeypatch):
    class Resp:
        url = "http://dadosabertos.inpi.gov.br/redir"
        status_code = 200

    monkeypatch.setattr(ip.requests, "get", lambda *a, **k: Resp())
    r = ip.validar_endpoint()
    assert r["https_ok"] is False
    assert r["alcancavel"] is False


# =========================================================================== #
# Loopback 3 — B6: contrato estável da seção data (mesmas chaves em todo path)
# =========================================================================== #
CHAVES_DATA = {
    "campo", "parseavel", "erro", "disponibilidade",
    "proporcao_parseaveis", "limiar_proporcao_parseaveis",
    "n_linhas_amostradas", "n_com_data", "n_parseaveis", "n_na_janela",
    "n_nao_plausiveis", "pct_na_janela", "cobertura_por_ano",
    "min_bruto", "max_bruto", "min_janela", "max_janela",
    "janela_2019_2026", "representatividade_ok", "qualidade", "bytes_membro",
}


def test_data_contrato_b6_identico_sucesso_e_erro():
    successo = ip.parsear_e_validar_data(_zip_janela_boa(), MEMBRO, ",")
    assert set(successo.keys()) == CHAVES_DATA
    assert successo["erro"] is None

    erro_membro = ip.parsear_e_validar_data(_zip_janela_boa(), "NAO_EXISTE.csv", ",")
    assert set(erro_membro.keys()) == CHAVES_DATA
    assert erro_membro["erro"] == "membro não encontrado"

    # default também espelha o contrato completo
    assert set(ip._data_default().keys()) == CHAVES_DATA
    assert ip._data_default()["erro"] is None
    for chave in CHAVES_DATA:
        assert chave in erro_membro


def test_data_contrato_b6_coluna_ausente_no_parsing():
    # membro existe, mas o cabeçalho NÃO tem data_deposito => branch
    # "coluna ausente no parsing" deve expor o MESMO contrato B6
    header = ",".join(c for c in COLS if c != "data_deposito")
    vals = ",".join(["x"] * (len(COLS) - 1))
    zf = fazer_zip(header, [vals])
    data = ip.parsear_e_validar_data(zf, MEMBRO, ",")
    assert set(data.keys()) == CHAVES_DATA
    assert "coluna ausente no parsing" in data["erro"]


def test_main_path_erro_data_contrato_b6(monkeypatch, tmp_path, capsys):
    zf_bytes = _zip_janela_boa_mb()
    arq_zip = tmp_path / "dump.zip"
    arq_zip.write_bytes(zf_bytes)

    def fake_endpoint():
        return {"https_ok": True, "host_ok": True, "status_ok": True, "alcancavel": True}

    def fake_garantir_dump(dump, url, dir_raw, progresso=True):
        return {"dump": dump, "baixado": True, "era_local": True, "integro": True,
                "bytes": len(zf_bytes), "caminho": str(arq_zip), "url": url}

    def boom(zf, membro, delimitador, encoding="utf-8", amostrar=None):
        raise RuntimeError("parsing quebrou")

    monkeypatch.setattr(ip, "validar_endpoint", fake_endpoint)
    monkeypatch.setattr(ip, "garantir_dump", fake_garantir_dump)
    monkeypatch.setattr(ip, "parsear_e_validar_data", boom)
    monkeypatch.setattr(ip, "DIR_RAW", str(tmp_path))
    monkeypatch.setattr(ip, "REPORT_PATH", str(tmp_path / "inpi_validacao.json"))
    monkeypatch.setattr(ip, "psutil", _FakePsutil())

    codigo = ip.main(["--dump", "dump.zip", "--membro", MEMBRO, "--no-progresso"])
    assert codigo == 1
    rel = json.loads((tmp_path / "inpi_validacao.json").read_text(encoding="utf-8"))
    assert set(rel["data"].keys()) == CHAVES_DATA
    assert "parsing quebrou" in rel["data"]["erro"]


# =========================================================================== #
# Loopback 3 — B7: amostragem RSS intra-chunk observada por teste
# =========================================================================== #
def test_parsear_e_validar_data_chama_amostrar_durante_parse(monkeypatch):
    monkeypatch.setattr(ip, "CHUNKSIZE", 40)
    contagem = {"n": 0}

    def contar():
        contagem["n"] += 1

    data = ip.parsear_e_validar_data(_zip_janela_boa(), MEMBRO, ",", amostrar=contar)
    n_linhas = data["n_linhas_amostradas"]
    n_chunks = n_linhas // 40 + (1 if n_linhas % 40 else 0)
    assert data["representatividade_ok"] is True
    # a amostragem roda DURANTE o parsing: chamadas por chunk (intra) além do
    # único callback entre chunks — um fix revertido falharia este assert
    assert contagem["n"] > n_chunks
    assert contagem["n"] >= n_chunks * 2


class _FakePsutilOscilante:
    """Memória baixa normalmente; espiga ACIMA do limite apenas em chamadas de
    amostragem que só existem na fix B5 (intra-chunk). O pico ocorre dentro de
    um chunk e desaparece: amostrar só entre chunks jamais o observa."""
    def __init__(self, acima_de=12, espigao=3):
        self._n = -1
        self._acima_de = acima_de
        self._espigao = espigao

    class _Proc:
        def __init__(self, mem):
            self._mem = mem
        def memory_info(self):
            return type("MI", (), {"rss": self._mem})()

    def Process(self):
        self._n += 1
        janela = self._acima_de <= self._n < self._acima_de + self._espigao
        rss = 3_000_000_000 if janela else 1_000_000
        return self._Proc(rss)


def test_main_oscilante_pico_intra_chunk_detectado(monkeypatch, tmp_path, capsys):
    # 241 linhas / CHUNKSIZE=40 => 7 chunks; o pico de RSS cai em chamadas 12..14,
    # todas intra-chunk (chunk 1 itera as chamadas 4..14). Com a fix revertida
    # (amostra só entre chunks) o máximo índice seria 11 e o pico passaria invisível.
    zf_bytes = _zip_janela_boa_mb()
    arq_zip = tmp_path / "dump.zip"
    arq_zip.write_bytes(zf_bytes)

    def fake_endpoint():
        return {"https_ok": True, "host_ok": True, "status_ok": True, "alcancavel": True}

    def fake_garantir_dump(dump, url, dir_raw, progresso=True):
        return {"dump": dump, "baixado": True, "era_local": True, "integro": True,
                "bytes": len(zf_bytes), "caminho": str(arq_zip), "url": url}

    monkeypatch.setattr(ip, "validar_endpoint", fake_endpoint)
    monkeypatch.setattr(ip, "garantir_dump", fake_garantir_dump)
    monkeypatch.setattr(ip, "CHUNKSIZE", 40)
    monkeypatch.setattr(ip, "DIR_RAW", str(tmp_path))
    monkeypatch.setattr(ip, "REPORT_PATH", str(tmp_path / "inpi_validacao.json"))
    monkeypatch.setattr(ip, "psutil", _FakePsutilOscilante())

    codigo = ip.main(["--dump", "dump.zip", "--membro", MEMBRO, "--no-progresso"])
    assert codigo == 1
    rel = json.loads((tmp_path / "inpi_validacao.json").read_text(encoding="utf-8"))
    assert rel["veredito"] == "pendente"
    assert rel["fr1"]["status"] == rel["veredito"]
    assert rel["fr1"]["memoria"] is False
    assert rel["memoria"]["dentro_do_limite"] is False
    assert rel["memoria"]["pico_memoria_mb"] >= 2048
    assert set(rel["data"].keys()) == CHAVES_DATA


# =========================================================================== #
# Loopback 3 — B8: drift reportado sem falhar (validar_schema)
# =========================================================================== #
def test_schema_b8_colunas_extras_nao_vazio():
    header = cabecalho_documentado() + ",sigilo,outra_extra"
    zf = fazer_zip(header, _linha("2019-01-01"))
    schema, _ = ip.validar_schema(zf, MEMBRO)
    assert schema["colunas_extras"] == ["sigilo", "outra_extra"]
    assert schema["colunas_extras"] == schema["colunas_extras_drift"]
    assert schema["schema_ok"] is True  # coluna extra não derruba o schema


def test_schema_b8_delimitador_detectado_reflete_o_real():
    # real "," (doc previa ";") => drift reportado, mas detectado corretamente
    zf = fazer_zip(cabecalho_documentado(), _linha("2019-01-01"))
    schema, _ = ip.validar_schema(zf, MEMBRO)
    assert schema["delimitador_detectado"] == ","
    assert schema["drift_delimitador"] is True

    # cabeçalho ";" real => sem drift
    header = ";".join(COLS)
    linhas = ["x" * len(COLS)]
    zf2 = fazer_zip(header, linhas)
    schema2, _ = ip.validar_schema(zf2, MEMBRO)
    assert schema2["delimitador_detectado"] == ";"
    assert schema2["drift_delimitador"] is False


def test_schema_b8_cabecalho_truncado_exposto():
    # truncado exposto no caminho de sucesso
    linha = b"a," * (200 * 1024)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MEMBRO, linha)
    buf.seek(0)
    schema, _ = ip.validar_schema(zipfile.ZipFile(buf), MEMBRO)
    assert "cabecalho_truncado" in schema
    assert schema["cabecalho_truncado"] is True

    # exposto também quando o membro não existe
    schema_ausente, _ = ip.validar_schema(_zip_janela_boa(), "NAO_EXISTE.csv")
    assert "cabecalho_truncado" in schema_ausente
    assert schema_ausente["cabecalho_truncado"] is False


def test_schema_b8_drift_encoding_por_validar_schema():
    # (a) byte cp1252 no cabeçalho => encoding detectado cp1252 + drift True
    header_cp = b"coluna1;coluna2\xe1"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MEMBRO, header_cp + b"\na;b\n")
    buf.seek(0)
    schema, _ = ip.validar_schema(zipfile.ZipFile(buf), MEMBRO)
    assert schema["encoding"] == ip.detectar_encoding(header_cp) == "cp1252"
    assert schema["drift_encoding"] is True

    # (b) cabeçalho utf-8 simples => sem drift de encoding
    schema_ok, _ = ip.validar_schema(
        fazer_zip(cabecalho_documentado(), _linha("2019-01-01")), MEMBRO
    )
    assert schema_ok["encoding"] == "utf-8"
    assert schema_ok["drift_encoding"] is False


def test_schema_b8_default_contrato():
    default = ip._schema_default()
    for chave in ("colunas_extras", "colunas_extras_drift", "delimitador_detectado",
                  "drift_delimitador", "encoding", "drift_encoding",
                  "cabecalho_truncado", "colunas_ausentes"):
        assert chave in default
    assert default["membro_encontrado"] is False
    assert default["schema_ok"] is False
    assert default["colunas_extras"] == default["colunas_extras_drift"] == []
    assert default["colunas_ausentes"] == list(ip.COLUNAS_DOCUMENTADAS)
