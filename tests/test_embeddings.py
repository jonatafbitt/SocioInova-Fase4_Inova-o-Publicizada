"""Suite pytest (Story 2.1) — vetorização do corpus e armazenamento no ChromaDB.

Cobre os critérios de aceitação: cada documento da Janela com texto vira 1+
Chunks por parágrafos completos e é inserido com texto/vetor/metadados
(fonte/instituicao/tipo/data); re-execução idempotente (sem duplicatas);
itens fora da Janela ou com texto vazio não são indexados, sem erro fatal;
persistência sobrevive a reinício; e CLI com exit 0 e relatório coerente.

A suite roda offline: o carregamento do modelo sentence-transformers é mockado
(``monkeypatch`` de ``embeddings.carregar_modelo``), de modo que nenhum modelo é
baixado ou carregado da rede. O ChromaDB real é usado sobre ``tmp_path``.
"""
import csv
import json

import numpy as np
import pytest

from rag_engine import embeddings, retriever

DIM_VETOR = 8


class _ModeloFake:
    """Fake de SentenceTransformer: retorna vetores determinísticos, sem rede."""

    def __init__(self, nome="fake-modelo"):
        self.nome = nome

    def encode(self, textos, normalize_embeddings=True):  # noqa: D401
        n = len(textos)
        # vetor determinístico por índice (depende da posição, com dimensão fixa)
        arr = np.zeros((n, DIM_VETOR), dtype=np.float32)
        for i in range(n):
            arr[i] = [float(i + 1)] * DIM_VETOR
        return arr


def _escrever_corpus(tmp_path, linhas):
    """Escreve um CSV sintético de corpus e devolve o caminho absoluto."""
    p = tmp_path / "corpus.csv"
    cols = ["fonte", "tipo", "data", "instituicao", "texto_límpido",
            "na_janela", "data_ausente", "texto_fallback_titulo", "instituicoes_aux"]
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for linha in linhas:
            base = {c: "" for c in cols}
            base.update(linha)
            writer.writerow(base)
    return str(p)


@pytest.fixture()
def patched_modelo(monkeypatch):
    """Torna o carregamento do modelo um fake (offline)."""
    fake = _ModeloFake()
    monkeypatch.setattr(embeddings, "carregar_modelo", lambda *a, **k: fake)
    return fake


def _query_tudo(colecao):
    resp = colecao.get(include=["documents", "metadatas", "embeddings"])
    return resp


# --------------------------------------------------------------------------- #
# Chunking: parágrafos completos preservados, sem cortar frase
# --------------------------------------------------------------------------- #
def test_chunk_paragrafos_preserva_paragrafos_completos():
    texto = ("Primeiro parágrafo com uma frase inteira. "
             "Outra frase ainda no primeiro bloco.\n\n"
             "Segundo parágrafo completo separado por linha em branco.\n\n"
             "Terceiro parágrafo curto.")
    chunks = embeddings.chunk_paragrafos(texto)
    # três parágrafos curtos agrupados tendem a um único Chunk
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "Primeiro parágrafo" in chunk
        assert "Segundo parágrafo" in chunk
        assert "Terceiro parágrafo" in chunk


def test_chunk_paragrafos_sem_quebra_nao_corta_frase():
    # sem linha em branco => um único parágrafo completo (nunca cortado)
    texto = ("Uma frase longa que continua e não deve ser partida. "
             "Outra frase que segue pertencendo ao mesmo bloco.")
    chunks = embeddings.chunk_paragrafos(texto, alvo_tokens=1)  # alvo mínimo
    assert chunks == [texto]
    assert "não deve ser partida" in chunks[0]


def test_chunk_paragrafos_vazio():
    assert embeddings.chunk_paragrafos("") == []
    assert embeddings.chunk_paragrafos("   \n\n   ") == []


def test_chunk_paragrafos_agrupa_em_multiplos_chunks_sem_cortar_paragrafo():
    # 5 parágrafos que, juntos, excedem o alvo de caracteres => o agrupamento
    # fecha chunks e abre novos (branch de splitting em chunk_paragrafos),
    # sempre preservando parágrafos completos (nunca cortando no meio).
    paragrafos = ["P" * 400] * 5
    texto = "\n\n".join(paragrafos)
    chunks = embeddings.chunk_paragrafos(texto, alvo_tokens=50)  # alvo ~200 chars
    assert len(chunks) >= 2  # múltiplos chunks gerados
    # cada chunk é formado por parágrafos inteiros concatenados (sem corte interno)
    for chunk in chunks:
        assert chunk.strip() == chunk
        # 400 chars por parágrafo: um chunk nunca contém *metade* de um parágrafo
        assert all((c in ("P", " ")) for c in chunk)
    # todos os parágrafos são cobertos exatamente uma vez, na ordem
    reconstituido = " ".join(chunks)
    for p in paragrafos:
        assert p in reconstituido


def test_dividir_paragrafos_ignora_vazios():
    assert embeddings._dividir_paragrafos("Um.\n\nDois.\n\n\n  \nTrês.") == \
        ["Um.", "Dois.", "Três."]


# --------------------------------------------------------------------------- #
# AC: documento na Janela com texto vira Chunks com texto+vetor+metadados
# --------------------------------------------------------------------------- #
def test_ac_documento_janela_com_texto_na_colecao(tmp_path, patched_modelo):
    corpus = _escrever_corpus(tmp_path, [
        {"fonte": "BR2020", "tipo": "inpi", "data": "2020-05-01",
         "instituicao": "INSTITUTO FEDERAL DE X", "texto_límpido": "Titulo Resumo A",
         "na_janela": "True"},
    ])
    db = str(tmp_path / "db")
    rel = embeddings.vetorizar_corpus(corpus, db_path=db, modelo=_ModeloFake())
    assert rel["n_chunks_adicionados"] >= 1
    assert rel["n_documentos_indexados"] == 1

    colecao = retriever.obter_colecao(db)
    assert colecao.count() >= 1
    dados = _query_tudo(colecao)
    metadados = dados["metadatas"][0]
    assert metadados["fonte"] == "BR2020"
    assert metadados["instituicao"] == "INSTITUTO FEDERAL DE X"
    assert metadados["tipo"] == "inpi"
    assert metadados["data"] == "2020-05-01"
    assert "Titulo Resumo" in dados["documents"][0]
    assert len(dados["embeddings"][0]) == DIM_VETOR


def test_ac_noticia_na_janela_com_texto(tmp_path, patched_modelo):
    corpus = _escrever_corpus(tmp_path, [
        {"fonte": "https://ifal.edu.br/a", "tipo": "noticia", "data": "2026-08-31",
         "instituicao": "IFAL", "texto_límpido": "Corpo da notícia inteiro.",
         "na_janela": "True"},
    ])
    db = str(tmp_path / "db")
    rel = embeddings.vetorizar_corpus(corpus, db_path=db, modelo=_ModeloFake())
    assert rel["n_chunks_adicionados"] >= 1
    colecao = retriever.obter_colecao(db)
    dados = _query_tudo(colecao)
    assert dados["metadatas"][0]["fonte"] == "https://ifal.edu.br/a"
    assert dados["metadatas"][0]["tipo"] == "noticia"


# --------------------------------------------------------------------------- #
# AC: idempotência (re-execução sem duplicatas)
# --------------------------------------------------------------------------- #
def test_ac_idempotencia_reexecucao(tmp_path, patched_modelo):
    corpus = _escrever_corpus(tmp_path, [
        {"fonte": "BR2021", "tipo": "inpi", "data": "2021-01-01",
         "instituicao": "IF Y", "texto_límpido": "Um e dois e três paragrafos.",
         "na_janela": "True"},
    ])
    db = str(tmp_path / "db")
    rel1 = embeddings.vetorizar_corpus(corpus, db_path=db, modelo=_ModeloFake())
    n1 = rel1["n_chunks_adicionados"]

    rel2 = embeddings.vetorizar_corpus(corpus, db_path=db, modelo=_ModeloFake())
    assert rel2["n_chunks_adicionados"] == 0
    assert rel2["n_chunks_ja_existentes"] == n1

    colecao = retriever.obter_colecao(db)
    assert colecao.count() == n1  # sem duplicatas


# --------------------------------------------------------------------------- #
# AC: fora da Janela e texto vazio não indexados, sem abortar
# --------------------------------------------------------------------------- #
def test_ac_fora_janela_e_texto_vazio_nao_indexados(tmp_path, patched_modelo):
    corpus = _escrever_corpus(tmp_path, [
        {"fonte": "BR2015", "tipo": "inpi", "data": "2015-01-01",
         "instituicao": "IF A", "texto_límpido": "Fora da janela.",
         "na_janela": "False"},
        {"fonte": "BR2022", "tipo": "inpi", "data": "2022-01-01",
         "instituicao": "IF B", "texto_límpido": "",
         "na_janela": "True"},
        {"fonte": "BR2023", "tipo": "inpi", "data": "2023-01-01",
         "instituicao": "IF C", "texto_límpido": "Dentro, com texto.",
         "na_janela": "True"},
    ])
    db = str(tmp_path / "db")
    rel = embeddings.vetorizar_corpus(corpus, db_path=db, modelo=_ModeloFake())
    assert rel["n_fora_janela"] == 1
    assert rel["n_texto_vazio"] == 1
    assert rel["n_candidatos_janela_com_texto"] == 1
    assert rel["n_chunks_adicionados"] >= 1

    colecao = retriever.obter_colecao(db)
    dados = _query_tudo(colecao)
    fontes = {m["fonte"] for m in dados["metadatas"]}
    assert {"BR2015", "BR2022"} & fontes == set()
    assert "BR2023" in fontes


# --------------------------------------------------------------------------- #
# AC: persistência sobrevive a reinício (nova cliente, mesmo path)
# --------------------------------------------------------------------------- #
def test_ac_persistencia_apos_reinicio(tmp_path, patched_modelo):
    corpus = _escrever_corpus(tmp_path, [
        {"fonte": "BR2024", "tipo": "inpi", "data": "2024-01-01",
         "instituicao": "IF D", "texto_límpido": "Texto persistente gravado.",
         "na_janela": "True"},
    ])
    db = str(tmp_path / "db")
    embeddings.vetorizar_corpus(corpus, db_path=db, modelo=_ModeloFake())

    # "reinício": nova cliente sobre o mesmo diretório recupera os Chunks
    nova_colecao = retriever.obter_colecao(db)
    assert nova_colecao.count() == 1
    dados = _query_tudo(nova_colecao)
    assert dados["metadatas"][0]["fonte"] == "BR2024"


# --------------------------------------------------------------------------- #
# CLI: exit 0 e relatório com contagens (indexados vs. excluídos)
# --------------------------------------------------------------------------- #
def test_cli_exit_0_e_relatorio(tmp_path, patched_modelo):
    corpus = _escrever_corpus(tmp_path, [
        {"fonte": "BR2025", "tipo": "inpi", "data": "2025-01-01",
         "instituicao": "IF E", "texto_límpido": "Com texto, na janela.",
         "na_janela": "True"},
        {"fonte": "BR2010", "tipo": "inpi", "data": "2010-01-01",
         "instituicao": "IF F", "texto_límpido": "Fora da janela.",
         "na_janela": "False"},
        {"fonte": "BR2026", "tipo": "inpi", "data": "2026-01-01",
         "instituicao": "IF G", "texto_límpido": "", "na_janela": "True"},
    ])
    db = str(tmp_path / "db")
    rel_path = str(tmp_path / "relatorio.json")

    codigo = embeddings.main(["--corpus", corpus, "--db", db,
                              "--relatorio", rel_path])
    assert codigo == 0

    rel = json.load(open(rel_path, encoding="utf-8"))
    assert rel["n_candidatos_janela_com_texto"] == 1
    assert rel["n_fora_janela"] == 1
    assert rel["n_texto_vazio"] == 1
    assert rel["n_chunks_adicionados"] >= 1
    assert rel["erros"] == []


def test_main_exit_1_com_erro(tmp_path, patched_modelo):
    db = str(tmp_path / "db")
    rel_path = str(tmp_path / "relatorio.json")
    codigo = embeddings.main(["--corpus", str(tmp_path / "nao-existe.csv"),
                              "--db", db, "--relatorio", rel_path])
    assert codigo == 1
    rel = json.load(open(rel_path, encoding="utf-8"))
    assert any("corpus_csv não encontrado" in e for e in rel["erros"])
