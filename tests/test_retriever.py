"""Suite pytest (Story 2.1) — helpers de configuração do ChromaDB.

Cobre a criação/validação da coleção persistente em ``data/chroma_db`` (ou
qualquer ``tmp_path``), a chave estável de Chunk e os helpers de inscrição/dedup
usados pela vetorização. Roda offline; usa o ChromaDB real sobre ``tmp_path``.
"""
import pytest

from rag_engine import retriever


# --------------------------------------------------------------------------- #
# Chave estável de Chunk
# --------------------------------------------------------------------------- #
def test_chave_chunk():
    assert retriever.chave_chunk("BR100", 0) == "BR100::0"
    assert retriever.chave_chunk("BR100", 3) == "BR100::3"
    assert retriever.chave_chunk("https://if.edu.br/a", 1) == "https://if.edu.br/a::1"


# --------------------------------------------------------------------------- #
# Criação/validação da coleção persistente
# --------------------------------------------------------------------------- #
def test_obter_colecao_cria_e_persiste(tmp_path):
    colecao = retriever.obter_colecao(str(tmp_path / "db"))
    assert colecao.count() == 0
    colecao.add(ids=["a0"], embeddings=[[0.1] * 8], documents=["doc"],
                metadatas=[{"fonte": "BR1", "tipo": "inpi"}])
    assert colecao.count() == 1


def test_obter_colecao_mesma_da_mesma(tmp_path):
    db = str(tmp_path / "db")
    c1 = retriever.obter_colecao(db)
    c2 = retriever.obter_colecao(db)
    assert c1.name == c2.name == retriever.COLECAO


# --------------------------------------------------------------------------- #
# Dedup (ids existentes)
# --------------------------------------------------------------------------- #
def test_ids_existentes(tmp_path):
    colecao = retriever.obter_colecao(str(tmp_path / "db"))
    colecao.add(ids=["a0", "a1"], embeddings=[[0.1] * 8, [0.2] * 8],
                documents=["t0", "t1"])
    existentes = retriever.ids_existentes(colecao, ["a0", "a1", "zz", "a2"])
    assert existentes == {"a0", "a1"}


def test_ids_existentes_vazio(tmp_path):
    colecao = retriever.obter_colecao(str(tmp_path / "db"))
    assert retriever.ids_existentes(colecao, []) == set()


def test_ids_existentes_atravessa_lote_de_500(tmp_path):
    # Exercita o loop de lotes (500) de ids_existentes de forma leve, com uma
    # coleção fake que apenas acumula as chamadas de `get`. O lote cruza o
    # limite de 500 (1000 consultados => 2 updates), verificando o branch de
    # múltiplos lotes sem o custo de 1000 ids reais no ChromaDB.
    class _ColecaoFake:
        def __init__(self):
            self.chamadas: list[list[str]] = []
            self.existentes = {"real0", "real1"}

        def get(self, ids, include=None):
            self.chamadas.append(list(ids))
            presentes = [i for i in ids if i in self.existentes]
            return {"ids": presentes}

    fake = _ColecaoFake()
    consulta = ["real0", "real1"] + [f"inexistente-{i}" for i in range(1000)]
    existentes = retriever.ids_existentes(fake, consulta)
    assert existentes == {"real0", "real1"}
    # o loop partiu os 1002 ids em lotes de 500 => 3 chamadas (`900/1001` cruzou)
    assert len(fake.chamadas) >= 3
    assert all(len(c) <= 500 for c in fake.chamadas)


# --------------------------------------------------------------------------- #
# Inserção
# --------------------------------------------------------------------------- #
def test_inserir_chunks_retorna_contagem(tmp_path):
    colecao = retriever.obter_colecao(str(tmp_path / "db"))
    n = retriever.inserir_chunks(
        colecao,
        ids=["a0", "a1"],
        embeddings=[[0.1] * 8, [0.2] * 8],
        metadatas=[{"fonte": "BR1", "tipo": "inpi", "instituicao": "IF",
                    "data": "2023-01-01", "chunk_idx": 0},
                   {"fonte": "BR1", "tipo": "inpi", "instituicao": "IF",
                    "data": "2023-01-01", "chunk_idx": 1}],
        documentos=["parágrafo um", "parágrafo dois"],
    )
    assert n == 2
    assert colecao.count() == 2


def test_inserir_chunks_vazio_nao_faz_nada(tmp_path):
    colecao = retriever.obter_colecao(str(tmp_path / "db"))
    n = retriever.inserir_chunks(colecao, ids=[], embeddings=[], metadatas=[], documentos=[])
    assert n == 0
    assert colecao.count() == 0


# --------------------------------------------------------------------------- #
# Persistência sobrevive a reinício (nova cliente, mesmo path)
# --------------------------------------------------------------------------- #
def test_persistencia_sobrevive_a_reinicio(tmp_path):
    db = str(tmp_path / "db")
    c1 = retriever.obter_colecao(db)
    c1.add(ids=["x0"], embeddings=[[0.5] * 8], documents=["doc persistente"],
           metadatas=[{"fonte": "BR9"}])
    # nova cliente sobre o mesmo diretório recupera o estado
    c2 = retriever.obter_colecao(db)
    assert c2.count() == 1
    dados = c2.get(ids=["x0"], include=["documents"])
    assert dados["documents"][0] == "doc persistente"


# --------------------------------------------------------------------------- #
# Checagem de disponibilidade do chromadb
# --------------------------------------------------------------------------- #
def test_checar_disponibilidade():
    assert retriever.checar_disponibilidade() is True
