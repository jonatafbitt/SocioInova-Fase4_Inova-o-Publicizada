"""Suite pytest (Story 2.1) — helpers de configuração do ChromaDB + busca Top-K (2.2).

Cobre a criação/validação da coleção persistente em ``data/chroma_db`` (ou
qualquer ``tmp_path``), a chave estável de Chunk e os helpers de inscrição/dedup
usados pela vetorização, além da busca semântica Top-K da Story 2.2
(``buscar_top_k`` e ``buscar``). Roda offline: o carregamento do modelo
sentence-transformers é mockado (modelo fake) e o ChromaDB real é usado sobre
``tmp_path``.
"""
import numpy as np
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


# --------------------------------------------------------------------------- #
# Busca semântica Top-K (Story 2.2) — buscar_top_k / buscar
# --------------------------------------------------------------------------- #
# Consulta [1,0,0] sob a métrica L2 do ChromaDB (Euclidiana ao quadrado):
#   BR1::0 → distance 0.0 → score 1.0
#   BR1::1 → distance 1.0 → score 0.0
#   BR2::0 → distance 2.0 → score -1.0
# O score (1 - distance, L2) é de PROXIMIDADE, não similaridade de cosseno.
_MEIO_INCLINADO = 0.8660254037844386  # sqrt(3)/2
_METADADOS_BR1 = {"fonte": "BR1", "instituicao": "IF A", "tipo": "inpi",
                  "data": "2024-01-01"}


class _ModeloFake:
    """Fake de SentenceTransformer: vetoriza qualquer texto para um vetor fixo."""

    def __init__(self, vetor=(1.0, 0.0, 0.0)):
        self.vetor = np.asarray(vetor, dtype=np.float32)

    def encode(self, textos, normalize_embeddings=True):  # noqa: D401
        arr = np.zeros((len(textos), self.vetor.size), dtype=np.float32)
        arr[:] = self.vetor
        return arr


class _ModeloProibido:
    """Fake que falha se ``encode`` for chamado (consulta em branco não vetoriza)."""

    def encode(self, textos, normalize_embeddings=True):  # noqa: D401
        raise AssertionError("não deve vetorizar consulta em branco")


def _colecao_populada(tmp_path):
    colecao = retriever.obter_colecao(str(tmp_path / "db"))
    colecao.add(
        ids=["BR1::0", "BR1::1", "BR2::0"],
        embeddings=[[1.0, 0.0, 0.0],
                    [0.5, 0.0, _MEIO_INCLINADO],
                    [0.0, 0.0, 1.0]],
        documents=[
            "Energia solar para irrigação no assentamento.",
            "Curso de organização comunitária para produtores rurais.",
            "Compostagem de resíduos orgânicos urbanos na cidade.",
        ],
        metadatas=[
            {**_METADADOS_BR1, "chunk_idx": 0, "n_chunks": 2},
            {**_METADADOS_BR1, "chunk_idx": 1, "n_chunks": 2},
            {"fonte": "https://if.edu.br/nota/1", "instituicao": "IF B",
             "tipo": "noticia", "data": "2026-08-30",
             "chunk_idx": 0, "n_chunks": 1},
        ],
    )
    return colecao


# --- buscar_top_k ---------------------------------------------------------- #
def test_buscar_top_k_retorna_chunks_anotados_com_score(tmp_path):
    colecao = _colecao_populada(tmp_path)
    r = retriever.buscar_top_k(colecao, [1.0, 0.0, 0.0], k=2)
    assert len(r) == 2
    for item in r:
        assert {"chave", "texto", "fonte", "instituicao", "tipo", "data",
                "score"} <= set(item)
    assert r[0]["chave"] == "BR1::0"
    assert r[0]["fonte"] == "BR1"
    assert r[0]["instituicao"] == "IF A"
    assert r[0]["tipo"] == "inpi"
    assert r[0]["data"] == "2024-01-01"
    assert "Energia solar" in r[0]["texto"]
    assert r[0]["score"] == pytest.approx(1.0)
    assert r[0]["score"] >= r[1]["score"]


def test_buscar_top_k_ordenado_por_score_decrescente(tmp_path):
    colecao = _colecao_populada(tmp_path)
    r = retriever.buscar_top_k(colecao, [1.0, 0.0, 0.0], k=3)
    scores = [item["score"] for item in r]
    assert scores == sorted(scores, reverse=True)


def test_buscar_top_k_k_zero_ou_negativo_retorna_vazio(tmp_path):
    colecao = _colecao_populada(tmp_path)
    assert retriever.buscar_top_k(colecao, [1.0, 0.0, 0.0], k=0) == []
    assert retriever.buscar_top_k(colecao, [1.0, 0.0, 0.0], k=-1) == []


def test_buscar_top_k_colecao_vazia_retorna_vazio(tmp_path):
    colecao = retriever.obter_colecao(str(tmp_path / "db"))
    assert retriever.buscar_top_k(colecao, [1.0, 0.0, 0.0], k=5) == []


def test_buscar_top_k_sem_vetor_retorna_vazio(tmp_path):
    colecao = _colecao_populada(tmp_path)
    assert retriever.buscar_top_k(colecao, None, k=5) == []


def test_buscar_top_k_k_maior_que_n_retorna_todos(tmp_path):
    colecao = _colecao_populada(tmp_path)
    assert colecao.count() == 3
    r = retriever.buscar_top_k(colecao, [1.0, 0.0, 0.0], k=10)
    assert len(r) == 3


def test_buscar_top_k_nao_modifica_colecao(tmp_path):
    colecao = _colecao_populada(tmp_path)
    antes = colecao.count()
    retriever.buscar_top_k(colecao, [1.0, 0.0, 0.0], k=3)
    assert colecao.count() == antes
    dados = colecao.get(include=["documents", "metadatas"])
    assert len(dados["documents"]) == 3


# --- buscar ----------------------------------------------------------------- #
def test_buscar_consulta_com_match(tmp_path):
    colecao = _colecao_populada(tmp_path)
    r = retriever.buscar(colecao, "energia solar no campo", _ModeloFake(), k=5)
    assert len(r) == 3
    assert r[0]["fonte"] == "BR1"
    assert r[0]["score"] == pytest.approx(1.0)
    assert "Energia solar" in r[0]["texto"]


def test_buscar_k_zero_ou_negativo_retorna_vazio(tmp_path):
    colecao = _colecao_populada(tmp_path)
    modelo = _ModeloProibido()
    assert retriever.buscar(colecao, "qualquer coisa", modelo, k=0) == []
    assert retriever.buscar(colecao, "qualquer coisa", modelo, k=-2) == []


def test_buscar_consulta_vazia_nao_vetoriza(tmp_path):
    colecao = _colecao_populada(tmp_path)
    for consulta in ("", "   ", "\n\t"):
        assert retriever.buscar(colecao, consulta, _ModeloProibido()) == []


def test_buscar_colecao_vazia_retorna_vazio(tmp_path):
    colecao = retriever.obter_colecao(str(tmp_path / "db"))
    assert retriever.buscar(colecao, "energia solar", _ModeloFake()) == []


def test_buscar_min_score_filtra_chunks_abaixo_do_limiar(tmp_path):
    colecao = _colecao_populada(tmp_path)
    r = retriever.buscar(colecao, "energia solar no campo", _ModeloFake(),
                         k=5, min_score=0.5)
    assert len(r) == 1
    assert r[0]["chave"] == "BR1::0"
    assert r[0]["score"] == pytest.approx(1.0)


def test_buscar_sem_match_relevante_retorna_vazio(tmp_path):
    # min_score acima de todos os scores => nada passa no limiar => nunca fabrica.
    colecao = _colecao_populada(tmp_path)
    r = retriever.buscar(colecao, "energia solar no campo", _ModeloFake(),
                         k=5, min_score=1.5)
    assert r == []


# --- Defeitos de resposta / guards ------------------------------------------ #
class _ColecaoComFalha:
    """Coleção fake cuja query sempre falha (simula erro real do ChromaDB)."""

    def count(self):
        return 3

    def query(self, *args, **kwargs):
        raise RuntimeError("falha simulada na query do ChromaDB")


class _ColecaoMisturada:
    """Coleção fake que devolve listas de comprimentos divergentes no query."""

    def count(self):
        return 2

    def query(self, *args, **kwargs):
        return {
            "ids": [["a"]],
            "documents": [["doc"]],
            "metadatas": [[None, None]],
            "distances": [[0.1]],
        }


def test_buscar_top_k_falha_na_query_levanta_runtime_error():
    with pytest.raises(RuntimeError, match="falha na busca Top-K"):
        retriever.buscar_top_k(_ColecaoComFalha(), [1.0, 0.0, 0.0], k=5)


def test_buscar_propaga_runtime_error_nao_retorna_vazio():
    # falha na busca NÃO pode ser silenciada como "sem match" => [].
    with pytest.raises(RuntimeError, match="falha na busca Top-K"):
        retriever.buscar(_ColecaoComFalha(), "energia solar", _ModeloFake(), k=5)


def test_buscar_top_k_listas_divergentes_levanta_runtime_error():
    with pytest.raises(RuntimeError, match="comprimentos divergentes"):
        retriever.buscar_top_k(_ColecaoMisturada(), [1.0, 0.0, 0.0], k=2)


# --- Limites extras de buscar (P6) ------------------------------------------- #
def test_buscar_k_1_limita_retorno_a_1(tmp_path):
    colecao = _colecao_populada(tmp_path)
    r = retriever.buscar(colecao, "energia solar no campo", _ModeloFake(), k=1)
    assert len(r) == 1


def test_buscar_k_maior_que_n_retorna_todos(tmp_path):
    colecao = _colecao_populada(tmp_path)
    r = retriever.buscar(colecao, "energia solar no campo", _ModeloFake(), k=10)
    assert len(r) == 3


def test_buscar_min_score_negativo_retorna_chunks(tmp_path):
    # limiar abaixo de todos os scores => nada é cortado.
    colecao = _colecao_populada(tmp_path)
    r = retriever.buscar(colecao, "energia solar no campo", _ModeloFake(),
                         k=5, min_score=-2.0)
    assert len(r) == 3


def test_buscar_min_score_inclusivo_e_truncado_para_k(tmp_path):
    # min_score baixo + janela k*2: há 3 Chunks acima do limiar, mas k=1
    # trunca o retorno de volta a 1 (hard limit preservado).
    colecao = _colecao_populada(tmp_path)
    r = retriever.buscar(colecao, "energia solar no campo", _ModeloFake(),
                         k=1, min_score=-10.0)
    assert len(r) == 1
    assert r[0]["chave"] == "BR1::0"


def test_buscar_min_score_limiar_exato_e_inclusivo(tmp_path):
    # min_score == score exato de um Chunk (limiar inclusive `>=`): ele entra.
    colecao = _colecao_populada(tmp_path)
    base = retriever.buscar_top_k(colecao, [1.0, 0.0, 0.0], k=3)
    limiar = base[1]["score"]  # score exato do 2º Chunk (BR1::1)
    r = retriever.buscar(colecao, "energia solar no campo", _ModeloFake(),
                         k=3, min_score=limiar)
    chaves = [item["chave"] for item in r]
    assert "BR1::1" in chaves
    assert any(item["score"] == limiar for item in r)


def test_buscar_consulta_nao_string_retorna_vazio(tmp_path):
    colecao = _colecao_populada(tmp_path)
    assert retriever.buscar(colecao, 123, _ModeloProibido()) == []
    assert retriever.buscar(colecao, ["energia"], _ModeloProibido()) == []


def test_buscar_k_float_retorna_vazio(tmp_path):
    colecao = _colecao_populada(tmp_path)
    assert retriever.buscar(colecao, "energia solar", _ModeloProibido(),
                            k=2.0) == []


def test_buscar_min_score_nao_numerico_retorna_vazio(tmp_path):
    colecao = _colecao_populada(tmp_path)
    assert retriever.buscar(colecao, "energia solar", _ModeloProibido(),
                            min_score="alto") == []
