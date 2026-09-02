"""Configuração mínima do ChromaDB para o Epic 2 (Story 2.1) + busca Top-K (2.2).

Este módulo concentra a configuração do banco vetorial local (caminho de
persistência ``data/chroma_db``), a criação/validação da coleção e os helpers
de inserção e dedup usados pela vetorização (``embeddings.py``). A partir da
Story 2.2, também expõe a busca semântica Top-K read-only sobre a coleção
(``buscar_top_k`` e ``buscar``).

Processamento 100% local/offline (AD-1): os vetores e textos ficam em disco,
nunca saem da máquina.
"""
from __future__ import annotations

import os

try:
    from chromadb import PersistentClient
    from chromadb.config import Settings
except Exception:  # noqa: BLE001 - chromadb opcional (fallback no embeddings)
    PersistentClient = None  # type: ignore[assignment,misc]
    Settings = None  # type: ignore[assignment,misc]

__version__ = "2.2.0"

# Caminho padrão de persistência do ChromaDB, relativo à raiz do projeto.
PROJETO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJETO_RAIZ, "data", "chroma_db")

# Nome da coleção persistente do corpus RFEPCT vetorizado.
COLECAO = "corpus_rfepct"

# Separador usado na chave estável de um Chunk: ``<fonte>::<índice>``.
CHAVE_SEP = "::"


def checar_disponibilidade() -> bool:
    """True quando o pacote chromadb está importável no ambiente atual."""
    return PersistentClient is not None


def obter_cliente(db_path: str = DB_PATH):
    """Retorna o PersistentClient do ChromaDB apontando para ``db_path``.

    ``db_path`` é criado se não existir. Com ``PersistentClient`` não
    disponível (ambiente sem chromadb), levanta ``ImportError`` com mensagem
    clara.
    """
    if PersistentClient is None:
        raise ImportError(
            "chromadb não está instalado/importável. Instale com "
            "`pip install chromadb` para persistir a base vetorial."
        )
    os.makedirs(db_path, exist_ok=True)
    return PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))


def obter_colecao(db_path: str = DB_PATH, nome: str = COLECAO):
    """Obtém (criando se preciso) a coleção persistente ``nome`` em ``db_path``."""
    cliente = obter_cliente(db_path)
    colecao = cliente.get_or_create_collection(name=nome)
    return colecao


def chave_chunk(fonte: str, idx: int) -> str:
    """Chave estável e única de um Chunk: ``fonte`` + índice do Chunk."""
    return f"{fonte}{CHAVE_SEP}{idx}"


def ids_existentes(colecao, ids: list[str]) -> set[str]:
    """Retorna o subconjunto de ``ids`` já presente na coleção (dedup).

    Consulta em lote os ids informados e devolve os que já existem, para que a
    re-execução não reinsira Chunks (idempotência). Uma falha na consulta
    **nunca é silenciada**: ela é re-lançada com mensagem clara, porque fingir
    que nada existe violaria a garantia de idempotência (reinserindo Chunks
    duplicados de forma invisível).
    """
    if not ids:
        return set()
    existentes: set[str] = set()
    # busca em lotes para evitar payloads excessivos na consulta
    lote = 500
    for i in range(0, len(ids), lote):
        fatia = ids[i : i + lote]
        try:
            resp = colecao.get(ids=fatia)
            existentes.update(resp.get("ids") or [])
        except Exception as exc:  # noqa: BLE001 - falha real não deve ser mascarada
            raise RuntimeError(
                f"falha na consulta de dedup (chunks {i}..{i + len(fatia)}): {exc}"
            ) from exc
    return existentes


def inserir_chunks(colecao, ids: list[str], embeddings: list,
                   metadatas: list[dict], documentos: list[str]) -> int:
    """Insere Chunks na coleção e retorna quantos foram efetivamente adicionados.

    Espera já aplicada a dedup (os ``ids`` são tidos como ainda inexistentes na
    coleção). ``add`` é chamado apenas se houver ids para inserir. Uma falha na
    gravação é propagada com mensagem clara (não deixamos um lote parcialmente
    gravado passar como sucesso).
    """
    if not ids:
        return 0
    try:
        colecao.add(ids=ids, embeddings=embeddings, metadatas=metadatas,
                    documents=documentos)
    except Exception as exc:  # noqa: BLE001 - erro de gravação deve ser visível
        raise RuntimeError(f"falha ao gravar {len(ids)} Chunks na coleção: {exc}") from exc
    return len(ids)


# --------------------------------------------------------------------------- #
# Busca semântica Top-K (Story 2.2) — read-only, nunca fabrica resultado
# --------------------------------------------------------------------------- #
def buscar_top_k(colecao, consulta_vetor, k):
    """Retorna os ``k`` Chunks mais próximos de ``consulta_vetor`` em ``colecao``.

    Consulta **bruta read-only**: nada é gravado na coleção. Cada Chunk
    devolvido traz ``texto``, a chave do Chunk, os metadados da Fonte
    (``fonte``/``instituicao``/``tipo``/``data``) e o ``score`` de proximidade.

    Escala do ``score``: é ``1 - distance`` calculado sobre a **métrica L2**
    (Euclidiana ao quadrado, default do ChromaDB). É um **score de
    proximidade** — maior = mais próximo — e **não** uma similaridade de
    cosseno normalizada; pode ser negativo. A troca de métrica/normalização
    fica para o benchmark OQ1 e não é feita aqui. Os resultados vêm ordenados
    por ``score`` decrescente.

    Com ``k`` não-int ou ``k <= 0``, ``consulta_vetor`` ausente/vazio ou
    coleção vazia, devolve ``[]`` **sem exceção** (nunca fabrica). Para ``k``
    maior que o total de Chunks, devolve todos os disponíveis
    (``n_results=min(k, count)``).
    """
    if not isinstance(k, int) or k <= 0:
        return []
    if not consulta_vetor:
        return []
    try:
        n_chunks = colecao.count()
        if n_chunks == 0:
            return []
        n_resultados = min(k, n_chunks)
        resp = colecao.query(
            query_embeddings=[consulta_vetor],
            n_results=n_resultados,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # noqa: BLE001 - falha real não deve ser mascarada
        raise RuntimeError(f"falha na busca Top-K (k={k}): {exc}") from exc

    chaves = (resp.get("ids") or [[]])[0]
    documentos = (resp.get("documents") or [[]])[0]
    metadatas = (resp.get("metadatas") or [[]])[0]
    distancias = (resp.get("distances") or [[]])[0]
    if not (len(chaves) == len(documentos) == len(metadatas) == len(distancias)):
        raise RuntimeError(
            "resposta da busca Top-K com listas de comprimentos divergentes "
            f"(ids={len(chaves)}, documentos={len(documentos)}, "
            f"metadatas={len(metadatas)}, distancias={len(distancias)})"
        )

    resultados = []
    for chave, documento, metadados, distancia in zip(
        chaves, documentos, metadatas, distancias
    ):
        metadados = metadados or {}
        resultados.append({
            "chave": chave,
            "texto": documento,
            "fonte": metadados.get("fonte", ""),
            "instituicao": metadados.get("instituicao", ""),
            "tipo": metadados.get("tipo", ""),
            "data": metadados.get("data", ""),
            "score": 1.0 - float(distancia),
        })
    resultados.sort(key=lambda r: r["score"], reverse=True)
    return resultados


def buscar(colecao, consulta, modelo, k=5, min_score=None):
    """Vetoriza ``consulta`` localmente e devolve os Top-K Chunks relevantes.

    Orquestra a busca semântica (FR-5): valida a entrada antes de qualquer
    trabalho — ``consulta`` não-str ou em branco, ``k`` não-int ou ``k <= 0``
    e ``min_score`` informado mas não-numérico devolvem ``[]`` **sem exceção**
    e sem vetorizar desnecessariamente. Vetoriza a consulta via
    ``embeddings.gerar_embeddings`` do modelo local e delega para
    ``buscar_top_k``.

    ``min_score``: quando informado, o corte é aplicado **antes da janela
    Top-K, não após truncar**: busca-se uma janela mais larga (``k * 2``) para
    não perder Chunks relevantes que ficariam fora do Top-K,
    mantêm-se os Chunks com ``score >= min_score`` (limiar **inclusivo**) e só
    então o resultado é truncado para no máximo ``k``. Quando ``min_score`` é
    ``None`` (default), devolve os ``k`` mais próximos sem corte.

    Nunca fabrica resultado fora da coleção e jamais modifica a coleção
    (read-only).
    """
    if not isinstance(consulta, str) or not consulta.strip():
        return []
    if not isinstance(k, int) or k <= 0:
        return []
    if min_score is not None and not isinstance(min_score, (int, float)):
        return []
    # import local para evitar ciclo no módulo (embeddings importa retriever)
    from rag_engine import embeddings

    vetores = embeddings.gerar_embeddings(modelo, [consulta])
    if not vetores or vetores[0] is None:
        return []
    vetor = vetores[0]

    if min_score is None:
        return buscar_top_k(colecao, vetor, k)

    janela = buscar_top_k(colecao, vetor, k * 2)
    resultados = [r for r in janela if r["score"] >= min_score]
    return resultados[:k]
