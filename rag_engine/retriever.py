"""Configuração mínima do ChromaDB para o Epic 2 (Story 2.1).

Este módulo concentra a configuração do banco vetorial local (caminho de
persistência ``data/chroma_db``), a criação/validação da coleção e os helpers
de inserção e dedup usados pela vetorização (``embeddings.py``). A busca
Top-K semântica é Story 2.2 e não faz parte deste módulo nesta etapa.

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

__version__ = "2.1.0"

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
