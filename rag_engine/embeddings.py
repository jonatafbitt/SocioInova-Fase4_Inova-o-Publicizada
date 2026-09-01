"""Vetoriza o corpus RFEPCT e o armazena localmente no ChromaDB (Story 2.1).

Divide cada documento do corpus em Chunks preservando **parágrafos completos**
(nunca corta frase/parágrafo no meio), gera embeddings locais via
sentence-transformers (modelo multilingue) e grava texto + metadados da Fonte +
vetor na coleção ChromeDB persistida em ``data/chroma_db``.

Restrições seguidas (AD-1 / FR-4):
- Processamento 100% local/offline; a única exceção é o **download inicial do
  modelo** (decisão humana documentada); execuções seguintes carregam do cache.
- Re-vetorização **idempotente**: re-execução não duplica Chunks (dedup por
  chave estável ``fonte::índice``).
- Somente itens com ``na_janela: true`` e ``texto_límpido`` não vazio são
  indexados; os demais são apenas contabilizados no relatório (sem erro fatal).

A CLI grava um relatório JSON com a contagem de indexados e excluídos.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

PROJETO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJETO_RAIZ not in sys.path:
    sys.path.insert(0, PROJETO_RAIZ)

from rag_engine import retriever  # noqa: E402

__version__ = "2.1.0"
CORPUS_DEFAULT = os.path.join(PROJETO_RAIZ, "data", "processed", "corpus_rfepct.csv")
DB_DEFAULT = retriever.DB_PATH
RELATORIO_DEFAULT = os.path.join(PROJETO_RAIZ, "data", "processed",
                                 "relatorio_vetorizacao.json")

# Modelo multilingue default (adotado na Story 2.1; validação em benchmark-piloto
# OQ1 é story dedicada e fica fora do escopo desta entrega).
MODELO_DEFAULT = "paraphrase-multilingual-MiniLM-L12-v2"

# Alvo de tamanho por Chunk (em tokens, aproximado). Como nunca cortamos um
# parágrafo, o tamanho real pode variar acima deste alvo para parágrafos longos.
ALVO_TOKENS = 400
# Conversão tokens -> caracteres (~4 chars/token em texto pt-BR médio).
CHARS_POR_TOKEN = 4

# Regex de separador de parágrafo: uma ou mais linhas em branco.
_RE_PARAGRAFO = re.compile(r"\n\s*\n+")
_RE_BRANCO = re.compile(r"\s+")

# Colunas do corpus (contrato da Story 1.5, reutilizado aqui apenas para leitura).
COLUNA_TEXTO = "texto_límpido"
COLUNA_FONTE = "fonte"
COLUNA_TIPO = "tipo"
COLUNA_DATA = "data"
COLUNA_INSTITUICAO = "instituicao"
COLUNA_JANELA = "na_janela"


# --------------------------------------------------------------------------- #
# Chunking por parágrafos
# --------------------------------------------------------------------------- #
def _dividir_paragrafos(texto: str) -> list[str]:
    """Divide o texto em parágrafos por linha em branco (``\\n\\n``).

    Preserva parágrafos completos: cada elemento retornado é um parágrafo
    inteiro, sem cortar frases. Linhas em branco/parágrafos vazios são
    descartados; espaços em excesso são normalizados.
    """
    if not texto:
        return []
    blocos = _RE_PARAGRAFO.split(texto)
    paragrafos = []
    for bloco in blocos:
        limpo = _RE_BRANCO.sub(" ", bloco).strip()
        if limpo:
            paragrafos.append(limpo)
    return paragrafos


def _estimar_caracteres_alvo(alvo_tokens: int | None) -> int:
    return max(1, int((alvo_tokens or ALVO_TOKENS) * CHARS_POR_TOKEN))


def chunk_paragrafos(texto: str, alvo_tokens: int | None = None) -> list[str]:
    """Agrupa parágrafos completos em Chunks de tamanho alvo (``~ALVO_TOKENS``).

    Nunca corta um parágrafo no meio: junta parágrafos consecutivos até atingir
    o alvo de caracteres; um parágrafo maior que o alvo vira um Chunk próprio.
    Retorna a lista de Chunks (cada um com um ou mais parágrafos completos).
    """
    paragrafos = _dividir_paragrafos(texto)
    if not paragrafos:
        return []
    alvo_caracteres = _estimar_caracteres_alvo(alvo_tokens)

    chunks: list[str] = []
    atual: list[str] = []
    tamanho_atual = 0
    for paragrafo in paragrafos:
        tamanho_paragrafo = len(paragrafo)
        # um parágrafo isolado sob o alvo mas que estouraria o alvo no grupo:
        # fecha o grupo atual e começa um novo com este parágrafo.
        if (atual and tamanho_atual + tamanho_paragrafo > alvo_caracteres):
            chunks.append(" ".join(atual))
            atual = []
            tamanho_atual = 0
        atual.append(paragrafo)
        tamanho_atual += tamanho_paragrafo
    if atual:
        chunks.append(" ".join(atual))
    return chunks


# --------------------------------------------------------------------------- #
# Modelo de embeddings (local/offline com download único documentado)
# --------------------------------------------------------------------------- #
def carregar_modelo(nome_modelo: str = MODELO_DEFAULT,
                    local_files_only: bool = False):
    """Carrega o modelo SentenceTransformer pelo nome.

    Na primeira execução, se o modelo ainda não estiver no cache local da
    HuggingFace, permite o **download único** (exceção documentada ao offline
    AD-1, decisão humana); a partir daí é carregado do cache local. Com
    ``local_files_only=True`` jamais tenta rede (falha clara se ausente).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "sentence-transformers não está instalado/importável. Instale com "
            "`pip install sentence-transformers` para gerar embeddings locais."
        ) from exc
    return SentenceTransformer(nome_modelo, local_files_only=local_files_only)


def gerar_embeddings(modelo, textos: list[str]) -> list:
    """Gera embeddings locais para ``textos`` e os devolve como lista de vetores.

    Garante que cada texto tenha um vetor correspondente (mesmo comprimento) e
    levanta ``RuntimeError`` em caso de descasamento, em vez de deixar o ``zip``
    do pipeline descartar documentos silenciosamente.
    """
    if not textos:
        return []
    vetores = modelo.encode(textos, normalize_embeddings=True)
    if len(vetores) != len(textos):
        raise RuntimeError(
            f"modelo retornou {len(vetores)} vetores para {len(textos)} textos; "
            "embeddings descasados com os Chunks."
        )
    # SentenceTransformer retorna um ndarray; normaliza para listas JSON-serializáveis.
    if hasattr(vetores, "tolist"):
        return vetores.tolist()
    return [list(v) for v in vetores]


# --------------------------------------------------------------------------- #
# Pipeline de vetorização
# --------------------------------------------------------------------------- #
def _ler_corpus(caminho_csv: str) -> tuple[list[dict], list[str]]:
    """Lê o CSV do corpus e devolve ``(linhas, erros)``.

    Linhas não-dict ou com elos perdidos são registradas nos erros, sem abortar.
    Se o header não trouxer a coluna ``texto_límpido`` (contrato da Story 1.5),
    registra erro claro em vez de indexar tudo vazio silenciosamente.
    """
    erros: list[str] = []
    if not os.path.exists(caminho_csv):
        return [], [f"corpus_csv não encontrado: {caminho_csv}"]
    try:
        with open(caminho_csv, encoding="utf-8-sig", newline="") as f:
            leitor = csv.DictReader(f)
            if leitor.fieldnames and COLUNA_TEXTO not in leitor.fieldnames:
                return [], [f"corpus_csv sem a coluna '{COLUNA_TEXTO}' (contrato 1.5)"]
            linhas = list(leitor)
        return linhas, erros
    except Exception as exc:  # noqa: BLE001
        return [], [f"corpus_csv: {exc}"]


def _item_na_janela(linha: dict) -> bool:
    return (linha.get(COLUNA_JANELA) or "").strip().lower() == "true"


def _item_com_texto(linha: dict) -> bool:
    return bool((linha.get(COLUNA_TEXTO) or "").strip())


def vetorizar_corpus(corpus_csv: str, db_path: str = DB_DEFAULT,
                     nome_modelo: str = MODELO_DEFAULT,
                     nome_colecao: str = retriever.COLECAO,
                     alvo_tokens: int | None = None,
                     local_files_only: bool = False,
                     modelo=None) -> dict:
    """Vetoriza o corpus e grava na coleção persistente; retorna o relatório.

    Fluxo:
    1. Lê o CSV (linhas fora da Janela ou com texto vazio são apenas contadas).
    2. Divide cada documento em Chunks por parágrafos (sem cortar frases).
    3. Gera embeddings locais para todos os Chunks de uma vez.
    4. Faz dedup por chave estável (``fonte::índice``) contra a coleção.
    5. Insere apenas os Chunks inexistentes e devolve o relatório.

    ``modelo`` pode ser passado diretamente (útil em testes, para mock do
    carregamento sem rede). Se ausente, ``carregar_modelo`` é invocado.
    """
    linhas, erros = _ler_corpus(corpus_csv)

    n_fora_janela = 0
    n_texto_vazio = 0
    candidatos: list[dict] = []
    for linha in linhas:
        if not _item_na_janela(linha):
            n_fora_janela += 1
            continue
        if not _item_com_texto(linha):
            n_texto_vazio += 1
            continue
        candidatos.append(linha)

    # --- Chunking --------------------------------------------------------- #
    # lista de (chunk_texto, fonte, metadados, chave)
    todos: list[tuple[str, str, dict, str]] = []
    n_chunks_por_doc: dict[str, int] = {}
    for linha in candidatos:
        fonte = (linha.get(COLUNA_FONTE) or "").strip() or "(sem fonte)"
        chunks = chunk_paragrafos(linha.get(COLUNA_TEXTO) or "", alvo_tokens)
        n_chunks_por_doc[fonte] = max(n_chunks_por_doc.get(fonte, 0), len(chunks))
        metadados_base = {
            "fonte": fonte,
            "instituicao": (linha.get(COLUNA_INSTITUICAO) or "").strip(),
            "tipo": (linha.get(COLUNA_TIPO) or "").strip(),
            "data": (linha.get(COLUNA_DATA) or "").strip(),
        }
        for idx, chunk in enumerate(chunks):
            metadados = dict(metadados_base)
            metadados["chunk_idx"] = idx
            metadados["n_chunks"] = len(chunks)
            chave = retriever.chave_chunk(fonte, idx)
            todos.append((chunk, fonte, metadados, chave))

    # --- Embeddings ------------------------------------------------------- #
    colecao = retriever.obter_colecao(db_path, nome_colecao)
    if modelo is None:
        modelo = carregar_modelo(nome_modelo, local_files_only=local_files_only)

    textos = [t[0] for t in todos]
    vetores = gerar_embeddings(modelo, textos) if todos else []

    # --- Dedup + inserção ------------------------------------------------- #
    chaves = [t[3] for t in todos]
    ja_existentes = retriever.ids_existentes(colecao, chaves) if todos else set()

    novos_ids: list[str] = []
    novos_emb: list = []
    novos_meta: list[dict] = []
    novos_docs: list[str] = []
    n_duplicatas = 0
    for (texto, _fonte, metadados, chave), vetor in zip(todos, vetores):
        if chave in ja_existentes:
            n_duplicatas += 1
            continue
        novos_ids.append(chave)
        novos_emb.append(vetor)
        novos_meta.append(metadados)
        novos_docs.append(texto)

    n_adicionados = retriever.inserir_chunks(
        colecao, novos_ids, novos_emb, novos_meta, novos_docs
    )

    # --- Relatório -------------------------------------------------------- #
    relatorio = {
        "n_total_corpus": len(linhas),
        "n_candidatos_janela_com_texto": len(candidatos),
        "n_fora_janela": n_fora_janela,
        "n_texto_vazio": n_texto_vazio,
        "n_chunks_gerados": len(todos),
        "n_chunks_adicionados": n_adicionados,
        "n_chunks_ja_existentes": n_duplicatas,
        "n_documentos_indexados": len({t[2]["fonte"] for t in todos}),
        "modelo": nome_modelo,
        "colecao": nome_colecao,
        "db_path": db_path,
        "chunks_por_documento_max": max(n_chunks_por_doc.values()) if n_chunks_por_doc else 0,
        "erros": erros,
        "metadados": {
            "versao_modulo": __version__,
            "corpus_csv": corpus_csv,
            "alvo_tokens": alvo_tokens or ALVO_TOKENS,
        },
    }
    return relatorio


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Vetoriza o corpus RFEPCT e o armazena localmente no "
                    "ChromaDB (Story 2.1)."
    )
    ap.add_argument("--corpus", default=CORPUS_DEFAULT,
                    help="CSV do corpus limpo (corpus_rfepct.csv)")
    ap.add_argument("--db", default=DB_DEFAULT,
                    help="caminho de persistência do ChromaDB (data/chroma_db)")
    ap.add_argument("--colecao", default=retriever.COLECAO,
                    help="nome da coleção ChromaDB")
    ap.add_argument("--modelo", default=MODELO_DEFAULT,
                    help="nome do modelo SentenceTransformer multilingue")
    ap.add_argument("--alvo-tokens", type=int, default=None,
                    help="tamanho alvo por Chunk em tokens (default 400)")
    ap.add_argument("--local-only", action="store_true",
                    help="não tentar download do modelo (usa só o cache local)")
    ap.add_argument("--relatorio", default=RELATORIO_DEFAULT,
                    help="JSON de saída do relatório de vetorização")
    args = ap.parse_args(argv)

    try:
        rel = vetorizar_corpus(
            args.corpus, db_path=args.db, nome_modelo=args.modelo,
            nome_colecao=args.colecao, alvo_tokens=args.alvo_tokens,
            local_files_only=args.local_only,
        )
    except Exception as exc:  # noqa: BLE001 - erro deve sair no relatório, não só em stderr
        rel = {
            "n_total_corpus": 0,
            "n_candidatos_janela_com_texto": 0,
            "n_fora_janela": 0,
            "n_texto_vazio": 0,
            "n_chunks_gerados": 0,
            "n_chunks_adicionados": 0,
            "n_chunks_ja_existentes": 0,
            "n_documentos_indexados": 0,
            "modelo": args.modelo,
            "colecao": args.colecao,
            "db_path": args.db,
            "chunks_por_documento_max": 0,
            "erros": [f"falha na vetorização: {exc}"],
            "metadados": {
                "versao_modulo": __version__,
                "corpus_csv": args.corpus,
                "alvo_tokens": args.alvo_tokens or ALVO_TOKENS,
            },
        }
        codigo_erro = True
    else:
        codigo_erro = False

    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.relatorio)) or ".",
                    exist_ok=True)
        with open(args.relatorio, "w", encoding="utf-8") as f:
            json.dump(rel, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        rel.setdefault("erros", []).append(f"falha ao gravar relatório: {exc}")

    sys.stdout.write(
        f"candidatos={rel['n_candidatos_janela_com_texto']} "
        f"fora_janela={rel['n_fora_janela']} texto_vazio={rel['n_texto_vazio']} "
        f"chunks_gerados={rel['n_chunks_gerados']} "
        f"adicionados={rel['n_chunks_adicionados']} "
        f"ja_existentes={rel['n_chunks_ja_existentes']} "
        f"relatorio={args.relatorio}\n"
    )
    # exit 0 apenas quando não há erros; exit 1 quando há erros registrados.
    return 1 if (rel["erros"] or codigo_erro) else 0


if __name__ == "__main__":
    sys.exit(main())
