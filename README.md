# RFEPCT-Inovação-RAG

**Ferramenta analítica para tese de doutorado em Ciências Sociais** — automatiza a coleta, a estruturação e a análise de como **a inovação é comunicada e registrada oficialmente** pelas instituições da **Rede Federal de Educação Profissional, Científica e Tecnológica (RFEPCT)**.

## Finalidade acadêmica

Esta base sustenta uma investigação sobre o modo como as instituições da RFEPCT *publicizam* a inovação: de um lado, os **registros de propriedade intelectual** depositados no INPI por essas instituições (patentes, programas de computador, desenhos industriais, marcas); de outro, as **notícias e itens publicizados** nos portais oficiais da Rede (Institutos Federais, CEFETs, Colégio Pedro II, MEC, MCTI, CONIF).

O corpus resultante — limitado à Janela 2019–2026 — é preparado para análise qualitativa e quantitativa, buscando detectar o **recorte social latente** (evidências de impacto social, desenvolvimento regional e governança pública) para além da correspondência superficial de palavras-chave, na linha da diretriz sociológica não negociável do projeto: nunca classificar sob ótica de antagonismo com o mercado, e sim de confluência entre desenvolvimento tecnológico e impacto social.

## Processamento local e offline

**Todo o processamento é local/offline.** O sistema opera com *LLM local* (Ollama) e *banco vetorial local* (ChromaDB). **É proibido** o uso de LLM em nuvem — decisão deliberada para proteger os dados de pesquisa e preservar o controle total do pesquisador sobre a semântica da análise.

Esta etapa (Story 1.1) é **apenas de scaffold**: estrutura de pastas, dependências e configuração de alvos. **Não há download de dados nem chamada a qualquer LLM.** Os dumps do INPI, a raspagem de portais e a vetorização ocorrerão em etapas posteriores (Stories 1.2–1.5 e Epic 2).

> ⚠️ Alternativa documentada: o SAD prevê **langchain** como orquestrador de RAG. Caso se prefira **llama-index**, é uma troca válida e documentada — mas a decisão corrente desta run é **langchain**.

## Estrutura de pastas

A organização espelha a arquitetura (SAD §2):

```
rfepct-inovacao-rag/
├── data/
│   ├── raw/          # dumps INPI e HTML bruto dos portais
│   ├── processed/    # textos límpidos e corpus estruturado (entrada do Epic 2)
│   └── chroma_db/    # banco vetorial local (uso no Epic 2)
├── config/
│   └── urls_rede_federal.json   # dicionário de alvos da RFEPCT (config-driven)
├── scrapers/         # módulos de ingestão (INPI + portais) — Stories 1.2–1.4
├── rag_engine/       # motor RAG local (embeddings/retriever/grounding) — Epic 2
├── requirements.txt  # dependências do SAD
└── README.md
```

Pastas usadas apenas por histórias posteriores (`analysis`, `jobs`, `data/audit`, `config/crosswalk_cnpj.json`) serão criadas nas próprias histórias.

## Como executar o pipeline

Este repositório ainda não implementa o código de coleta. O fluxo previsto (Stories 1.2–1.5) será:

1. **Preparar o ambiente local** (Python 3.x):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate       # Windows
   pip install -r requirements.txt
   playwright install          # navegadores para portais dinâmicos
   ```
   > O diretório `.venv/` **não deve** ser versionado (ver `.gitignore`, quando o VCS for inicializado).

2. **Validar o acesso ao INPI (Story 1.2)** — prototipar o endpoint HTTPS de Dados Abertos, conferir schema e qualidade do campo de data antes de aceitar a escala. A interface BuscaWeb é metodologia **rejeitada** (CAPTCHA/sessão/bloqueio de IP).

3. **Ingerir registros de PI (Story 1.3)** — download em lote (requests + zipfile/tarfile), parsing em pedaços (pandas `chunksize`) e filtragem de depositantes RFEPCT por crosswalk de CNPJ + raiz nominal, com retenção por união (preserva co-titularidade).

4. **Raspar portais (Story 1.4)** — config-driven por domínio; seletores próprios por portal; idempotência por URL canônica; respeito a `robots.txt` e atraso de polidez.

5. **Estruturar e limpar (Story 1.5)** — aplicar a Janela 2019–2026 **sem perda silenciosa**: itens fora da Janela ou sem data são retidos e marcados, nunca descartados em silêncio.

## Verificação desta etapa (scaffold)

- JSON válido e cobertura do dicionário: `python -c "import json; json.load(open('config/urls_rede_federal.json', encoding='utf-8'))"`
- Pastas presentes: conferir `config`, `scrapers`, `rag_engine` e `data/raw|processed|chroma_db`.

## Stack prevista (SAD)

`beautifulsoup4`, `requests`, `playwright`, `pandas` (coleta/parsing) · `chromadb` + `sentence-transformers` (vetorização) · `langchain` + `ollama` (RAG local) · `streamlit` (interface). Todas listadas em `requirements.txt`.

---

*Ferramenta de pesquisa científica. Processamento local. Fase atual: planejamento e inicialização (scaffold).*
