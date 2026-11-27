# OPENAI + VECTOR DATABASES
**`SEMANTIC RETRIEVAL • EMBEDDING PIPELINES • HYBRID SEARCH`**

`OpenAI API` · `pgvector` · `Supabase` · `LangChain` · `Python`

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-text--embedding--3--small-412991?style=flat-square&logo=openai&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791?style=flat-square&logo=postgresql&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-Vector%20Store-3ECF8E?style=flat-square&logo=supabase&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

---

## Overview

**Hybrid Semantic Movie Recommendation Engine** is an end-to-end vector retrieval and RAG-inspired discovery application built on PostgreSQL (`pgvector`), OpenAI embeddings, and Streamlit. Rather than relying solely on exact keyword matching or static collaborative filtering, the system encodes rich narrative descriptions, thematic keywords, and TMDB metadata into **1,536-dimensional dense vectors** to support natural-language vibe search and item-to-item similarity alongside deterministic SQL filtering.

### Key Capabilities
* **Vibe-Based Semantic Search:** Query the catalog using natural language prompts (e.g., *"slow-burn neo-noir detective thriller with rain-soaked visuals and moral ambiguity"*) ranked by cosine similarity.
* **Item-to-Item Similarity:** Select any title in the database to retrieve conceptually and tonally adjacent films directly from its stored embedding vector.
* **Hybrid SQL + Vector Filtering:** Combine approximate nearest-neighbor vector search with structured PostgreSQL predicates (`genres`, `vote_average`, and `release_year`) inside single-roundtrip RPC functions.
* **Inline Curator Breakdowns:** Renders similarity scores, thematic alignment notes, and structured metadata inside each result card in the Streamlit interface.

---

## System Architecture

| Layer | Technology | Responsibility |
| :--- | :--- | :--- |
| **Data Ingestion** | Python, TMDB API, `LangChain` | Fetches movie metadata, formats structured semantic documents, and chunks long text blocks with token overlap. |
| **Embedding Pipeline** | `OpenAI API` (`text-embedding-3-small`) | Generates 1,536-dimensional vector representations in batched API calls for ingestion and live query encoding. |
| **Vector Storage** | `Supabase` (`PostgreSQL` + `pgvector`) | Persists relational metadata alongside `vector(1536)` columns indexed via HNSW (`vector_cosine_ops`). |
| **Retrieval Engine** | PostgreSQL Stored Functions (`RPC`) | Executes filtered cosine similarity queries (`1 - (embedding <=> query_embedding)`) server-side. |
| **Frontend Interface** | `Streamlit` | Provides interactive search modes, similarity thresholds, metadata filters, and per-card breakdowns. |

### Pipeline Execution Flow
1. **Document Construction (`ingest_tmdb.py`):** Pulls movie records from the TMDB API, normalizes overviews, genres, keywords, and release attributes, and applies `RecursiveCharacterTextSplitter` where needed.
2. **Batch Vectorization:** Encodes each composite movie document into a 1,536-float embedding vector using `text-embedding-3-small` and upserts records into Supabase.
3. **Indexed Hybrid Querying:** At runtime, user prompts are embedded on the fly and passed to PostgreSQL RPC functions (`match_movies` or `match_similar_movies`), applying `WHERE` filters before returning top-$k$ nearest neighbors.

---

## Database Schema (`schema.sql`)

Execute the following SQL in the Supabase SQL Editor to enable `pgvector`, create the `movies` table, build the HNSW index, and register the hybrid search RPC functions:

```sql
-- 1. Enable the pgvector extension
create extension if not exists vector;

-- 2. Create the movies table with structured metadata and 1,536-dim embeddings
create table if not exists movies (
  id bigserial primary key,
  tmdb_id integer unique not null,
  title text not null,
  overview text not null,
  genres text[] default '{}',
  release_year integer,
  vote_average numeric(3, 1),
  poster_path text,
  embedding vector(1536) not null,
  created_at timestamptz default now()
);

-- 3. Create an HNSW index for low-latency cosine similarity search
create index if not exists movies_embedding_hnsw_idx
  on movies
  using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);

-- 4. RPC: Semantic vibe search with relational metadata filters
create or replace function match_movies (
  query_embedding vector(1536),
  match_threshold float default 0.35,
  match_count int default 12,
  min_rating float default 0.0,
  min_year int default 1900,
  filter_genre text default null
)
returns table (
  id bigint,
  tmdb_id integer,
  title text,
  overview text,
  genres text[],
  release_year integer,
  vote_average numeric,
  poster_path text,
  similarity float
)
language sql stable
as $$
  select
    m.id,
    m.tmdb_id,
    m.title,
    m.overview,
    m.genres,
    m.release_year,
    m.vote_average,
    m.poster_path,
    1 - (m.embedding <=> query_embedding) as similarity
  from movies m
  where 1 - (m.embedding <=> query_embedding) > match_threshold
    and coalesce(m.vote_average, 0) >= min_rating
    and coalesce(m.release_year, 0) >= min_year
    and (filter_genre is null or filter_genre = any(m.genres))
  order by m.embedding <=> query_embedding
  limit match_count;
$$;

-- 5. RPC: Item-to-item similarity from an existing movie's embedding
create or replace function match_similar_movies (
  source_tmdb_id integer,
  match_count int default 8,
  min_rating float default 0.0
)
returns table (
  id bigint,
  tmdb_id integer,
  title text,
  overview text,
  genres text[],
  release_year integer,
  vote_average numeric,
  poster_path text,
  similarity float
)
language sql stable
as $$
  with target as (
    select embedding
    from movies
    where tmdb_id = source_tmdb_id
    limit 1
  )
  select
    m.id,
    m.tmdb_id,
    m.title,
    m.overview,
    m.genres,
    m.release_year,
    m.vote_average,
    m.poster_path,
    1 - (m.embedding <=> target.embedding) as similarity
  from movies m, target
  where m.tmdb_id <> source_tmdb_id
    and coalesce(m.vote_average, 0) >= min_rating
  order by m.embedding <=> target.embedding
  limit match_count;
$$;
```

---

## Repository Structure

```text
Hybrid_Semantic_Movie_Recommendation_Engine/
├── .streamlit/
│   └── config.toml          # GitHub-dark theme and layout configuration
├── app.py                   # Streamlit search UI, filters, and RPC query client
├── ingest_tmdb.py           # TMDB fetcher, LangChain text prep, and OpenAI batch embedder
├── schema.sql               # PostgreSQL table, HNSW index, and RPC definitions
├── requirements.txt         # Pinned Python dependencies
├── .gitignore               # Excludes .env, virtual environments, and cache files
└── README.md                # Project architecture and setup documentation
```

---

## Setup & Local Execution

### 1. Clone and Create a Virtual Environment

```bash
cd Hybrid_Semantic_Movie_Recommendation_Engine
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Core packages included in `requirements.txt`:
* `openai`
* `supabase`
* `langchain-text-splitters`
* `streamlit`
* `requests`
* `python-dotenv`

### 3. Configure Environment Variables

Create a `.env` file in the project root with your API credentials:

```env
OPENAI_API_KEY=sk-proj-...
SUPABASE_URL=[https://xyzcompany.supabase.co](https://xyzcompany.supabase.co)
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
TMDB_API_KEY=your_tmdb_v3_api_key
```

### 4. Ingest Data and Launch the App

```bash
# Fetch TMDB catalog, compute 1,536-dim embeddings, and upsert to Supabase
python ingest_tmdb.py

# Start the Streamlit application
streamlit run app.py
```

---

## Technical Design Notes

* **Unified Relational + Vector Architecture:** Using `pgvector` inside Supabase avoids dual-write synchronization issues between a standalone vector store and a relational database while supporting native array filtering (`filter_genre = any(m.genres)`) in the same query plan.
* **Cosine Distance Operator (`<=>`):** Because OpenAI embeddings are normalized, cosine distance provides scale-invariant semantic similarity across short queries and multi-sentence plot synopses.
* **HNSW Indexing:** Configured with `m = 16` and `ef_construction = 64` to balance high recall and sub-50ms query latency without requiring periodic index retraining as new TMDB releases are ingested.
