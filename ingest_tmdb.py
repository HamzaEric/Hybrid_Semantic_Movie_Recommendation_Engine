import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import APIError, OpenAI
from requests.adapters import HTTPAdapter
from supabase import Client, create_client
from urllib3.util.retry import Retry

# ==============================================================================
# 1. LOAD SECRETS & VALIDATE BACKEND PRIVILEGES
# ==============================================================================
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

TMDB_READ_TOKEN = os.getenv("TMDB_READ_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.getenv("SUPABASE_KEY", "").strip()
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()

if not TMDB_READ_TOKEN or not OPENAI_API_KEY or not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError(
        "Missing credentials in .env. Ensure TMDB_READ_TOKEN, OPENAI_API_KEY, "
        "SUPABASE_URL, and SUPABASE_SERVICE_KEY are all set."
    )

if SUPABASE_SERVICE_KEY == SUPABASE_ANON_KEY or SUPABASE_SERVICE_KEY.startswith("sb_publishable_"):
    raise RuntimeError(
        "SUPABASE_SERVICE_KEY is set to your public/publishable key. "
        "Replace it in .env with your Supabase 'sb_secret_...' or 'service_role' key."
    )

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions (matches vector(1536))
PAGES_TO_FETCH = 15  # 20 movies per page -> up to 300 high-vote films
BATCH_SIZE = 50      # Fetch, embed, and save every 50 movies as a checkpoint


# ==============================================================================
# 2. POOLED TMDB HTTP CLIENT WITH RETRY BACKOFF
# ==============================================================================
def create_tmdb_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {TMDB_READ_TOKEN}",
            "Accept": "application/json",
        }
    )

    retry_strategy = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,  # Waits 1s, 2s, 4s, 8s, 16s on connection resets or 429/5xx
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    return session


tmdb_session = create_tmdb_session()


def fetch_tmdb_json(endpoint: str, params: dict | None = None) -> dict:
    try:
        response = tmdb_session.get(
            f"{TMDB_BASE_URL}{endpoint}",
            params=params or {},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as err:
        raise RuntimeError(f"TMDB request failed for '{endpoint}': {err}") from err


# ==============================================================================
# 3. CHECK ALREADY-INDEXED MOVIES IN SUPABASE
# ==============================================================================
try:
    existing_res = supabase_client.table("movies").select("tmdb_id").execute()
    existing_ids = {int(row["tmdb_id"]) for row in (existing_res.data or [])}
except Exception as err:
    raise RuntimeError(f"Failed to query existing movies from Supabase: {err}") from err

print(f"Found {len(existing_ids)} movies already indexed in Supabase.")

# ==============================================================================
# 4. DISCOVER CANDIDATE MOVIE IDS FROM TMDB
# ==============================================================================
candidate_ids = []
for page in range(1, PAGES_TO_FETCH + 1):
    discover_data = fetch_tmdb_json(
        "/discover/movie",
        params={
            "language": "en-US",
            "sort_by": "vote_count.desc",
            "vote_count.gte": 1500,
            "include_adult": "false",
            "page": page,
        },
    )
    for item in discover_data.get("results", []):
        movie_id = int(item["id"])
        if movie_id not in existing_ids and movie_id not in candidate_ids:
            candidate_ids.append(movie_id)

if len(candidate_ids) == 0:
    print("No new movies to ingest. Increase PAGES_TO_FETCH if you want a larger catalog.")
    tmdb_session.close()
    raise SystemExit(0)

print(f"Discovered {len(candidate_ids)} new movies. Processing in batches of {BATCH_SIZE}...")

# ==============================================================================
# 5. INCREMENTAL CHECKPOINTS: FETCH -> EMBED -> UPSERT EVERY 50 MOVIES
# ==============================================================================
total_stored = 0

for batch_start in range(0, len(candidate_ids), BATCH_SIZE):
    batch_ids = candidate_ids[batch_start : batch_start + BATCH_SIZE]
    records = []
    texts_to_embed = []

    for movie_id in batch_ids:
        details = fetch_tmdb_json(
            f"/movie/{movie_id}",
            params={"language": "en-US", "append_to_response": "credits,keywords"},
        )

        overview = (details.get("overview") or "").strip()
        if not overview:
            continue

        title = (details.get("title") or "").strip()
        release_date = details.get("release_date") or ""
        release_year = (
            int(release_date[:4])
            if len(release_date) >= 4 and release_date[:4].isdigit()
            else None
        )
        vote_average = round(float(details.get("vote_average") or 0.0), 2)
        vote_count = int(details.get("vote_count") or 0)
        tagline = (details.get("tagline") or "").strip()
        poster_path = details.get("poster_path")

        genre_list = [g["name"] for g in details.get("genres", []) if g.get("name")]
        keyword_list = [
            kw["name"]
            for kw in details.get("keywords", {}).get("keywords", [])
            if kw.get("name")
        ][:12]

        crew = details.get("credits", {}).get("crew", [])
        directors = [person["name"] for person in crew if person.get("job") == "Director"]
        director_str = ", ".join(directors) if directors else "Unknown"

        cast = details.get("credits", {}).get("cast", [])
        top_cast = [person["name"] for person in cast[:5] if person.get("name")]
        cast_str = ", ".join(top_cast)

        semantic_profile = (
            f"Title: {title} ({release_year or 'N/A'}) | "
            f"Genres: {', '.join(genre_list) if genre_list else 'N/A'} | "
            f"Themes & Keywords: {', '.join(keyword_list) if keyword_list else 'N/A'} | "
            f"Director: {director_str} | "
            f"Cast: {cast_str} | "
            f"Tagline: {tagline} | "
            f"Plot: {overview}"
        )

        texts_to_embed.append(semantic_profile)
        records.append(
            {
                "tmdb_id": movie_id,
                "title": title,
                "release_year": release_year,
                "vote_average": vote_average,
                "vote_count": vote_count,
                "genres": genre_list,
                "keywords": keyword_list,
                "director": director_str,
                "cast_members": cast_str,
                "tagline": tagline,
                "overview": overview,
                "poster_path": poster_path,
            }
        )
        time.sleep(0.08)

    if len(records) == 0:
        continue

    try:
        embedding_response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts_to_embed,
        )
    except APIError as err:
        tmdb_session.close()
        raise RuntimeError(
            f"OpenAI embeddings failed on batch starting at {batch_start}: {err}"
        ) from err

    for i, rec in enumerate(records):
        rec["embedding"] = embedding_response.data[i].embedding

    try:
        supabase_client.table("movies").upsert(records, on_conflict="tmdb_id").execute()
        total_stored += len(records)
        processed_count = min(batch_start + BATCH_SIZE, len(candidate_ids))
        print(
            f"Checkpoint saved: {total_stored} movies indexed "
            f"({processed_count}/{len(candidate_ids)} processed)..."
        )
    except Exception as err:
        tmdb_session.close()
        raise RuntimeError(
            f"Supabase upsert failed on batch starting at {batch_start}: {err}"
        ) from err

tmdb_session.close()
print(f"Done. Successfully indexed {total_stored} new TMDB movies into Supabase.")