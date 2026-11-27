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
BATCH_SIZE = 50


# ==============================================================================
# 2. POOLED TMDB HTTP CLIENT & WATCH PROVIDER PARSER
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
        backoff_factor=1.0,
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


def extract_watch_providers(watch_providers_payload: dict, force_netflix: bool = False) -> list[str]:
    results = watch_providers_payload.get("results", {})
    providers: list[str] = ["Netflix"] if force_netflix else []

    for region in ("US", "KE"):
        region_data = results.get(region, {})
        for item in region_data.get("flatrate", []):
            raw_name = (item.get("provider_name") or "").strip()
            if not raw_name:
                continue
            normalized = "Netflix" if "netflix" in raw_name.lower() else raw_name
            if normalized not in providers:
                providers.append(normalized)

    return providers


# ==============================================================================
# 3. INSPECT EXISTING MOVIES & BACKFILL STREAMING PROVIDERS ($0 OPENAI COST)
# ==============================================================================
try:
    existing_res = supabase_client.table("movies").select("tmdb_id, watch_providers").execute()
    existing_rows = existing_res.data or []
except Exception as err:
    tmdb_session.close()
    raise RuntimeError(f"Failed to query existing movies from Supabase: {err}") from err

existing_ids = {int(row["tmdb_id"]) for row in existing_rows}
rows_missing_providers = [
    int(row["tmdb_id"]) for row in existing_rows if not row.get("watch_providers")
]

print(f"Found {len(existing_ids)} movies already indexed in Supabase.")

if len(rows_missing_providers) > 0:
    print(f"Backfilling streaming providers for {len(rows_missing_providers)} existing movies ($0 OpenAI cost)...")
    for idx, movie_id in enumerate(rows_missing_providers, start=1):
        wp_data = fetch_tmdb_json(f"/movie/{movie_id}/watch/providers")
        providers = extract_watch_providers(wp_data)
        # Store ["None"] if no flatrate streamer exists so backfill does not repeat on future runs
        stored_providers = providers if providers else ["None"]
        supabase_client.table("movies").update({"watch_providers": stored_providers}).eq(
            "tmdb_id", movie_id
        ).execute()
        if idx % 50 == 0:
            print(f"Updated streaming info for {idx}/{len(rows_missing_providers)} existing movies...")
        time.sleep(0.05)

# ==============================================================================
# 4. DISCOVER RECENT (2023-2026) & NETFLIX MOVIES FROM TMDB
# ==============================================================================
discovery_feeds = [
    {
        "label": "Recent Releases (2023-2026)",
        "pages": 8,
        "force_netflix": False,
        "params": {
            "language": "en-US",
            "sort_by": "popularity.desc",
            "primary_release_date.gte": "2023-01-01",
            "primary_release_date.lte": "2026-12-31",
            "vote_count.gte": 150,
            "include_adult": "false",
        },
    },
    {
        "label": "Netflix Streaming Catalog (Provider ID 8)",
        "pages": 8,
        "force_netflix": True,
        "params": {
            "language": "en-US",
            "sort_by": "popularity.desc",
            "watch_region": "US",
            "with_watch_providers": "8",
            "with_watch_monetization_types": "flatrate",
            "vote_count.gte": 100,
            "include_adult": "false",
        },
    },
]

candidate_movies: dict[int, bool] = {}  # tmdb_id -> force_netflix flag

for feed in discovery_feeds:
    print(f"Scanning TMDB feed: {feed['label']}...")
    for page in range(1, feed["pages"] + 1):
        query_params = dict(feed["params"])
        query_params["page"] = page
        discover_data = fetch_tmdb_json("/discover/movie", params=query_params)

        for item in discover_data.get("results", []):
            movie_id = int(item["id"])
            if movie_id in existing_ids:
                continue
            if movie_id not in candidate_movies:
                candidate_movies[movie_id] = feed["force_netflix"]
            elif feed["force_netflix"]:
                candidate_movies[movie_id] = True

candidate_list = list(candidate_movies.items())

if len(candidate_list) == 0:
    print("No new recent or Netflix movies to ingest.")
    tmdb_session.close()
    raise SystemExit(0)

print(f"Discovered {len(candidate_list)} new movies. Processing in batches of {BATCH_SIZE}...")

# ==============================================================================
# 5. INCREMENTAL CHECKPOINTS: FETCH -> EMBED -> UPSERT EVERY 50 MOVIES
# ==============================================================================
total_stored = 0

for batch_start in range(0, len(candidate_list), BATCH_SIZE):
    batch_items = candidate_list[batch_start : batch_start + BATCH_SIZE]
    records = []
    texts_to_embed = []

    for movie_id, force_netflix in batch_items:
        details = fetch_tmdb_json(
            f"/movie/{movie_id}",
            params={
                "language": "en-US",
                "append_to_response": "credits,keywords,watch/providers",
            },
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

        providers = extract_watch_providers(
            details.get("watch/providers", {}),
            force_netflix=force_netflix,
        )
        stored_providers = providers if providers else ["None"]

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
            f"Streaming: {', '.join(providers) if providers else 'Theatrical / VOD'} | "
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
                "watch_providers": stored_providers,
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
        processed_count = min(batch_start + BATCH_SIZE, len(candidate_list))
        print(
            f"Checkpoint saved: {total_stored} new movies indexed "
            f"({processed_count}/{len(candidate_list)} processed)..."
        )
    except Exception as err:
        tmdb_session.close()
        raise RuntimeError(
            f"Supabase upsert failed on batch starting at {batch_start}: {err}"
        ) from err

tmdb_session.close()
print(f"Done. Indexed {total_stored} recent and Netflix movies into Supabase.")