import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import APIError, OpenAI
from supabase import Client, create_client

st.set_page_config(page_title="TMDB pgvector Movie Recommender", layout="wide")

# ==============================================================================
# 1. BACKEND SECRET LOADING & CACHED RESOURCES
# ==============================================================================
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

if not OPENAI_API_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    st.error(
        "Missing credentials in .env. Ensure OPENAI_API_KEY, SUPABASE_URL, and SUPABASE_KEY are set."
    )
    st.stop()

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w342"

GENRE_OPTIONS = [
    "All Genres",
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Thriller",
    "War",
    "Western",
]

CURATOR_SYSTEM_PROMPT = (
    "You are a concise, perceptive film curator. Given a user's preference and a list of "
    "matched movies from our TMDB vector database, write 2-3 sharp sentences per movie explaining "
    "why it fits their request based ONLY on the provided plot, keywords, director, cast, and rating. "
    "Never recommend any movie outside the provided list. "
    "Return ONLY a valid JSON object mapping each movie's exact string 'tmdb_id' to its explanation string."
)


@st.cache_resource
def get_clients() -> tuple[OpenAI, Client]:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return openai_client, supabase_client


@st.cache_data(ttl=300)
def load_movie_catalog() -> list[dict]:
    _, supabase_client = get_clients()
    # Select only lightweight metadata for dropdowns—never pull 1536-float vectors into UI memory
    res = (
        supabase_client.table("movies")
        .select("tmdb_id, title, release_year")
        .order("title")
        .execute()
    )
    return res.data or []


# ==============================================================================
# 2. HYBRID SEARCH, PER-MOVIE CURATOR NOTES & CARD RENDERER
# ==============================================================================
def search_movies_by_vibe(
    vibe_query: str,
    match_threshold: float,
    match_count: int,
    min_rating: float,
    min_year: int,
    max_year: int,
    filter_genre: str | None,
    openai_client: OpenAI,
    supabase_client: Client,
) -> list[dict]:
    embedding_response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=vibe_query,
    )
    query_vector = embedding_response.data[0].embedding

    rpc_response = supabase_client.rpc(
        "match_movies",
        {
            "query_embedding": query_vector,
            "match_threshold": match_threshold,
            "match_count": match_count,
            "min_rating": min_rating,
            "min_year": min_year,
            "max_year": max_year,
            "filter_genre": filter_genre,
        },
    ).execute()

    return rpc_response.data or []


def explain_recommendations(
    user_intent: str,
    movies: list[dict],
    openai_client: OpenAI,
) -> dict[str, str]:
    movie_blocks = []
    for m in movies:
        genres_str = ", ".join(m.get("genres") or [])
        keywords_str = ", ".join((m.get("keywords") or [])[:8])
        movie_blocks.append(
            f"- tmdb_id: \"{m['tmdb_id']}\" | Title: {m['title']} ({m.get('release_year', 'N/A')}) | "
            f"TMDB Rating: {m.get('vote_average', 'N/A')}/10 | Director: {m.get('director', 'N/A')} | "
            f"Genres: {genres_str} | Themes: {keywords_str}\n"
            f"  Plot: {m['overview']}"
        )

    prompt = (
        f"User Request: {user_intent}\n\n"
        f"Matched Movies:\n" + "\n".join(movie_blocks)
    )

    completion = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": CURATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    raw_json = completion.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw_json)
        return {str(k): str(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        return {}


def render_movie_cards(movies: list[dict], pitches_by_id: dict[str, str]) -> None:
    for movie in movies:
        sim_pct = round(float(movie["similarity"]) * 100, 1)
        genres_text = ", ".join(movie.get("genres") or [])
        keywords_text = ", ".join((movie.get("keywords") or [])[:6])
        curator_note = pitches_by_id.get(str(movie["tmdb_id"]), "").strip()

        with st.container(border=True):
            col_poster, col_info = st.columns([1, 4])
            with col_poster:
                if movie.get("poster_path"):
                    st.image(
                        f"{TMDB_POSTER_BASE}{movie['poster_path']}",
                        use_container_width=True,
                    )
            with col_info:
                st.markdown(
                    f"### {movie['title']} ({movie.get('release_year', 'N/A')}) "
                    f"— TMDB **{movie.get('vote_average', 'N/A')}/10** | Match: **{sim_pct}%**"
                )
                if movie.get("tagline"):
                    st.markdown(f"*\"{movie['tagline']}\"*")
                st.caption(
                    f"Genres: {genres_text} | Director: {movie.get('director', 'N/A')} | "
                    f"Cast: {movie.get('cast_members', 'N/A')}"
                )
                if keywords_text:
                    st.caption(f"Themes: {keywords_text}")
                st.write(movie["overview"])

                if curator_note:
                    st.info(f"**Curator Note:** {curator_note}")


# ==============================================================================
# 3. STREAMLIT UI
# ==============================================================================
openai_client, supabase_client = get_clients()

with st.sidebar:
    st.header("Hybrid SQL Filters")
    st.caption("Applied in PostgreSQL before ranking vector similarity.")

    min_rating = st.slider(
        "Minimum TMDB Rating",
        min_value=0.0,
        max_value=9.0,
        value=6.8,
        step=0.1,
    )
    year_range = st.slider(
        "Release Year Range",
        min_value=1940,
        max_value=2026,
        value=(1985, 2026),
    )
    selected_genre = st.selectbox("Required Genre", options=GENRE_OPTIONS, index=0)

    st.divider()
    st.header("Vector Search Settings")
    match_count = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
    match_threshold = st.slider(
        "Min Cosine Similarity (Vibe Mode)",
        min_value=0.20,
        max_value=0.80,
        value=0.35,
        step=0.05,
    )
    include_ai_pitch = st.checkbox("Generate AI Curator Pitch", value=True)
    st.divider()
    st.caption("Movie metadata and posters provided by TMDB.")

active_genre = None if selected_genre == "All Genres" else selected_genre
min_year, max_year = year_range

st.title("TMDB pgvector Movie Recommender")
st.caption(
    "Semantic plot, keyword, and mood matching powered by TMDB metadata, OpenAI embeddings, and Supabase pgvector."
)

tab_vibe, tab_similar = st.tabs(
    ["1. Recommend by Vibe / Plot", "2. More Like This Movie ($0 Vector Lookup)"]
)

# ------------------------------------------------------------------------------
# TAB 1: NATURAL-LANGUAGE VIBE & PLOT SEARCH
# ------------------------------------------------------------------------------
with tab_vibe:
    vibe_input = st.text_area(
        "Describe the story, mood, or themes you want to watch",
        height=100,
        placeholder="e.g., A mind-bending sci-fi thriller about memory manipulation, dreams, and high-stakes espionage",
    )

    if st.button("Find Matching Movies", type="primary"):
        clean_vibe = vibe_input.strip()
        if not clean_vibe:
            st.warning("Describe a movie vibe or plot above first.")
        else:
            with st.spinner("Running hybrid vector + SQL search..."):
                try:
                    results = search_movies_by_vibe(
                        vibe_query=clean_vibe,
                        match_threshold=match_threshold,
                        match_count=match_count,
                        min_rating=min_rating,
                        min_year=min_year,
                        max_year=max_year,
                        filter_genre=active_genre,
                        openai_client=openai_client,
                        supabase_client=supabase_client,
                    )

                    if len(results) == 0:
                        st.info(
                            "No movies matched both your semantic query and sidebar SQL filters. "
                            "Try lowering Minimum TMDB Rating or Similarity Threshold."
                        )
                    else:
                        pitches_by_id: dict[str, str] = {}
                        if include_ai_pitch:
                            with st.spinner("Writing per-movie curator notes..."):
                                pitches_by_id = explain_recommendations(
                                    user_intent=clean_vibe,
                                    movies=results,
                                    openai_client=openai_client,
                                )

                        st.subheader(f"Top {len(results)} Matched Films")
                        render_movie_cards(movies=results, pitches_by_id=pitches_by_id)

                except APIError as err:
                    st.error(f"OpenAI API error: {err}")
                except Exception as err:
                    st.error(f"Database search error: {err}")

# ------------------------------------------------------------------------------
# TAB 2: ITEM-TO-ITEM VECTOR SIMILARITY ($0 EMBEDDING COST)
# ------------------------------------------------------------------------------
with tab_similar:
    col_desc, col_refresh = st.columns([5, 1])
    with col_desc:
        st.write(
            "Pick a movie already in your Supabase database to find its nearest semantic neighbors "
            "directly inside PostgreSQL—zero OpenAI embedding calls required."
        )
    with col_refresh:
        if st.button("Refresh Catalog", use_container_width=True):
            load_movie_catalog.clear()
            st.rerun()

    try:
        catalog = load_movie_catalog()
    except Exception as err:
        st.error(f"Could not load movie catalog from Supabase: {err}")
        catalog = []

    if len(catalog) == 0:
        st.warning("Your 'movies' table is empty. Run `python ingest_tmdb.py` first.")
    else:
        movie_options = {
            f"{m['title']} ({m.get('release_year', 'N/A')})": int(m["tmdb_id"])
            for m in catalog
        }
        chosen_movie_label = st.selectbox(
            "Select a reference movie",
            options=list(movie_options.keys()),
        )

        if st.button("Find Similar Movies", type="primary"):
            target_id = movie_options[chosen_movie_label]
            with st.spinner("Computing cosine neighbors in PostgreSQL..."):
                try:
                    rpc_res = supabase_client.rpc(
                        "recommend_similar_movies",
                        {
                            "target_tmdb_id": target_id,
                            "match_count": match_count,
                            "min_rating": min_rating,
                            "min_year": min_year,
                            "max_year": max_year,
                            "filter_genre": active_genre,
                        },
                    ).execute()
                    similar_movies = rpc_res.data or []

                    if len(similar_movies) == 0:
                        st.info("No similar movies matched your current sidebar rating/year/genre filters.")
                    else:
                        pitches_by_id: dict[str, str] = {}
                        if include_ai_pitch:
                            with st.spinner("Writing per-movie curator notes..."):
                                pitches_by_id = explain_recommendations(
                                    user_intent=f"Movies similar in theme, tone, and style to {chosen_movie_label}",
                                    movies=similar_movies,
                                    openai_client=openai_client,
                                )

                        st.subheader(f"Top {len(similar_movies)} Similar Films")
                        render_movie_cards(movies=similar_movies, pitches_by_id=pitches_by_id)

                except Exception as err:
                    st.error(f"Recommendation query failed: {err}")