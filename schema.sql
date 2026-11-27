create extension if not exists vector;

-- Clean reset if re-running setup
drop function if exists public.match_movies(vector(1536), float, int, float, int, int, text);
drop function if exists public.recommend_similar_movies(bigint, int, float, int, int, text);
drop table if exists public.movies cascade;

-- 1. Movies table combining structured TMDB metadata with pgvector embeddings
create table public.movies (
  id bigserial primary key,
  tmdb_id bigint unique not null,
  title text not null,
  release_year int,
  vote_average float,
  vote_count int,
  genres text[] not null default '{}',
  keywords text[] not null default '{}',
  director text,
  cast_members text,
  tagline text,
  overview text not null,
  poster_path text,
  embedding vector(1536) not null
);

-- Indexes for fast hybrid SQL filtering
create index idx_movies_rating on public.movies (vote_average);
create index idx_movies_year on public.movies (release_year);
create index idx_movies_genres on public.movies using gin (genres);

-- 2. Hybrid search RPC: natural-language vibe + SQL metadata filters
create or replace function public.match_movies (
  query_embedding vector(1536),
  match_threshold float,
  match_count int,
  min_rating float default 0.0,
  min_year int default 1900,
  max_year int default 2030,
  filter_genre text default null
)
returns table (
  tmdb_id bigint,
  title text,
  release_year int,
  vote_average float,
  vote_count int,
  genres text[],
  keywords text[],
  director text,
  cast_members text,
  tagline text,
  overview text,
  poster_path text,
  similarity float
)
language sql stable
as $$
  select
    m.tmdb_id,
    m.title,
    m.release_year,
    m.vote_average,
    m.vote_count,
    m.genres,
    m.keywords,
    m.director,
    m.cast_members,
    m.tagline,
    m.overview,
    m.poster_path,
    1 - (m.embedding <=> query_embedding) as similarity
  from public.movies m
  where 1 - (m.embedding <=> query_embedding) > match_threshold
    and coalesce(m.vote_average, 0) >= min_rating
    and coalesce(m.release_year, 0) between min_year and max_year
    and (filter_genre is null or filter_genre = any(m.genres))
  order by similarity desc
  limit match_count;
$$;

-- 3. Item-to-item RPC: find similar movies using an existing DB vector ($0 embedding cost)
create or replace function public.recommend_similar_movies (
  target_tmdb_id bigint,
  match_count int,
  min_rating float default 0.0,
  min_year int default 1900,
  max_year int default 2030,
  filter_genre text default null
)
returns table (
  tmdb_id bigint,
  title text,
  release_year int,
  vote_average float,
  vote_count int,
  genres text[],
  keywords text[],
  director text,
  cast_members text,
  tagline text,
  overview text,
  poster_path text,
  similarity float
)
language sql stable
as $$
  with target as (
    select embedding
    from public.movies
    where tmdb_id = target_tmdb_id
    limit 1
  )
  select
    m.tmdb_id,
    m.title,
    m.release_year,
    m.vote_average,
    m.vote_count,
    m.genres,
    m.keywords,
    m.director,
    m.cast_members,
    m.tagline,
    m.overview,
    m.poster_path,
    1 - (m.embedding <=> target.embedding) as similarity
  from public.movies m
  cross join target
  where m.tmdb_id != target_tmdb_id
    and coalesce(m.vote_average, 0) >= min_rating
    and coalesce(m.release_year, 0) between min_year and max_year
    and (filter_genre is null or filter_genre = any(m.genres))
  order by similarity desc
  limit match_count;
$$;

-- 4. Role privileges & Row Level Security
grant usage on schema public to anon, authenticated, service_role;
grant select on table public.movies to anon, authenticated;
grant all on table public.movies to service_role;
grant usage, select on sequence public.movies_id_seq to service_role;

grant execute on function public.match_movies(vector(1536), float, int, float, int, int, text)
  to anon, authenticated, service_role;
grant execute on function public.recommend_similar_movies(bigint, int, float, int, int, text)
  to anon, authenticated, service_role;

alter table public.movies enable row level security;

create policy "Allow read-only access to movies"
  on public.movies
  for select
  to anon, authenticated
  using (true);