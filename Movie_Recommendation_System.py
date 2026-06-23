# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║          🎬  MOVIE RECOMMENDATION SYSTEM — Google Colab Notebook            ║
# ║         Content-Based Filtering using TF-IDF + KNN + Cosine Similarity      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# DATASET REQUIRED (from Kaggle — "The Movies Dataset"):
#   • movies_metadata.csv
#   • keywords.csv
#   • credits.csv


# ──────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install & Import Libraries
# ──────────────────────────────────────────────────────────────────────────────
# Install the core machine-learning and data-processing libraries.
# --quiet suppresses the verbose pip output to keep the notebook clean.

# In[1]:
!pip install scikit-learn pandas numpy --quiet

import pandas as pd          # DataFrame manipulation
import numpy as np           # Numerical operations
import ast                   # Safely parse stringified Python dicts/lists from CSV
import warnings
warnings.filterwarnings('ignore')   # Suppress non-critical deprecation warnings

# NLP / ML tools
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

print('✅ All libraries imported successfully!')


# ──────────────────────────────────────────────────────────────────────────────
# CELL 2 — Upload Dataset Files
# ──────────────────────────────────────────────────────────────────────────────
# Use Colab's built-in file upload widget to bring the three CSV files
# into the Colab runtime environment.
# Upload all three files when prompted: movies_metadata.csv, keywords.csv, credits.csv

# In[2]:
from google.colab import files

print('Please upload your 3 CSV files: movies_metadata.csv, keywords.csv, credits.csv')
uploaded = files.upload()
print('\n✅ Files uploaded:', list(uploaded.keys()))


# ──────────────────────────────────────────────────────────────────────────────
# CELL 3 — Load CSVs into DataFrames
# ──────────────────────────────────────────────────────────────────────────────
# Read each uploaded CSV file into a separate Pandas DataFrame.
# low_memory=False prevents mixed-type warnings on the large movies file.

# In[3]:
movies_df   = pd.read_csv('movies_metadata.csv', low_memory=False)
keywords_df = pd.read_csv('keywords.csv')
credits_df  = pd.read_csv('credits.csv')

# Print shape of each dataset to verify successful loading
print(f'movies_metadata : {movies_df.shape[0]:,} rows  |  {movies_df.shape[1]} columns')
print(f'keywords        : {keywords_df.shape[0]:,} rows  |  {keywords_df.shape[1]} columns')
print(f'credits         : {credits_df.shape[0]:,} rows  |  {credits_df.shape[1]} columns')

print('\n--- movies_metadata columns ---')
print(movies_df.columns.tolist())


# ──────────────────────────────────────────────────────────────────────────────
# CELL 4 — Clean & Prepare movies_metadata
# ──────────────────────────────────────────────────────────────────────────────
# • Keep only the columns we actually need for the recommender
# • Fix corrupt 'id' values (some rows contain file paths instead of integers)
# • Parse release_date to extract a clean 'year' column
# • Fill NaN text fields with empty strings so vectorizers don't crash

# In[4]:
# Keep only relevant columns
movies_df = movies_df[[
    'id', 'title', 'overview', 'genres',
    'release_date', 'vote_average', 'vote_count',
    'popularity', 'runtime', 'tagline'
]].copy()

# Convert id to numeric — some rows have corrupt values like '/path/...'
movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
movies_df.dropna(subset=['id'], inplace=True)
movies_df['id'] = movies_df['id'].astype(int)

# Drop duplicate movie ids to prevent duplicate recommendations
movies_df.drop_duplicates(subset=['id'], inplace=True)

# Extract release year from release_date string
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
movies_df['year'] = movies_df['release_date'].dt.year

# Fill missing text fields with empty strings
movies_df['overview'].fillna('', inplace=True)
movies_df['tagline'].fillna('', inplace=True)
movies_df['title'].fillna('', inplace=True)

print(f'✅ movies_df cleaned  →  {movies_df.shape[0]:,} rows')
movies_df.head(3)


# ──────────────────────────────────────────────────────────────────────────────
# CELL 5 — Helper Functions for Parsing JSON-like Columns
# ──────────────────────────────────────────────────────────────────────────────
# The genres, keywords, cast, and crew columns are stored as stringified
# Python lists of dicts (e.g. "[{'id': 28, 'name': 'Action'}, ...]").
# These two helpers safely parse and extract values from those strings.

# In[5]:
def safe_parse(val):
    """
    Safely convert a stringified Python list or dict back to a Python object.
    Returns an empty list if parsing fails (malformed data).
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []


def extract_names(val, key='name', limit=None):
    """
    Extract a list of values for a given key from a list of dicts.

    Args:
        val   : raw string or already-parsed list of dicts
        key   : the dictionary key to extract (default 'name')
        limit : optional max number of items to return

    Returns:
        A plain Python list of strings.
    Example:
        "[{'name': 'Action'}, {'name': 'Drama'}]"  →  ['Action', 'Drama']
    """
    parsed = safe_parse(val) if isinstance(val, str) else val
    if not isinstance(parsed, list):
        return []
    names = [d[key] for d in parsed if isinstance(d, dict) and key in d]
    return names[:limit] if limit else names


# ──────────────────────────────────────────────────────────────────────────────
# CELL 6 — Extract Genre Names from movies_metadata
# ──────────────────────────────────────────────────────────────────────────────
# Apply the helper to parse the 'genres' column into a clean Python list
# e.g. "[{'id': 28, 'name': 'Action'}]"  →  ['Action']

# In[6]:
movies_df['genres_list'] = movies_df['genres'].apply(
    lambda x: extract_names(x, 'name')
)

print('Sample genres_list:')
print(movies_df[['title', 'genres_list']].head(4).to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# CELL 7 — Clean & Parse keywords.csv
# ──────────────────────────────────────────────────────────────────────────────
# Fix id dtype and parse the stringified keywords column into a list of names.
# Keywords capture plot themes like 'dystopia', 'time travel', etc.

# In[7]:
keywords_df['id'] = pd.to_numeric(keywords_df['id'], errors='coerce').astype('Int64')
keywords_df.dropna(subset=['id'], inplace=True)
keywords_df['id'] = keywords_df['id'].astype(int)

# Parse stringified keyword dicts into plain name lists
keywords_df['keywords_list'] = keywords_df['keywords'].apply(
    lambda x: extract_names(x, 'name')
)

print('Sample keywords_list:')
print(keywords_df[['id', 'keywords_list']].head(3).to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# CELL 8 — Clean & Parse credits.csv (Cast + Director)
# ──────────────────────────────────────────────────────────────────────────────
# Extract:
#   • Top 5 cast members per movie (actors matter most for recommendations)
#   • The Director from the crew list (identified by job == 'Director')

# In[8]:
credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce').astype('Int64')
credits_df.dropna(subset=['id'], inplace=True)
credits_df['id'] = credits_df['id'].astype(int)

# Top 5 cast members per movie (limiting avoids noise from minor roles)
credits_df['cast_list'] = credits_df['cast'].apply(
    lambda x: extract_names(x, 'name', limit=5)
)


def get_director(crew_str):
    """
    Scan the crew list and return the name of the person with job == 'Director'.
    Returns an empty string if no director entry is found.
    """
    crew = safe_parse(crew_str) if isinstance(crew_str, str) else crew_str
    if not isinstance(crew, list):
        return ''
    for member in crew:
        if isinstance(member, dict) and member.get('job') == 'Director':
            return member.get('name', '')
    return ''


credits_df['director'] = credits_df['crew'].apply(get_director)

print('Sample credits:')
print(credits_df[['id', 'cast_list', 'director']].head(3).to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# CELL 9 — Merge All Three DataFrames
# ──────────────────────────────────────────────────────────────────────────────
# Left-join keywords and credits onto movies using the shared 'id' column.
# Left join ensures we keep all movies even if keywords/credits data is missing.

# In[9]:
df = movies_df.merge(
    keywords_df[['id', 'keywords_list']], on='id', how='left'
).merge(
    credits_df[['id', 'cast_list', 'director']], on='id', how='left'
)

# Fill missing lists and strings to avoid NaN errors downstream
df['keywords_list'] = df['keywords_list'].apply(lambda x: x if isinstance(x, list) else [])
df['cast_list']     = df['cast_list'].apply(lambda x: x if isinstance(x, list) else [])
df['director']      = df['director'].fillna('')

print(f'✅ Merged dataframe  →  {df.shape[0]:,} rows  |  {df.shape[1]} columns')
print(df.columns.tolist())


# ──────────────────────────────────────────────────────────────────────────────
# CELL 10 — Build the "Soup" — Combined Feature String per Movie
# ──────────────────────────────────────────────────────────────────────────────
# The "soup" is a single string of tokens representing each movie's identity.
# It combines genres, keywords, cast, director (weighted 3x), and overview words.
#
# Why remove spaces in names?
#   "Tom Hanks" → "tomhanks" so the vectorizer treats it as ONE token, not two.
#
# Why repeat the director 3 times?
#   Director has outsized influence on a film's style — triple-weighting
#   makes the similarity model respect that.

# In[10]:
def clean_token(name):
    """Lowercase and remove spaces so multi-word names become single tokens."""
    return str(name).lower().replace(' ', '')


# Apply token cleaning to each list/string column
df['genres_clean']   = df['genres_list'].apply(lambda lst: [clean_token(n) for n in lst])
df['keywords_clean'] = df['keywords_list'].apply(lambda lst: [clean_token(n) for n in lst])
df['cast_clean']     = df['cast_list'].apply(lambda lst: [clean_token(n) for n in lst])
df['director_clean'] = df['director'].apply(clean_token)


def make_soup(row):
    """
    Concatenate all cleaned feature tokens into a single space-separated string.
    Director is repeated 3 times to give it more weight in the TF-IDF model.
    First 50 words of the overview are appended for thematic context.
    """
    parts = (
        row['genres_clean']
        + row['keywords_clean']
        + row['cast_clean']
        + [row['director_clean']] * 3    # 3x weight on director
    )
    overview_words = str(row['overview']).lower().split()[:50]
    return ' '.join(parts) + ' ' + ' '.join(overview_words)


df['soup'] = df.apply(make_soup, axis=1)

print('✅ Content soup created!')
print('\nSample soup (Toy Story):')
print(df[df['title'] == 'Toy Story']['soup'].values[0][:300])


# ──────────────────────────────────────────────────────────────────────────────
# CELL 11 — TF-IDF Vectorization & KNN Model Training
# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF (Term Frequency–Inverse Document Frequency):
#   Converts the soup strings into a numerical matrix where each column is a
#   word/bigram and each value reflects how uniquely important that word is
#   to a particular movie vs. the whole corpus.
#
# KNN (K-Nearest Neighbors):
#   Finds the n most similar movies to a query movie by measuring cosine
#   distance between their TF-IDF vectors. Cosine distance ignores magnitude
#   (movie length, etc.) and focuses purely on the angle between vectors.

# In[11]:
# Reset index so iloc lookups are safe after all the merges and drops
df.reset_index(drop=True, inplace=True)

# Build the TF-IDF matrix from the soup column
tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),   # use both single words and 2-word phrases
    min_df=2,             # ignore tokens that appear in fewer than 2 movies
    stop_words='english'  # remove common words like 'the', 'a', 'is'
)

tfidf_matrix = tfidf.fit_transform(df['soup'])
print(f'✅ TF-IDF matrix shape: {tfidf_matrix.shape}')
#   → (num_movies, num_unique_tokens)

# Train KNN model with cosine distance on the sparse TF-IDF matrix
knn_model = NearestNeighbors(
    n_neighbors=11,       # fetch 11 neighbors: index 0 = the movie itself, 1–10 = recommendations
    metric='cosine',
    algorithm='brute',    # brute force works well for sparse high-dimensional data
    n_jobs=-1             # use all available CPU cores
)
knn_model.fit(tfidf_matrix)
print('✅ KNN model fitted!')

# Build a lowercase title → DataFrame index lookup dictionary
title_index = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()
print('✅ Title index built!')


# ──────────────────────────────────────────────────────────────────────────────
# CELL 12 — Helper: Pretty-Print Results
# ──────────────────────────────────────────────────────────────────────────────
# Shared display function used by all search functions below.
# Formats the result DataFrame consistently with a title banner.

# In[12]:
def display_results(result_df, title='Results', max_rows=20):
    """
    Print a formatted table of movie results.

    Args:
        result_df : DataFrame containing the filtered/ranked movies
        title     : Banner title string shown above the table
        max_rows  : Maximum number of rows to display
    """
    cols = ['title', 'year', 'vote_average', 'genres_list', 'director']
    cols = [c for c in cols if c in result_df.columns]   # only keep existing cols
    out  = result_df[cols].head(max_rows).copy()
    out['year']         = out['year'].astype('Int64') if 'year' in out.columns else ''
    out['vote_average'] = out['vote_average'].round(1) if 'vote_average' in out.columns else ''
    out.index = range(1, len(out) + 1)   # 1-based index for readability
    print(f'\n{"="*70}')
    print(f'  {title}')
    print(f'{"="*70}')
    print(out.to_string())
    print(f'{"─"*70}\n')


# ──────────────────────────────────────────────────────────────────────────────
# CELL 13 — Function 1: Search Movies by Genre
# ──────────────────────────────────────────────────────────────────────────────
# Filters movies whose genres_list contains the requested genre, then ranks
# them by vote_average. A min_votes threshold removes movies with very few
# ratings (which can have inflated scores).
#
# Usage: get_movies_by_genre('Action')

# In[13]:
def get_movies_by_genre(genre, top_n=20, min_votes=50):
    """
    Return the top_n highest-rated movies for a given genre.

    Args:
        genre     : Genre name string (case-insensitive), e.g. 'Sci-Fi'
        top_n     : Number of movies to return (default 20)
        min_votes : Minimum vote_count to filter out obscure movies (default 50)
    """
    genre_lower = genre.lower()
    # Check if any genre in the list matches (case-insensitive)
    mask = df['genres_list'].apply(
        lambda g: any(genre_lower == x.lower() for x in g)
    )
    result = df[mask & (df['vote_count'] >= min_votes)].copy()
    result = result.sort_values('vote_average', ascending=False)
    if result.empty:
        print(f'  ⚠️  No movies found for genre "{genre}".')
        return
    display_results(result, title=f'Top {top_n} Movies — Genre: {genre.title()}', max_rows=top_n)


# ──────────────────────────────────────────────────────────────────────────────
# CELL 14 — Function 2: Search Movies by Actor
# ──────────────────────────────────────────────────────────────────────────────
# Searches the top-5 cast list for an actor name (exact match first,
# then partial match fallback) and returns their filmography sorted by rating.
#
# Usage: get_movies_by_actor('Tom Hanks')

# In[14]:
def get_movies_by_actor(actor_name, top_n=20):
    """
    Return movies featuring the given actor, sorted by rating.

    Args:
        actor_name : Full or partial actor name (case-insensitive)
        top_n      : Number of movies to return (default 20)
    """
    actor_lower = actor_name.lower()

    # First try exact match
    mask = df['cast_list'].apply(
        lambda lst: any(actor_lower == a.lower() for a in lst)
    )
    result = df[mask].copy().sort_values('vote_average', ascending=False)

    # If no exact match found, fall back to partial (substring) match
    if result.empty:
        mask2  = df['cast_list'].apply(
            lambda lst: any(actor_lower in a.lower() for a in lst)
        )
        result = df[mask2].copy().sort_values('vote_average', ascending=False)

    if result.empty:
        print(f'  ⚠️  No movies found for actor "{actor_name}".')
        return
    display_results(result, title=f'Movies featuring: {actor_name.title()}', max_rows=top_n)


# ──────────────────────────────────────────────────────────────────────────────
# CELL 15 — Function 3: Search Movies by Director
# ──────────────────────────────────────────────────────────────────────────────
# Matches director names using a case-insensitive substring search on the
# 'director' column and returns their filmography ranked by vote_average.
#
# Usage: get_movies_by_director('Christopher Nolan')

# In[15]:
def get_movies_by_director(director_name, top_n=20):
    """
    Return all movies by the given director, sorted by rating.

    Args:
        director_name : Full or partial director name (case-insensitive)
        top_n         : Number of movies to return (default 20)
    """
    dir_lower = director_name.lower()
    # str.contains allows partial matching (e.g. 'Nolan' matches 'Christopher Nolan')
    mask   = df['director'].str.lower().str.contains(dir_lower, na=False)
    result = df[mask].copy().sort_values('vote_average', ascending=False)
    if result.empty:
        print(f'  ⚠️  No movies found for director "{director_name}".')
        return
    display_results(result, title=f'Movies by Director: {director_name.title()}', max_rows=top_n)


# ──────────────────────────────────────────────────────────────────────────────
# CELL 16 — Function 4: Search Movies by Release Year
# ──────────────────────────────────────────────────────────────────────────────
# Filters movies released in the specified year and ranks them by vote_average.
# A min_votes guard prevents obscure low-vote films from appearing at the top.
#
# Usage: get_movies_by_year(2010)

# In[16]:
def get_movies_by_year(year, top_n=20, min_votes=30):
    """
    Return top-rated movies released in a specific year.

    Args:
        year      : Integer year, e.g. 2010
        top_n     : Number of movies to return (default 20)
        min_votes : Minimum vote count filter (default 30)
    """
    result = df[
        (df['year'] == year) & (df['vote_count'] >= min_votes)
    ].copy().sort_values('vote_average', ascending=False)
    if result.empty:
        print(f'  ⚠️  No movies found for year {year}.')
        return
    display_results(result, title=f'Top Movies of {year}', max_rows=top_n)


# ──────────────────────────────────────────────────────────────────────────────
# CELL 17 — Function 5: Best Rated Movie of a Year (IMDb Weighted Formula)
# ──────────────────────────────────────────────────────────────────────────────
# Uses the IMDb Bayesian weighted rating formula to find the single best movie:
#
#     WR = (v / (v + m)) × R  +  (m / (v + m)) × C
#
#   Where:
#     R = movie's raw average rating
#     v = number of votes the movie has
#     m = minimum votes threshold (acts as a Bayesian prior strength)
#     C = global mean rating across all movies
#
# This formula penalizes movies with few votes — a film rated 9.0 from 5 votes
# scores lower than one rated 8.5 from 10,000 votes.
#
# Usage: get_best_rated_of_year(2014)

# In[17]:
def get_best_rated_of_year(year, min_votes=100):
    """
    Find and display the single best movie of a year using IMDb's weighted rating.

    Args:
        year      : Integer year, e.g. 2008
        min_votes : Minimum votes threshold for the Bayesian formula (default 100)
    """
    subset = df[(df['year'] == year) & (df['vote_count'] >= min_votes)].copy()
    if subset.empty:
        # Relax the vote filter if no results found
        subset = df[df['year'] == year].copy()
    if subset.empty:
        print(f'  ⚠️  No movies found for year {year}.')
        return

    C = df['vote_average'].mean()   # global mean rating (the Bayesian prior)
    m = min_votes

    # Compute weighted rating for each movie in this year's subset
    subset['weighted_rating'] = (
        (subset['vote_count'] / (subset['vote_count'] + m)) * subset['vote_average']
        + (m / (subset['vote_count'] + m)) * C
    )
    best = subset.sort_values('weighted_rating', ascending=False).iloc[0]

    print(f'\n{"="*70}')
    print(f'  🏆  Best Rated Movie of {year}')
    print(f'{"="*70}')
    print(f'  Title        : {best["title"]}')
    print(f'  Year         : {int(best["year"]) if pd.notna(best["year"]) else "N/A"}')
    print(f'  Rating       : {best["vote_average"]:.1f}  ({int(best["vote_count"]):,} votes)')
    print(f'  W. Rating    : {best["weighted_rating"]:.3f}')
    print(f'  Genre        : {", ".join(best["genres_list"]) if best["genres_list"] else "N/A"}')
    print(f'  Director     : {best["director"] if best["director"] else "N/A"}')
    print(f'  Overview     : {str(best["overview"])[:200]}...')
    print(f'{"─"*70}\n')


# ──────────────────────────────────────────────────────────────────────────────
# CELL 18 — Function 6: KNN Content-Based Similar Movie Finder
# ──────────────────────────────────────────────────────────────────────────────
# Queries the KNN model with the TF-IDF vector of the input movie to find
# the n most similar movies by cosine distance on their soup representations.
#
# Steps:
#   1. Look up movie by title (exact, then partial fallback)
#   2. Fetch its TF-IDF row vector from tfidf_matrix
#   3. Ask KNN for the 11 nearest neighbors (first is always itself)
#   4. Convert cosine distance → cosine similarity (1 - distance)
#   5. Display ranked results
#
# Usage: get_similar_movies_knn('Interstellar')

# In[18]:
def get_similar_movies_knn(movie_title, n_recommendations=10):
    """
    Find movies most similar to the given title using KNN on TF-IDF vectors.

    Args:
        movie_title      : Title string (case-insensitive), e.g. 'The Dark Knight'
        n_recommendations: Number of similar movies to return (default 10)
    """
    movie_lower = movie_title.lower().strip()

    # Step 1: Locate the movie's row index
    if movie_lower in title_index:
        idx = title_index[movie_lower]
    else:
        # Partial match: find any title that contains the query string
        matches = [t for t in title_index.index if movie_lower in t]
        if not matches:
            print(f'  ⚠️  Movie "{movie_title}" not found in the dataset.')
            print('  Tip: check spelling or use the full title.')
            return
        idx = title_index[matches[0]]
        print(f'  ℹ️  Exact match not found. Using closest: "{df.loc[idx, "title"]}"')

    # Step 2: Retrieve the movie's TF-IDF vector and query KNN
    query_vec = tfidf_matrix[idx]
    distances, indices = knn_model.kneighbors(
        query_vec, n_neighbors=n_recommendations + 1   # +1 to skip the movie itself
    )

    # Step 3: Exclude index 0 (the query movie itself)
    similar_indices   = indices.flatten()[1:]
    similar_distances = distances.flatten()[1:]

    result = df.iloc[similar_indices].copy()
    # Step 4: Convert cosine distance to cosine similarity score (0–1 scale)
    result['similarity'] = (1 - similar_distances).round(4)

    # Step 5: Display results
    src = df.iloc[idx]
    print(f'\n{"="*70}')
    print(f'  🎬  Movies Similar to: "{src["title"]}" ({int(src["year"]) if pd.notna(src["year"]) else "N/A"})')
    print(f'      Genre: {", ".join(src["genres_list"])}  |  Director: {src["director"]}')
    print(f'{"="*70}')
    cols = ['title', 'year', 'vote_average', 'similarity', 'genres_list', 'director']
    out  = result[cols].copy()
    out['year'] = out['year'].astype('Int64')
    out.index   = range(1, len(out) + 1)
    print(out.to_string())
    print(f'{"─"*70}\n')


print('✅ All 6 recommendation functions are ready!')


# ──────────────────────────────────────────────────────────────────────────────
# CELL 19 — Interactive Widget UI
# ──────────────────────────────────────────────────────────────────────────────
# Builds a clickable GUI inside the Colab notebook using ipywidgets.
# Users can select a feature, type a query, adjust result count, and click Run
# without having to write any code themselves.
#
# Components:
#   • Dropdown  — select which of the 6 features to use
#   • Text box  — enter the search query (genre name, actor, year, movie title)
#   • Slider    — choose how many results to display (5–50)
#   • Button    — trigger the search
#   • Output    — displays results below the controls

# In[19]:
import ipywidgets as widgets
from IPython.display import display, clear_output

# ── Dropdown: choose which recommendation feature to use ──────────────────
feature_dropdown = widgets.Dropdown(
    options=[
        ('🔍  Search by Genre',               'genre'),
        ('🎭  Search by Actor',                'actor'),
        ('🎬  Search by Director',             'director'),
        ('📅  Movies by Year',                 'year'),
        ('🏆  Best Rated Movie of a Year',     'best'),
        ('🤖  Find Similar Movies  (KNN)',     'knn'),
    ],
    description='Feature:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

# ── Text input: the user's search query ──────────────────────────────────
query_input = widgets.Text(
    placeholder='e.g. Action  /  Tom Hanks  /  2014  /  Interstellar',
    description='Search:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='500px')
)

# ── Slider: number of results to display ─────────────────────────────────
n_slider = widgets.IntSlider(
    value=10, min=5, max=50, step=5,
    description='Results:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

# ── Run button: triggers the search ──────────────────────────────────────
run_button = widgets.Button(
    description='  Run',
    button_style='success',
    icon='play',
    layout=widgets.Layout(width='120px', height='35px')
)

# ── Output area: results appear here ─────────────────────────────────────
output_area = widgets.Output()

# ── Header label ─────────────────────────────────────────────────────────
title_label = widgets.HTML(
    value="<h3 style='color:#4A90D9;'>🎬 Movie Recommendation System</h3>"
)


def on_run_clicked(b):
    """
    Callback executed when the Run button is clicked.
    Reads widget values, validates input, and calls the appropriate function.
    """
    with output_area:
        clear_output(wait=True)   # clear previous results before showing new ones
        feature = feature_dropdown.value
        query   = query_input.value.strip()
        n       = n_slider.value

        if not query:
            print('⚠️  Please enter a search query.')
            return

        if feature == 'genre':
            get_movies_by_genre(query, top_n=n)

        elif feature == 'actor':
            get_movies_by_actor(query, top_n=n)

        elif feature == 'director':
            get_movies_by_director(query, top_n=n)

        elif feature == 'year':
            if not query.isdigit():
                print('⚠️  Please enter a valid year (e.g. 2010).')
                return
            get_movies_by_year(int(query), top_n=n)

        elif feature == 'best':
            if not query.isdigit():
                print('⚠️  Please enter a valid year (e.g. 2014).')
                return
            get_best_rated_of_year(int(query))

        elif feature == 'knn':
            get_similar_movies_knn(query, n_recommendations=n)


# Bind the click event handler to the button
run_button.on_click(on_run_clicked)

# Assemble and render the full widget layout
display(widgets.VBox([
    title_label,
    widgets.HTML("<hr>"),
    feature_dropdown,
    query_input,
    n_slider,
    run_button,
    widgets.HTML("<hr>"),
    output_area
]))


# ──────────────────────────────────────────────────────────────────────────────
# CELL 20 — Quick Demo: Test All 6 Functions
# ──────────────────────────────────────────────────────────────────────────────
# Run sample queries against each function to verify the system works correctly.
# These can also serve as usage examples.

# In[20]:
# 1. Top Sci-Fi movies
get_movies_by_genre('Sci-Fi', top_n=15)

# 2. All movies featuring Leonardo DiCaprio
get_movies_by_actor('Leonardo DiCaprio', top_n=10)

# 3. Christopher Nolan's filmography
get_movies_by_director('Christopher Nolan', top_n=10)

# 4. Best movies of 2014
get_movies_by_year(2014, top_n=15)

# 5. Best rated movie of 2008 (The Dark Knight era)
get_best_rated_of_year(2008)

# 6. Movies similar to Interstellar using KNN
get_similar_movies_knn('Interstellar', n_recommendations=10)


# ──────────────────────────────────────────────────────────────────────────────
# CELL 21 — Alternative Recommender: Precomputed Cosine Similarity Matrix
# ──────────────────────────────────────────────────────────────────────────────
# This is a second recommendation approach using a full N×N cosine similarity
# matrix instead of KNN. It is faster to query but uses much more memory,
# so it is restricted to popular movies only (vote_count >= 200).
#
# How it differs from KNN:
#   • KNN   : computes distances on-the-fly for each query → memory efficient
#   • Cosine: precomputes ALL pairwise similarities at once → query is instant
#     but the matrix can be huge (e.g. 10,000 movies → 10,000×10,000 floats)
#
# CountVectorizer is used here instead of TF-IDF — it simply counts token
# frequencies without the IDF normalization, which sometimes works better for
# short, keyword-heavy text like the soup strings.

# In[21]:
# Restrict to popular movies to keep the matrix size manageable
pop_df = df[df['vote_count'] >= 200].copy().reset_index(drop=True)
print(f'Popular movies subset: {len(pop_df):,} films')

# Build a raw count matrix (no IDF weighting)
count_vec    = CountVectorizer(stop_words='english', min_df=2)
count_matrix = count_vec.fit_transform(pop_df['soup'])

# Compute the full pairwise cosine similarity matrix (N × N)
cosine_sim   = cosine_similarity(count_matrix, count_matrix)

# Build a title → index lookup for this subset
pop_title_index = pd.Series(
    pop_df.index, index=pop_df['title'].str.lower()
).drop_duplicates()

print(f'✅ Cosine similarity matrix built: {cosine_sim.shape}')


def get_similar_movies_cosine(movie_title, n=10):
    """
    Find similar movies using the precomputed cosine similarity matrix.
    Only works for movies with vote_count >= 200 (popular subset).

    Args:
        movie_title : Title string (case-insensitive)
        n           : Number of similar movies to return (default 10)
    """
    movie_lower = movie_title.lower().strip()

    # Locate the movie; fall back to partial match if needed
    if movie_lower not in pop_title_index:
        matches = [t for t in pop_title_index.index if movie_lower in t]
        if not matches:
            print(f'  ⚠️  "{movie_title}" not found in the popular-movies subset.')
            return
        movie_lower = matches[0]
        print(f'  ℹ️  Closest match: "{pop_df.loc[pop_title_index[movie_lower], "title"]}"')

    idx = pop_title_index[movie_lower]

    # Get all pairwise similarity scores for this movie, sort descending
    sims = sorted(
        list(enumerate(cosine_sim[idx])),
        key=lambda x: x[1],
        reverse=True
    )

    # Skip index 0 (the movie itself) and take the next n
    sims = sims[1:n + 1]
    movie_indices = [i for i, _ in sims]

    result = pop_df.iloc[movie_indices].copy()
    result['similarity'] = [round(s, 4) for _, s in sims]

    src = pop_df.iloc[idx]
    print(f'\n{"="*70}')
    print(f'  🎬  [Cosine] Movies Similar to: "{src["title"]}"')
    print(f'{"="*70}')
    cols = ['title', 'year', 'vote_average', 'similarity', 'genres_list', 'director']
    out  = result[cols].copy()
    out['year'] = out['year'].astype('Int64')
    out.index   = range(1, len(out) + 1)
    print(out.to_string())
    print(f'{"─"*70}\n')


# ── Demo: compare KNN vs Cosine results for the same movie ───────────────
print('--- KNN Recommender ---')
get_similar_movies_knn('The Dark Knight', n_recommendations=10)

print('--- Cosine Similarity Recommender ---')
get_similar_movies_cosine('The Dark Knight', n=10)
