import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(page_title="MovieMart", page_icon="🎬", layout="wide")

GENRE_STYLE = {
    "Sci-Fi":    {"color": "#3b82f6", "emoji": "🚀"},
    "Comedy":    {"color": "#f59e0b", "emoji": "😂"},
    "Drama":     {"color": "#8b5cf6", "emoji": "🎭"},
    "Horror":    {"color": "#ef4444", "emoji": "👻"},
    "Romance":   {"color": "#ec4899", "emoji": "💕"},
    "Animation": {"color": "#10b981", "emoji": "🧸"},
    "Thriller":  {"color": "#64748b", "emoji": "🔪"},
    "Action":    {"color": "#dc2626", "emoji": "💥"},
}
DEFAULT_STYLE = {"color": "#7c3aed", "emoji": "🎬"}

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top left, #1a1a2e 0%, #0e0e16 45%);
    }
    h1, h2, h3, h4 { font-family: 'Helvetica Neue', sans-serif; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        background: #15151f;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.55);
    }
    .match-badge {
        display: inline-block;
        background: linear-gradient(90deg, #7c3aed, #ec4899);
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 999px;
        margin-bottom: 6px;
    }
    .stButton>button {
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.08);
        transition: border-color 0.15s ease;
    }
    .stButton>button:hover { border-color: #ec4899; color: #ec4899; }
    .poster-wrap {
        position: relative;
        width: 100%;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 10px;
    }
    .poster-img {
        width: 100%;
        height: 230px;
        object-fit: cover;
        display: block;
        transition: transform 0.35s ease;
    }
    .poster-wrap:hover .poster-img { transform: scale(1.08); }
    .poster-tint {
        position: absolute;
        inset: 0;
        mix-blend-mode: color;
        opacity: 0.55;
        pointer-events: none;
    }
    .poster-overlay {
        position: absolute;
        left: 0; right: 0; bottom: 0;
        background: linear-gradient(to top, rgba(0,0,0,0.95), rgba(0,0,0,0.05));
        padding: 30px 10px 8px 10px;
    }
    .poster-title {
        color: #fff;
        font-weight: 700;
        font-size: 0.95rem;
        line-height: 1.2;
    }
    .poster-meta {
        color: #d8d8e2;
        font-size: 0.78rem;
        margin-top: 3px;
    }
    .poster-badge {
        position: absolute;
        top: 8px;
        left: 8px;
        background: linear-gradient(90deg, #7c3aed, #ec4899);
        color: white;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 999px;
        z-index: 2;
    }
    .genre-chip {
        position: absolute;
        top: 8px;
        right: 8px;
        background: rgba(0,0,0,0.55);
        backdrop-filter: blur(4px);
        color: white;
        font-size: 0.85rem;
        width: 28px; height: 28px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        z-index: 2;
    }
    .price-tag {
        color: #22d3ee;
        font-weight: 700;
        font-size: 1rem;
    }
    .hero-banner {
        padding: 32px 28px;
        border-radius: 16px;
        margin-bottom: 22px;
        background: linear-gradient(120deg, rgba(124,58,237,0.35), rgba(236,72,153,0.25)), #15151f;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .hero-title { margin-bottom: 4px; font-size: 2.2rem; }
    .hero-sub { color: #c9c9d6; font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# DATA LOADING + RECOMMENDATION ENGINE
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    return df

@st.cache_resource
def build_similarity_matrix(df: pd.DataFrame):
    # Combine genre (weighted x3 so genre matters more) + overview text
    # so the recommender considers both theme and story content.
    combined_text = (df["genre"] + " ") * 3 + df["overview"]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return sim_matrix

def get_recommendations(movie_id: int, df: pd.DataFrame, sim_matrix, top_n: int = 5):
    """Returns (dataframe of recommended movies, dict of movie_id -> match %)."""
    idx = df.index[df["movie_id"] == movie_id][0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]
    rec_indices = [s[0] for s in scores]
    match_pct = {int(df.iloc[i]["movie_id"]): round(score * 100) for i, score in scores}
    return df.iloc[rec_indices], match_pct

def get_recommendations_for_user(seed_movie_ids, df: pd.DataFrame, sim_matrix, top_n: int = 5):
    """Averages similarity across every movie the user has interacted with
    (cart + wishlist) so the homepage recommendation reflects their overall
    taste, not just a single click."""
    if not seed_movie_ids:
        return None, None
    seed_indices = [df.index[df["movie_id"] == mid][0] for mid in seed_movie_ids if mid in df["movie_id"].values]
    if not seed_indices:
        return None, None
    avg_scores = sim_matrix[seed_indices].mean(axis=0)
    scored = [(i, s) for i, s in enumerate(avg_scores) if i not in seed_indices]
    scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    rec_indices = [i for i, _ in scored]
    match_pct = {int(df.iloc[i]["movie_id"]): round(s * 100) for i, s in scored}
    return df.iloc[rec_indices], match_pct

movies_df = load_data()
similarity_matrix = build_similarity_matrix(movies_df)

# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
if "cart" not in st.session_state:
    st.session_state.cart = {}  # movie_id -> qty
if "wishlist" not in st.session_state:
    st.session_state.wishlist = set()
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None
if "page" not in st.session_state:
    st.session_state.page = "Browse"

def add_to_cart(movie_id):
    st.session_state.cart[movie_id] = st.session_state.cart.get(movie_id, 0) + 1
    st.toast("Added to cart 🛒")

def remove_from_cart(movie_id):
    if movie_id in st.session_state.cart:
        del st.session_state.cart[movie_id]

def toggle_wishlist(movie_id):
    if movie_id in st.session_state.wishlist:
        st.session_state.wishlist.remove(movie_id)
    else:
        st.session_state.wishlist.add(movie_id)
        st.toast("Added to wishlist ❤️")

def view_movie(movie_id):
    st.session_state.selected_movie = movie_id
    st.session_state.page = "Detail"

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
st.sidebar.title("🎬 MovieMart")

def _set_page_from_nav():
    st.session_state.page = st.session_state.nav_radio

if "nav_radio" not in st.session_state:
    st.session_state.nav_radio = "Browse"

# key="nav_radio" makes the widget remember its own position across reruns;
# on_change only fires on an actual user click, so opening a movie's Detail
# page (set elsewhere) is never silently overwritten by this widget re-rendering.
st.sidebar.radio(
    "Navigate", ["Browse", "Cart", "Wishlist"],
    key="nav_radio", on_change=_set_page_from_nav,
)

st.sidebar.markdown("---")
st.sidebar.metric("Items in cart", sum(st.session_state.cart.values()))
st.sidebar.metric("Wishlist items", len(st.session_state.wishlist))

st.sidebar.markdown("---")
search_query = st.sidebar.text_input("Search movies", "")
genre_options = ["All"] + sorted(movies_df["genre"].unique().tolist())
selected_genre = st.sidebar.selectbox("Filter by genre", genre_options)

# ----------------------------------------------------------------------
# MOVIE CARD COMPONENT
# ----------------------------------------------------------------------
def movie_card(row, key_prefix="", match_pct=None):
    with st.container(border=True):
        style = GENRE_STYLE.get(row["genre"], DEFAULT_STYLE)
        badge_html = f'<div class="poster-badge">🎯 {match_pct}% Match</div>' if match_pct is not None else ""
        # NOTE: built as ONE continuous string with no leading indentation —
        # markdown treats 4+ leading spaces on a line as a code block, which
        # would make this render as literal text instead of an image.
        poster_html = (
            f'<div class="poster-wrap">'
            f'<img src="{row["poster_url"]}" class="poster-img" />'
            f'<div class="poster-tint" style="background:{style["color"]};"></div>'
            f'{badge_html}'
            f'<div class="genre-chip">{style["emoji"]}</div>'
            f'<div class="poster-overlay">'
            f'<div class="poster-title">{row["title"]}</div>'
            f'<div class="poster-meta">{row["genre"]} • ⭐ {row["rating"]}</div>'
            f'</div></div>'
        )
        st.markdown(poster_html, unsafe_allow_html=True)
        st.markdown(f'<span class="price-tag">₹{row["price"]}</span>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🛒 Add", key=f"{key_prefix}_cart_{row['movie_id']}", use_container_width=True):
                add_to_cart(row["movie_id"])
        with c2:
            liked = row["movie_id"] in st.session_state.wishlist
            label = "❤️ Saved" if liked else "🤍 Wishlist"
            if st.button(label, key=f"{key_prefix}_wish_{row['movie_id']}", use_container_width=True):
                toggle_wishlist(row["movie_id"])
        if st.button("View details", key=f"{key_prefix}_view_{row['movie_id']}", use_container_width=True):
            view_movie(row["movie_id"])
            st.rerun()

def movie_grid(df, key_prefix="grid", cols=5, match_scores=None):
    rows = list(df.iterrows())
    for i in range(0, len(rows), cols):
        chunk = rows[i:i + cols]
        columns = st.columns(cols)
        for col, (_, row) in zip(columns, chunk):
            with col:
                pct = match_scores.get(int(row["movie_id"])) if match_scores else None
                movie_card(row, key_prefix=f"{key_prefix}_{i}", match_pct=pct)

# ----------------------------------------------------------------------
# PAGE: BROWSE
# ----------------------------------------------------------------------
if st.session_state.page == "Browse":
    st.markdown(
        '<div class="hero-banner">'
        '<div class="hero-title">🎬 <b>MovieMart</b></div>'
        '<div class="hero-sub">Stream-worthy picks, smart recommendations, one click away.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not search_query and selected_genre == "All":
        st.markdown("#### 🔥 Trending Now")
        trending = movies_df.sort_values("rating", ascending=False).head(5)
        movie_grid(trending, key_prefix="trending", cols=5)

        seed_ids = list(st.session_state.wishlist) + list(st.session_state.cart.keys())
        seed_ids = list(dict.fromkeys(seed_ids))  # dedupe, keep order
        if seed_ids:
            st.markdown("---")
            st.markdown("#### 🎯 Recommended For You")
            st.caption("Based on the movies in your cart & wishlist — powered by the same content-based engine.")
            personal_recs, personal_scores = get_recommendations_for_user(
                seed_ids, movies_df, similarity_matrix, top_n=5
            )
            if personal_recs is not None and not personal_recs.empty:
                movie_grid(personal_recs, key_prefix="foryou", cols=5, match_scores=personal_scores)
        else:
            st.markdown("---")
            st.info("🎯 Add a movie to your Cart or Wishlist and personalized recommendations will show up here!")

        st.markdown("---")
        st.markdown("#### All Movies")

    filtered = movies_df.copy()
    if selected_genre != "All":
        filtered = filtered[filtered["genre"] == selected_genre]
    if search_query:
        filtered = filtered[filtered["title"].str.contains(search_query, case=False)]

    if filtered.empty:
        st.warning("No movies match your search/filter.")
    else:
        movie_grid(filtered, key_prefix="browse")

# ----------------------------------------------------------------------
# PAGE: DETAIL
# ----------------------------------------------------------------------
elif st.session_state.page == "Detail":
    movie_id = st.session_state.selected_movie
    row = movies_df[movies_df["movie_id"] == movie_id].iloc[0]

    if st.button("← Back to browse"):
        st.session_state.page = "Browse"
        st.rerun()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(row["poster_url"], use_container_width=True)
    with col2:
        st.title(row["title"])
        st.caption(f"{row['genre']} • ⭐ {row['rating']} • ₹{row['price']}")
        st.write(row["overview"])
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🛒 Add to Cart", use_container_width=True):
                add_to_cart(row["movie_id"])
        with c2:
            liked = row["movie_id"] in st.session_state.wishlist
            label = "❤️ In Wishlist" if liked else "🤍 Add to Wishlist"
            if st.button(label, use_container_width=True):
                toggle_wishlist(row["movie_id"])

    st.markdown("---")
    st.subheader(f"🎯 Because you viewed *{row['title']}* — you may also like")
    st.caption("Powered by a content-based recommendation engine (TF-IDF + cosine similarity on genre & plot).")
    recs, rec_scores = get_recommendations(row["movie_id"], movies_df, similarity_matrix, top_n=5)
    movie_grid(recs, key_prefix=f"rec_{movie_id}", cols=5, match_scores=rec_scores)

    # Second-level recommendations: related-to-the-related, so the chain
    # of "similar to similar" is visible without another click.
    if not recs.empty:
        st.markdown("---")
        anchor_id = int(recs.iloc[0]["movie_id"])
        anchor_title = recs.iloc[0]["title"]
        st.subheader(f"More like *{anchor_title}* (related to your top recommendation)")
        deeper_recs, deeper_scores = get_recommendations(anchor_id, movies_df, similarity_matrix, top_n=5)
        # exclude the movie the user is currently viewing to avoid a loop
        deeper_recs = deeper_recs[deeper_recs["movie_id"] != movie_id]
        movie_grid(deeper_recs, key_prefix=f"rec2_{movie_id}", cols=5, match_scores=deeper_scores)

# ----------------------------------------------------------------------
# PAGE: CART
# ----------------------------------------------------------------------
elif st.session_state.page == "Cart":
    st.title("🛒 Your Cart")
    if not st.session_state.cart:
        st.info("Your cart is empty. Go browse some movies!")
    else:
        total = 0
        for movie_id, qty in list(st.session_state.cart.items()):
            row = movies_df[movies_df["movie_id"] == movie_id].iloc[0]
            c1, c2, c3, c4 = st.columns([1, 3, 2, 2])
            with c1:
                st.image(row["poster_url"], width=80)
            with c2:
                st.markdown(f"**{row['title']}**")
                st.caption(row["genre"])
            with c3:
                st.write(f"Qty: {qty}")
                st.write(f"₹{row['price'] * qty}")
            with c4:
                if st.button("Remove", key=f"remove_{movie_id}"):
                    remove_from_cart(movie_id)
                    st.rerun()
            total += row["price"] * qty
            st.markdown("---")
        st.subheader(f"Total: ₹{total}")
        st.button("Proceed to Checkout", type="primary")

# ----------------------------------------------------------------------
# PAGE: WISHLIST
# ----------------------------------------------------------------------
elif st.session_state.page == "Wishlist":
    st.title("❤️ Your Wishlist")
    if not st.session_state.wishlist:
        st.info("No movies saved yet. Tap the heart on any movie to save it here.")
    else:
        wishlist_df = movies_df[movies_df["movie_id"].isin(st.session_state.wishlist)]
        movie_grid(wishlist_df, key_prefix="wishlist_page")