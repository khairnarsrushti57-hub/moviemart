# 🎬 MovieMart — Movie Ecommerce Website with Recommendation Engine

A simple ecommerce-style website for movies, built with **Streamlit**.
Users can browse movies, add them to a cart or wishlist, and get
**personalized recommendations** powered by a content-based
recommendation engine (TF-IDF + Cosine Similarity).

## ✨ Features

- Browse movies with poster images, ratings, and prices
- Add to Cart / Add to Wishlist
- Search + filter by genre
- **Content-based recommendation engine** (TF-IDF + Cosine Similarity
  on genre & plot) — shows "Because you viewed X, you may also like"
  and cascades to "More like [top recommendation]"
- **Personalized "Recommended For You"** on the homepage, based on
  everything in your cart & wishlist
- Cart and Wishlist pages with totals

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **Recommendation Engine:** scikit-learn (TfidfVectorizer + cosine_similarity)
- **Data handling:** Pandas
- **Dataset:** movies.csv — 47 movies across 8 genres

## 🚀 How to Run

\`\`\`
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## 🤖 How the Recommendation Engine Works

Each movie's **genre** (weighted 3x) and **overview** text are combined
and converted into TF-IDF vectors. When you view a movie or add
something to your cart/wishlist, the app computes cosine similarity
between movies and shows the closest matches, along with a **match %**
score.