# =============================================================================
# app.py — YentiFlix
# Location: YentiFlix/app.py
# Run: python app.py
# =============================================================================

import os
import sys
import joblib
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash
from database import init_db, save_review

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.utils import clean_text, MODEL_PATH, VECTORIZER_PATH, is_valid_review, format_label

# ── Movie API import — movie_api.py root mein hona chahiye ──────────────────
try:
    from src.movie_api import get_complete_movie_data, check_api_keys
    _s = check_api_keys()
    MOVIE_API_READY = _s["tmdb_ok"] or _s["omdb_ok"]
    print(f"[movie_api] {_s['message']}")
except ImportError as e:
    MOVIE_API_READY = False
    print(f"[movie_api] Import failed: {e}")
    print("[movie_api] movie_api.py root folder (YentiFlix/) mein hona chahiye")


# =============================================================================
# Flask App
# =============================================================================

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32))
    return app

app = create_app()

# =============================================================================
# Model Loading
# =============================================================================

model      = None
vectorizer = None

def load_model_artifacts():
    global model, vectorizer
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("  → Run: python src/train_model.py")
        return False
    if not os.path.exists(VECTORIZER_PATH):
        print(f"[ERROR] Vectorizer not found: {VECTORIZER_PATH}")
        print("  → Run: python src/train_model.py")
        return False
    try:
        model      = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("[OK] Model and vectorizer loaded.")
        return True
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return False

MODEL_LOADED = load_model_artifacts()
init_db()


# =============================================================================
# Prediction Helper
# =============================================================================

def run_prediction(raw_review: str) -> dict:
    cleaned       = clean_text(raw_review)
    review_vector = vectorizer.transform([cleaned])
    label         = model.predict(review_vector)[0]
    probs         = model.predict_proba(review_vector)[0]
    confidence    = f"{max(probs) * 100:.1f}%"
    result        = format_label(label)
    result["confidence"] = confidence
    result["review"]     = raw_review
    return result


# =============================================================================
# Context Processor
# =============================================================================

@app.context_processor
def inject_globals():
    from datetime import datetime
    return {"app_name": "YentiFlix 🎬", "year": datetime.now().year}


# =============================================================================
# Routes
# =============================================================================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_ready=MODEL_LOADED)


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        flash("Model available nahi hai. python src/train_model.py run karo.", "error")
        return redirect(url_for("index"))

    # ── Form se data padho ────────────────────────────────────────────────────
    raw_review   = request.form.get("review",      "").strip()
    movie_name   = request.form.get("movie_name",  "").strip()
    user_rating  = request.form.get("user_rating", "").strip()  # user ka rating (1-10)

    # ── Validate review ───────────────────────────────────────────────────────
    is_valid, error_msg = is_valid_review(raw_review)
    if not is_valid:
        flash(error_msg, "warning")
        return redirect(url_for("index"))

    # ── Sentiment prediction ──────────────────────────────────────────────────
    try:
        result = run_prediction(raw_review)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        flash("Prediction mein error aaya. Dobara koshish karo.", "error")
        return redirect(url_for("index"))

    # ── Movie API data ────────────────────────────────────────────────────────
    # Default values — agar API fail ho toh bhi page crash na kare
    movie_data = {
        "title"       : movie_name or "N/A",
        "release_date": "N/A",
        "rating"      : "N/A",
        "vote_count"  : 0,
        "status"      : "N/A",
        "status_color": "na",
        "poster"      : "",
        "backdrop"    : "",
        "overview"    : "N/A",
        "genre"       : "N/A",
        "cast"        : [],
        "trailer"     : "",
        "imdb_rating" : "N/A",
        "imdb_votes"  : "N/A",
        "box_office"  : "N/A",
        "awards"      : "N/A",
        "runtime"     : "N/A",
        "director"    : "N/A",
        "metascore"   : "N/A",
    }

    if MOVIE_API_READY and movie_name:
         try:
              fetched = get_complete_movie_data(movie_name)
              movie_data.update(fetched)
              print("=" * 50)
              print(movie_data["release_date"])
              print(len(movie_data["cast"]))
              print(movie_data["cast"])
              print("=" * 50)
               

              print(
                  f"[movie_api] '{movie_name}': "
                  f"status={movie_data['status']} "
                  f"rating={movie_data['rating']} "
                  f"cast={len(movie_data['cast'])}"
                  
                   
                 )

         except Exception as e:
             print(f"[movie_api] Error for '{movie_name}': {e}")
    else:
        if not MOVIE_API_READY:
            print("[movie_api] MOVIE_API_READY=False — keys check karo")
        if not movie_name:
            print("[movie_api] movie_name empty — user ne naam nahi diya")

    # ── User rating validate karo ─────────────────────────────────────────────
    try:
        user_rating_float = float(user_rating) if user_rating else None
        if user_rating_float is not None:
            user_rating_float = max(1.0, min(10.0, user_rating_float))
    except ValueError:
        user_rating_float = None
    
    # ── Save review to SQLite ─────────────────────────────────────────────────
    try:
        save_review(
            movie_name=movie_name,
            rating=user_rating_float,
            review=raw_review,
            sentiment=result["sentiment"],
            confidence=result["confidence"]
            )
        
    except Exception as e:
        print(f"[database] Save failed: {e}")

    # ── Render result page ────────────────────────────────────────────────────
    return render_template(
        "result.html",

        # Sentiment (ML model)
        sentiment    = result["sentiment"],
        display      = result["display"],
        emoji        = result["emoji"],
        css          = result["css"],
        message      = result["message"],
        confidence   = result["confidence"],
        review       = result["review"],

        # User input
        user_rating  = user_rating_float,

        # Movie data (TMDb + OMDb)
        movie_name   = movie_data.get("title",        movie_name or "N/A"),
        release_date = movie_data.get("release_date", "N/A"),
        tmdb_rating  = movie_data.get("rating",       "N/A"),
        vote_count   = movie_data.get("vote_count",   0),
        status       = movie_data.get("status",       "N/A"),
        status_color = movie_data.get("status_color", "na"),
        poster       = movie_data.get("poster",       ""),
        backdrop     = movie_data.get("backdrop",     ""),
        overview     = movie_data.get("overview",     "N/A"),
        genre        = movie_data.get("genre",        "N/A"),
        cast         = movie_data.get("cast",         []),
        trailer      = movie_data.get("trailer",      ""),
        imdb_rating  = movie_data.get("imdb_rating",  "N/A"),
        imdb_votes   = movie_data.get("imdb_votes",   "N/A"),
        box_office   = movie_data.get("box_office",   "N/A"),
        awards       = movie_data.get("awards",       "N/A"),
        runtime      = movie_data.get("runtime",      "N/A"),
        director     = movie_data.get("director",     "N/A"),
        metascore    = movie_data.get("metascore",    "N/A"),
    )


@app.route("/health", methods=["GET"])
def health():
    return {
        "status"         : "ready" if MODEL_LOADED else "model_not_loaded",
        "movie_api_ready": MOVIE_API_READY,
        "app"            : "YentiFlix",
    }, 200


@app.route("/about", methods=["GET"])
def about():
    return redirect(url_for("index"))


# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    flash("Page nahi mila.", "warning")
    return redirect(url_for("index"))

@app.errorhandler(500)
def server_error(e):
    print(f"[ERROR] 500: {e}")
    flash("Server error. Dobara koshish karo.", "error")
    return redirect(url_for("index"))


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🎬  YentiFlix — Starting")
    print("=" * 55)
    print(f"  URL        : http://127.0.0.1:5000")
    print(f"  Model      : {'✅ Loaded' if MODEL_LOADED else '❌ Run train_model.py'}")
    print(f"  Movie API  : {'✅ Ready'  if MOVIE_API_READY else '❌ Keys set nahi'}")
    print("=" * 55 + "\n")
    app.run(debug=True, host="127.0.0.1", port=5000)