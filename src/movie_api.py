
# =============================================================================
# movie_api.py — YentiFlix
# Location: YentiFlix/movie_api.py  (root folder, same as app.py)
# =============================================================================

import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("movie_api")

# =============================================================================
# ★★★  APNI API KEYS YAHAN LIKHO  ★★★
# =============================================================================

TMDB_API_KEY = os.getenv("TMDB_API_KEY")   # <<< TMDB KEY YAHAN
OMDB_API_KEY = os.getenv("OMDB_API_KEY")  # <<< OMDB KEY YAHAN

# =============================================================================
# Keys kahan se milegi (DONO FREE HAIN):
#   TMDb : https://www.themoviedb.org/settings/api
#   OMDb : https://www.omdbapi.com/apikey.aspx
# =============================================================================

TMDB_BASE_URL   = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_BACKDROP   = "https://image.tmdb.org/t/p/w1280"
OMDB_BASE_URL   = "https://www.omdbapi.com/"

REQUEST_TIMEOUT  = 30
MAX_CAST_MEMBERS = 8

GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
}


def _keys_valid():
    """Check karo ki keys set hain ya placeholder hain."""
    tmdb_ok = bool(TMDB_API_KEY) and "YAHAN" not in TMDB_API_KEY
    omdb_ok = bool(OMDB_API_KEY) and "YAHAN" not in OMDB_API_KEY
    return tmdb_ok, omdb_ok


def search_movie(movie_name: str) -> dict:
    """TMDb se movie search karo."""
    tmdb_ok, _ = _keys_valid()
    if not tmdb_ok:
        logger.warning("TMDB_API_KEY set nahi hai.")
        return {}
    if not movie_name or not movie_name.strip():
        return {}

    try:
        resp = requests.get(
            f"{TMDB_BASE_URL}/search/movie",
            params={
                "api_key": TMDB_API_KEY,
                "query": movie_name.strip(),
                "language": "en-US",
                "page": 1,
                "include_adult": False,
            },
            timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return {}

        m = results[0]
        pp = m.get("poster_path")
        bp = m.get("backdrop_path")

        return {
            "movie_id"    : m.get("id"),
            "title"       : m.get("title", movie_name),
            "release_date": m.get("release_date", "N/A"),
            "rating"      : round(float(m.get("vote_average", 0)), 1),
            "genre"       : ", ".join([GENRE_MAP.get(g, "") for g in m.get("genre_ids", []) if GENRE_MAP.get(g)]) or "N/A",
            "poster_url"  : f"{TMDB_IMAGE_BASE}{pp}" if pp else "",
            "backdrop_url": f"{TMDB_BACKDROP}{bp}"   if bp else "",
            "overview"    : m.get("overview", "N/A"),
            "vote_count"  : m.get("vote_count", 0),
        }
    except Exception as e:
        logger.error(f"TMDb search error: {e}")
        return {}


def get_movie_cast(movie_id: int) -> list:
    """Top 5 cast members fetch karo."""
    tmdb_ok, _ = _keys_valid()
    if not tmdb_ok or not movie_id:
        return []
    try:
        resp = requests.get(
            f"{TMDB_BASE_URL}/movie/{movie_id}/credits",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        cast_list = resp.json().get("cast", [])
        return [
            {
                "name"       : c.get("name", "Unknown"),
                "character"  : c.get("character", ""),
                "profile_url": f"{TMDB_IMAGE_BASE}{c['profile_path']}" if c.get("profile_path") else "",
            }
            for c in cast_list[:MAX_CAST_MEMBERS]
        ]
    except Exception as e:
        logger.error(f"TMDb cast error: {e}")
        return []


def get_movie_trailer(movie_id: int) -> str:
    """YouTube trailer URL fetch karo."""
    tmdb_ok, _ = _keys_valid()
    if not tmdb_ok or not movie_id:
        return ""
    try:
        resp = requests.get(
            f"{TMDB_BASE_URL}/movie/{movie_id}/videos",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        videos   = resp.json().get("results", [])
        trailers = [v for v in videos if v.get("site") == "YouTube" and v.get("type") == "Trailer"]
        if not trailers:
            trailers = [v for v in videos if v.get("site") == "YouTube"]
        if not trailers:
            return ""
        best = next((v for v in trailers if "official" in v.get("name", "").lower()), trailers[0])
        key  = best.get("key", "")
        return f"https://www.youtube.com/watch?v={key}" if key else ""
    except Exception as e:
        logger.error(f"TMDb trailer error: {e}")
        return ""


def get_omdb_details(movie_name: str) -> dict:
    """OMDb se extra details fetch karo."""
    empty = {"imdb_rating":"N/A","imdb_votes":"N/A","box_office":"N/A",
             "awards":"N/A","runtime":"N/A","director":"N/A","metascore":"N/A"}
    _, omdb_ok = _keys_valid()
    if not omdb_ok or not movie_name:
        return empty
    try:
        resp = requests.get(
            OMDB_BASE_URL,
            params={"apikey": OMDB_API_KEY, "t": movie_name.strip(), "type": "movie", "r": "json"},
            timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        d = resp.json()
        if d.get("Response") == "False":
            return empty
        def c(v): return v if v and v != "N/A" else "N/A"
        return {
            "imdb_rating": c(d.get("imdbRating")),
            "imdb_votes" : c(d.get("imdbVotes")),
            "box_office" : c(d.get("BoxOffice")),
            "awards"     : c(d.get("Awards")),
            "runtime"    : c(d.get("Runtime")),
            "director"   : c(d.get("Director")),
            "metascore"  : c(d.get("Metascore")),
        }
    except Exception as e:
        logger.error(f"OMDb error: {e}")
        return empty


def calculate_movie_status(rating) -> dict:
    """Rating se movie status calculate karo."""
    try:
        r = float(rating) if rating not in (None, "", "N/A") else 0.0
    except (ValueError, TypeError):
        r = 0.0

    if r >= 8.5:
        return {"status_name":"Blockbuster","badge_color":"blockbuster"}
    elif r >= 7.5:
        return {"status_name":"Super Hit",  "badge_color":"superhit"}
    elif r >= 6.5:
        return {"status_name":"Hit",        "badge_color":"hit"}
    elif r >= 5.5:
        return {"status_name":"Average",    "badge_color":"average"}
    else:
        return {"status_name":"Flop",       "badge_color":"flop"}


def get_complete_movie_data(movie_name: str) -> dict:
    """
    Ek call mein TMDb + OMDb dono se pura data fetch karo.
    Returns a complete dict — never raises exception.
    """
    logger.info(f"Fetching data for: '{movie_name}'")

    result = {
        "title":"N/A","release_date":"N/A","rating":"N/A","vote_count":0,
        "status":"N/A","status_color":"na","poster":"","backdrop":"",
        "overview":"N/A","genre":"N/A","cast":[],"trailer":"",
        "imdb_rating":"N/A","imdb_votes":"N/A","box_office":"N/A",
        "awards":"N/A","runtime":"N/A","director":"N/A","metascore":"N/A",
    }

    # Step 1 — TMDb search
    tmdb = search_movie(movie_name)
    movie_id = None
    if tmdb:
        movie_id              = tmdb.get("movie_id")
        result["title"]       = tmdb.get("title",        movie_name)
        result["release_date"]= tmdb.get("release_date", "N/A")
        result["rating"]      = tmdb.get("rating",       "N/A")
        result["vote_count"]  = tmdb.get("vote_count",   0)
        result["poster"]      = tmdb.get("poster_url",   "")
        result["backdrop"]    = tmdb.get("backdrop_url", "")
        result["overview"]    = tmdb.get("overview",     "N/A")
        result["genre"]       = tmdb.get("genre",        "N/A")
        logger.info(f"TMDb: '{result['title']}' id={movie_id} rating={result['rating']}")

    # Step 2 — Cast
    if movie_id:
        result["cast"] = get_movie_cast(movie_id)
        logger.info(f"Cast: {len(result['cast'])} members")

    # Step 3 — Trailer
    if movie_id:
        result["trailer"] = get_movie_trailer(movie_id)
        logger.info(f"Trailer: {result['trailer'] or 'not found'}")

    # Step 4 — OMDb
    omdb = get_omdb_details(result["title"])
    result.update({k: omdb[k] for k in omdb})
    logger.info(f"OMDb: imdb={result['imdb_rating']} box={result['box_office']}")

    # Step 5 — Fallback rating
    if result["rating"] in (None, "N/A", 0, 0.0) and result["imdb_rating"] != "N/A":
        result["rating"] = result["imdb_rating"]

    # Step 6 — Status
    s = calculate_movie_status(result["rating"])
    result["status"]       = s["status_name"]
    result["status_color"] = s["badge_color"]

    # Step 7 — Clean None values
    for k, v in result.items():
        if v is None:
            result[k] = "N/A"

    logger.info(f"Done: status={result['status']}")
    return result


def check_api_keys() -> dict:
    tmdb_ok, omdb_ok = _keys_valid()
    if tmdb_ok and omdb_ok:
        msg = "✅ Dono keys set hain — TMDb aur OMDb ready."
    elif tmdb_ok:
        msg = "⚠️  Sirf TMDb key hai."
    elif omdb_ok:
        msg = "⚠️  Sirf OMDb key hai."
    else:
        msg = "❌ Koi key set nahi — line 21-22 mein keys daalo."
    return {"tmdb_ok": tmdb_ok, "omdb_ok": omdb_ok, "message": msg}


if __name__ == "__main__":
    import json
    print("=" * 60)
    print("  YentiFlix — movie_api.py test")
    print("=" * 60)
    keys = check_api_keys()
    print(f"\n{keys['message']}\n")
    if not keys["tmdb_ok"] and not keys["omdb_ok"]:
        print("Line 21-22 mein apni real keys daalo phir run karo.")
        exit(0)
    data = get_complete_movie_data("Inception")
    print(json.dumps(data, indent=2, ensure_ascii=False))

