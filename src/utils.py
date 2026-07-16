# =============================================================================
# utils.py — YentiFlix Sentiment Analysis
# Purpose : Shared constants and helper functions used across the project.
#           Import from this file in predict.py, app.py, or any other module
#           that needs file paths or text cleaning without duplicating code.
# Usage   : from src.utils import clean_text, MODEL_PATH, VECTORIZER_PATH
# =============================================================================

# ── Imports ───────────────────────────────────────────────────────────────────

import os       # Builds file paths that work on Windows, Mac, and Linux
import re       # Regular expressions — used to detect and remove HTML tags
import string   # Provides string.punctuation: all 32 punctuation characters

import nltk                                  # Natural Language Toolkit
from nltk.corpus import stopwords           # List of common English words to remove
from nltk.stem import WordNetLemmatizer     # Reduces words to their dictionary base form

# ── NLTK Data Downloads ───────────────────────────────────────────────────────
# NLTK language data must be downloaded separately from the library itself.
# quiet=True suppresses the "already downloaded" message on repeated runs.
# These four packages cover everything needed for cleaning:
#   stopwords  — the English stopword word list (~180 common words)
#   wordnet    — the dictionary used by the lemmatizer
#   omw-1.4   — Open Multilingual WordNet, required by WordNetLemmatizer
#   punkt      — sentence/word tokenizer data (useful if tokenization is added later)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)
nltk.download("punkt",     quiet=True)


# =============================================================================
# FILE PATH CONSTANTS
# =============================================================================
# Centralising all paths here means:
#   - You only change a path in ONE place if the folder structure changes
#   - Every file that imports from utils.py automatically gets the update
#   - No risk of predict.py and app.py pointing to slightly different paths
#
# os.path.abspath(__file__)           → full path to this file (src/utils.py)
# os.path.dirname(...)                → directory containing this file (src/)
# os.path.dirname(... dirname ...)    → parent of src/ → YentiFlix/ root
# os.path.join(BASE_DIR, ...)         → safely builds the full path for each file

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the cleaned training dataset (output of preprocess.py)
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_reviews.csv")

# Path to the saved Logistic Regression model (output of train_model.py)
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")

# Path to the saved TF-IDF vectorizer (output of train_model.py)
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

# Path to the original raw dataset (never modified)
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "reviews.csv")

# Path to the charts folder (used by visualize.py and served by Flask)
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")


# =============================================================================
# SHARED OBJECTS — initialised once at module load
# =============================================================================
# Both objects below are slightly expensive to create:
#   - STOP_WORDS  loads and parses the NLTK stopword list from disk
#   - LEMMATIZER  loads the WordNet database
#
# By creating them here at module level (not inside a function), they are
# initialised ONCE when utils.py is first imported and then reused on every
# subsequent call to clean_text() — across all 50K training rows or all
# incoming Flask requests.
#
# Using a set() for STOP_WORDS gives O(1) lookup time per word instead of
# O(n) linear scan through a list — a meaningful speed difference at scale.

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# =============================================================================
# TEXT CLEANING HELPERS
# Each step is its own small function so it can be:
#   - Tested individually in a notebook
#   - Replaced or adjusted without touching the others
#   - Understood clearly by reading its name alone
# =============================================================================

def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags such as <br />, <p>, <b> from a review string.

    IMDB reviews were scraped from the web and often contain raw HTML,
    most commonly <br /> line-break tags inside the review body.

    re.sub(pattern, replacement, string):
        "<[^>]+"  — matches any substring starting with < and ending with >
                    with no > character in between → any HTML tag
        " "       — replaces the tag with a space so words don't merge
    """
    return re.sub(r"<[^>]+>", " ", text)


def convert_to_lowercase(text: str) -> str:
    """
    Convert all characters in the string to lowercase.

    Ensures 'Brilliant', 'brilliant', and 'BRILLIANT' are all treated as
    the same token by the TF-IDF vectorizer and classifier.
    """
    return text.lower()


def remove_punctuation(text: str) -> str:
    """
    Strip all punctuation characters from the text.

    string.punctuation contains: !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~

    str.maketrans("", "", chars) builds a translation table that maps every
    punctuation character to None (deletion).
    text.translate(table) applies the deletion efficiently in one pass.
    """
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


def remove_stopwords(text: str) -> str:
    """
    Remove common English words that carry no sentiment signal.

    Words like 'the', 'is', 'and', 'a' appear in almost every review.
    They inflate the feature space without helping the model distinguish
    positive from negative sentiment.

    The text is split into individual word tokens, each token is checked
    against the STOP_WORDS set, and only non-stopwords are kept.
    """
    tokens = text.split()                                          # Split on whitespace
    tokens = [word for word in tokens if word not in STOP_WORDS]  # Drop stopwords
    return " ".join(tokens)                                        # Rejoin into string


def lemmatize(text: str) -> str:
    """
    Reduce each word to its base dictionary form (lemma).

    Examples:
        'running'  → 'run'
        'movies'   → 'movie'
        'greatest' → 'greatest'  (adjectives default to noun form)

    Lemmatization is preferred over stemming because it produces real
    dictionary words — stemming can produce non-words like 'happi'.

    Each token is passed individually to the lemmatizer, then rejoined.
    """
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens]
    return " ".join(tokens)


# =============================================================================
# MAIN CLEANING FUNCTION — chains all steps in the correct order
# =============================================================================

def clean_text(text: str) -> str:
    """
    Apply the full text-cleaning pipeline to a single review string.

    This is the PRIMARY function to import in other modules:
        from src.utils import clean_text

    Order matters — each step prepares the text for the next:
        1. Remove HTML   → prevents tags from polluting tokens
        2. Lowercase     → normalise before any word comparison
        3. Punctuation   → remove symbols before tokenizing
        4. Stopwords     → filter on clean, lowercase tokens
        5. Lemmatize     → final vocabulary normalisation

    Parameters
    ----------
    text : str
        A raw or partially cleaned review string.
        Can be a single sentence or a full multi-paragraph review.

    Returns
    -------
    str
        Fully cleaned review text, ready to be passed to the TF-IDF
        vectorizer for training or prediction.

    Example
    -------
        >>> clean_text("The film was AMAZING!! <br/> Loved every moment.")
        'film amazing loved every moment'
    """
    # Guard: if the input is not a string (e.g. NaN from a CSV), return empty
    if not isinstance(text, str):
        return ""

    text = remove_html_tags(text)       # Step 1 — strip <br />, <p> etc.
    text = convert_to_lowercase(text)   # Step 2 — 'Amazing' → 'amazing'
    text = remove_punctuation(text)     # Step 3 — remove ! . , ? etc.
    text = remove_stopwords(text)       # Step 4 — remove 'the', 'is', 'a' etc.
    text = lemmatize(text)              # Step 5 — 'movies' → 'movie'
    return text


# =============================================================================
# LABEL FORMATTING HELPER
# =============================================================================

def format_label(label: str) -> dict:
    if str(label).lower() == "positive":
        return {
            "label": "positive",
            "sentiment": "positive",
            "display": "Positive",
            "emoji": "😊",
            "css": "positive",
            "message": "This review expresses a GOOD opinion."
        }

    return {
        "label": "negative",
        "sentiment": "negative",
        "display": "Negative",
        "emoji": "😡",
        "css": "negative",
        "message": "This review expresses a BAD opinion."
    }
# =============================================================================
# VALIDATION HELPER
# =============================================================================

def is_valid_review(text: str) -> tuple:
    """
    Check whether a user-supplied review is suitable for prediction.

    Returns a tuple of (is_valid: bool, error_message: str).
    The error_message is empty string when the review is valid.

    Checks performed:
        1. Input is a non-empty string
        2. At least 10 characters long (prevents one-word inputs)
        3. After cleaning, at least 2 meaningful words remain
           (guards against inputs that are entirely HTML or punctuation)

    Used by both predict.py and app.py to validate input before sending
    it to the model — consistent validation in one place.

    Parameters
    ----------
    text : str — raw user input

    Returns
    -------
    (True,  "")            — review is valid, proceed with prediction
    (False, "error msg")   — review is invalid, show error to user
    """
    # Check 1 — must be a non-empty string
    if not isinstance(text, str) or len(text.strip()) == 0:
        return False, "Please enter a review before submitting."

    # Check 2 — minimum character length
    if len(text.strip()) < 10:
        return False, "Review is too short. Please write at least a sentence."

    # Check 3 — must have meaningful words after cleaning
    cleaned = clean_text(text)
    if len(cleaned.split()) < 2:
        return False, "Review contains no meaningful words after cleaning. Please try again."

    return True, ""


# =============================================================================
# Quick self-test — runs only when this file is executed directly
# =============================================================================
# Running  python src/utils.py  will execute this block and print a quick
# sanity check. It does NOT run when utils.py is imported by another module.

if __name__ == "__main__":
    print("=" * 55)
    print("  YentiFlix — utils.py self-test")
    print("=" * 55)

    # Test clean_text
    sample = 'The film was ABSOLUTELY amazing!! <br/> One of the best movies I have ever seen.'
    cleaned = clean_text(sample)
    print(f"\nOriginal : {sample}")
    print(f"Cleaned  : {cleaned}")

    # Test format_label
    print(f"\nPositive label: {format_label('positive')}")
    print(f"Negative label: {format_label('negative')}")

    # Test is_valid_review
    tests = [
        "Great movie!",
        "",
        "ok",
        "This was an absolutely terrible film. I hated every second of it.",
        "<br/><br/>!!!???",
    ]
    print("\nValidation tests:")
    for t in tests:
        valid, msg = is_valid_review(t)
        status = "✅ Valid" if valid else f"❌ Invalid — {msg}"
        print(f"  '{t[:45]}' → {status}")

    # Print resolved file paths
    print("\nResolved file paths:")
    print(f"  DATA_PATH       : {DATA_PATH}")
    print(f"  MODEL_PATH      : {MODEL_PATH}")
    print(f"  VECTORIZER_PATH : {VECTORIZER_PATH}")
    print(f"  RAW_DATA_PATH   : {RAW_DATA_PATH}")
    print(f"  CHARTS_DIR      : {CHARTS_DIR}")
    print("\n" + "=" * 55)
