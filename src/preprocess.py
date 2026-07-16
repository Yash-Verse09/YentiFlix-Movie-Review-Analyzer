# =============================================================================
# preprocess.py — YentiFlix Sentiment Analysis
# Purpose : Load raw IMDB reviews, clean the text, and save processed data
# Input   : data/raw/reviews.csv
# Output  : data/processed/cleaned_reviews.csv
# Run     : python src/preprocess.py
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────

import os           # For building file paths that work on Windows, Mac, Linux
import re           # Regular expressions — used to find and remove patterns in text
import string       # Provides a ready-made list of all punctuation characters

import pandas as pd # Pandas — loads and manipulates the CSV dataset as a DataFrame

import nltk                                      # NLTK — Natural Language Toolkit
from nltk.corpus import stopwords               # Pre-built list of common English words to remove
from nltk.stem import WordNetLemmatizer         # Reduces words to their dictionary base form

# ── NLTK Data Downloads ───────────────────────────────────────────────────────
# NLTK ships as a library but its language data (stopwords, lemma dictionary)
# must be downloaded separately. These calls are safe to run multiple times —
# NLTK skips the download if the data is already present on your machine.

nltk.download("stopwords", quiet=True)   # Downloads the English stopword list
nltk.download("wordnet", quiet=True)     # Downloads the WordNet lemmatization database
nltk.download("omw-1.4", quiet=True)    # Open Multilingual WordNet — required by WordNetLemmatizer

# ── Constants ─────────────────────────────────────────────────────────────────
# Keeping file paths as constants at the top makes them easy to update later
# without hunting through the code.

# Build paths relative to this file's location so the script works regardless
# of which directory you run it from.
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH        = os.path.join(BASE_DIR, "data", "raw", "reviews.csv")
PROCESSED_PATH  = os.path.join(BASE_DIR, "data", "processed", "cleaned_reviews.csv")

# ── Initialise shared objects once (not inside the loop) ─────────────────────
# Creating these objects is slightly expensive. By initialising them once here
# and reusing them across all 50K rows we avoid unnecessary overhead.

STOP_WORDS  = set(stopwords.words("english"))  # Convert to a set for O(1) lookup speed
LEMMATIZER  = WordNetLemmatizer()              # Single lemmatizer instance reused for every review


# =============================================================================
# STEP 1 — Load the dataset
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Read the raw CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Absolute path to reviews.csv

    Returns
    -------
    pd.DataFrame with columns: review, sentiment
    """
    print(f"[1/6] Loading dataset from: {path}")

    # read_csv parses the file and returns a DataFrame.
    # encoding="utf-8" prevents errors from special characters in reviews.
    df = pd.read_csv(path, encoding="utf-8")

    print(f"      Loaded {len(df):,} rows and {len(df.columns)} columns.")
    print(f"      Columns found: {list(df.columns)}")

    # Validate that the expected columns exist before proceeding.
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError(
            "Expected columns 'review' and 'sentiment' not found. "
            "Please check your CSV file."
        )

    return df


# =============================================================================
# STEP 2 — Remove HTML tags
# =============================================================================

def remove_html(text: str) -> str:
    """
    Strip HTML tags such as <br />, <p>, <b> from a review string.

    The IMDB dataset was scraped from the web, so many reviews contain raw
    HTML — most commonly <br /> line breaks inside the review body.

    re.sub(pattern, replacement, string):
        - "<[^>]+>"  matches any substring that starts with < and ends with >
          with no > character in between — i.e. any HTML tag.
        - " "        replaces the tag with a single space so words don't merge.
    """
    return re.sub(r"<[^>]+>", " ", text)


# =============================================================================
# STEP 3 — Convert to lowercase
# =============================================================================

def to_lowercase(text: str) -> str:
    """
    Convert all characters to lowercase.

    Ensures that 'Excellent', 'excellent', and 'EXCELLENT' are all treated
    as the same word by TF-IDF and the classifier.
    """
    return text.lower()


# =============================================================================
# STEP 4 — Remove punctuation
# =============================================================================

def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation characters from the text.

    string.punctuation is a built-in Python string:
        !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~

    str.maketrans("", "", chars_to_remove) creates a translation table that
    maps each punctuation character to None (i.e. deletion).
    text.translate(table) applies that deletion to the entire string efficiently.
    """
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


# =============================================================================
# STEP 5 — Remove stopwords
# =============================================================================

def remove_stopwords(text: str) -> str:
    """
    Remove common English words that carry no sentiment signal.

    Words like 'the', 'is', 'and', 'a' appear in nearly every review and
    add noise without adding meaning. Removing them reduces feature space
    and improves model accuracy.

    The text is split into individual words (tokens), each token is checked
    against the STOP_WORDS set, and only non-stopwords are kept.
    The remaining tokens are joined back into a single string.

    Using a set (not a list) for STOP_WORDS makes each lookup O(1) instead
    of O(n) — important when processing 50K reviews.
    """
    tokens = text.split()                                      # Split on whitespace into word list
    tokens = [word for word in tokens if word not in STOP_WORDS]  # Keep only non-stopwords
    return " ".join(tokens)                                    # Re-join into a single string


# =============================================================================
# STEP 6 — Lemmatize
# =============================================================================

def lemmatize_text(text: str) -> str:
    """
    Reduce each word to its base dictionary form (lemma).

    Examples:
        'running' → 'run'
        'movies'  → 'movie'
        'better'  → 'good'  (with POS tagging; defaults to 'better' as noun)

    Lemmatization is preferred over stemming for production NLP because it
    produces real words (stemming can produce non-words like 'happi').

    The lemmatizer is called once per token; each token is individually
    processed and the results are joined back into a string.
    """
    tokens  = text.split()
    tokens  = [LEMMATIZER.lemmatize(word) for word in tokens]
    return " ".join(tokens)


# =============================================================================
# STEP 7 — Master cleaning function (chains all steps together)
# =============================================================================

def clean_text(text: str) -> str:
    """
    Apply the full cleaning pipeline to a single review string.

    Order matters:
        1. Remove HTML first — otherwise tags pollute lowercasing and tokenization.
        2. Lowercase — normalise before removing punctuation or stopwords.
        3. Remove punctuation — clean up symbols before tokenizing.
        4. Remove stopwords — work on clean tokens.
        5. Lemmatize — final vocabulary normalisation.

    Parameters
    ----------
    text : str
        A raw review string from the dataset.

    Returns
    -------
    str — fully cleaned review text, ready for TF-IDF vectorization.
    """
    text = remove_html(text)
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


# =============================================================================
# STEP 8 — Apply cleaning to the whole DataFrame and save
# =============================================================================

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text() to every row in the 'review' column.

    pandas .apply() passes each cell value to the provided function and
    collects the results into a new Series. This is significantly faster
    than a Python for-loop over 50K rows.

    A progress counter prints every 10,000 rows so you can confirm the
    script is running and estimate remaining time.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with 'review' and 'sentiment' columns.

    Returns
    -------
    pd.DataFrame with an added 'cleaned_review' column.
    """
    print("[2/6] Removing HTML tags ...")
    print("[3/6] Converting to lowercase ...")
    print("[4/6] Removing punctuation ...")
    print("[5/6] Removing stopwords ...")
    print("[6/6] Applying lemmatization ...")
    print("      Processing 50,000 rows — this may take 1–3 minutes ...")

    # Apply the full pipeline to every review cell.
    # Each call to clean_text() handles one review string.
    df["cleaned_review"] = df["review"].apply(clean_text)

    # Drop any rows where cleaning produced an empty string
    # (e.g. a review that was nothing but HTML tags and stopwords).
    before = len(df)
    df = df[df["cleaned_review"].str.strip() != ""]
    after  = len(df)

    if before != after:
        print(f"      Dropped {before - after} empty rows after cleaning.")

    print(f"      Cleaning complete. {after:,} rows ready.")
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """
    Save the cleaned DataFrame to a CSV file.

    Only 'cleaned_review' and 'sentiment' columns are saved — the raw
    'review' column is intentionally dropped to keep the file smaller.

    index=False prevents pandas from writing the row numbers as a column,
    which would just add noise to the training data.

    Parameters
    ----------
    df   : pd.DataFrame — cleaned DataFrame
    path : str          — destination file path
    """
    # Create the output directory if it doesn't already exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save only the two columns needed downstream
    df[["cleaned_review", "sentiment"]].to_csv(path, index=False, encoding="utf-8")

    print(f"\n✅ Cleaned dataset saved to: {path}")
    print(f"   Rows saved : {len(df):,}")
    print(f"   Columns    : cleaned_review, sentiment")


# =============================================================================
# Entry point
# =============================================================================

def main():
    """
    Orchestrates the full preprocessing pipeline in order.
    Called when the script is run directly: python src/preprocess.py
    """
    print("=" * 60)
    print("  YentiFlix — Text Preprocessing Pipeline")
    print("=" * 60)

    # Load → Clean → Save
    df = load_data(RAW_PATH)
    df = preprocess_dataset(df)
    save_data(df, PROCESSED_PATH)

    print("\nPreview of cleaned data (first 3 rows):")
    print("-" * 60)
    # Print a truncated preview so the terminal doesn't flood with text
    for i, row in df.head(3).iterrows():
        print(f"Row {i} | Sentiment : {row['sentiment']}")
        print(f"        Review    : {row['cleaned_review'][:120]}...")
        print()

    print("=" * 60)
    print("  Preprocessing complete. Run train_model.py next.")
    print("=" * 60)


# This block ensures main() only runs when this file is executed directly,
# not when it is imported as a module by another script (e.g. app.py).
if __name__ == "__main__":
    main()
