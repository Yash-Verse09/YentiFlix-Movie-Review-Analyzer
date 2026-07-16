# =============================================================================
# predict.py — YentiFlix Sentiment Analysis
# Purpose : Load the saved model + vectorizer, accept a movie review from the
#           user, clean it, and predict whether it is Positive or Negative.
# Input   : User types a review in the terminal
# Output  : Positive 😊  or  Negative 😡
# Run     : python src/predict.py
# =============================================================================

# ── Imports ───────────────────────────────────────────────────────────────────

import os       # Builds file paths that work on Windows, Mac, and Linux

import joblib   # Loads the saved .pkl files (model and vectorizer) from disk
                # Must be the same library used to save them in train_model.py

# Import the clean_text function directly from preprocess.py.
# This guarantees the EXACT same cleaning steps are applied to new input
# as were applied to the training data — no inconsistency possible.
#
# sys.path trick explained:
#   When you run  python src/predict.py  from the YentiFlix/ root, Python
#   adds  src/  to its path. But if the working directory is different,
#   the import may fail. Adding the project root manually makes the import
#   reliable regardless of where you launch the script from.

import sys

# Navigate two levels up from this file (src/predict.py → src/ → YentiFlix/)
# and add that root directory to Python's module search path.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Now Python can find preprocess.py inside the src/ folder
from src.preprocess import clean_text   # The full cleaning pipeline (HTML → lowercase → stopwords → lemmatize)


# ── File paths ────────────────────────────────────────────────────────────────

MODEL_PATH      = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")


# =============================================================================
# STEP 1 — Load saved model and vectorizer
# =============================================================================

def load_artifacts():
    """
    Load the trained Logistic Regression model and fitted TF-IDF vectorizer
    from disk using joblib.

    These files were created by train_model.py. Loading them here means we
    reuse all the training work without repeating it — predictions are
    near-instantaneous.

    joblib.load(path) deserialises the .pkl file back into the original
    Python object exactly as it was when saved.

    Returns
    -------
    model      : trained LogisticRegression classifier
    vectorizer : fitted TfidfVectorizer with the learned vocabulary
    """
    # Check that both files actually exist before trying to load them.
    # Gives a helpful error message instead of a confusing FileNotFoundError.
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at:\n  {MODEL_PATH}\n"
            "Please run train_model.py first to generate the model."
        )

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Vectorizer file not found at:\n  {VECTORIZER_PATH}\n"
            "Please run train_model.py first to generate the vectorizer."
        )

    model      = joblib.load(MODEL_PATH)       # Load the classifier
    vectorizer = joblib.load(VECTORIZER_PATH)  # Load the TF-IDF vectorizer

    return model, vectorizer


# =============================================================================
# STEP 2 — Preprocess the user's input
# =============================================================================

def preprocess_input(raw_review: str) -> str:
    """
    Apply the same cleaning pipeline to user input that was applied during
    training (defined in preprocess.py).

    Why must preprocessing be identical?
        The TF-IDF vectorizer was fitted on cleaned text. If the user types
        "The film was AMAZING!!!" and we feed that raw text to the vectorizer,
        it looks for the token "AMAZING!!!" — which was never in the training
        vocabulary. The word effectively disappears. Cleaning first ensures
        "amazing" matches the trained vocabulary correctly.

    Parameters
    ----------
    raw_review : str — the review text exactly as typed by the user

    Returns
    -------
    str — cleaned review text, ready for vectorization
    """
    cleaned = clean_text(raw_review)
    return cleaned


# =============================================================================
# STEP 3 — Vectorize the cleaned text
# =============================================================================

def vectorize_input(cleaned_review: str, vectorizer):
    """
    Convert the cleaned review string into a TF-IDF numeric vector using the
    ALREADY-FITTED vectorizer from training.

    vectorizer.transform([text]) expects a LIST (or iterable) of strings,
    even if there is only one review. The result is a sparse matrix with
    shape (1, vocabulary_size).

    We do NOT call fit_transform here — that would rebuild the vocabulary
    from this single review, completely ignoring everything learned during
    training. We only call transform.
    """
    # Wrap the single string in a list — transform() requires an iterable
    review_vector = vectorizer.transform([cleaned_review])
    return review_vector


# =============================================================================
# STEP 4 — Predict sentiment
# =============================================================================

def predict_sentiment(review_vector, model) -> str:
    """
    Run the trained classifier on the vectorized review and return a label.

    model.predict(X) returns a NumPy array of predicted class labels.
    Since we passed one review, the array has one element — we take [0].

    The raw label is 'positive' or 'negative' (strings), exactly matching
    the values in the original sentiment column of the dataset.

    Returns
    -------
    str — 'positive' or 'negative'
    """
    prediction = model.predict(review_vector)  # Returns array e.g. ['positive']
    label       = prediction[0]                # Extract the single string label
    return label


# =============================================================================
# STEP 5 — Format and print the result
# =============================================================================

def display_result(label: str, raw_review: str) -> None:
    """
    Print the prediction result in a clear, human-readable format.

    model.predict_proba() returns the model's confidence score for each class.
    The output is an array like [[0.08, 0.92]] meaning:
        8%  probability → negative
        92% probability → positive
    max() gives us the higher of the two — the model's confidence in its prediction.
    Multiplying by 100 converts it to a percentage for display.
    """
    print("\n" + "=" * 50)
    print("  🎬  YentiFlix Sentiment Result")
    print("=" * 50)
    print(f"\n  Review  : {raw_review[:80]}{'...' if len(raw_review) > 80 else ''}")

    if label == "positive":
        print("\n  Result  : ✅  POSITIVE 😊")
        print("  Verdict : This review expresses a GOOD opinion.")
    else:
        print("\n  Result  : ❌  NEGATIVE 😡")
        print("  Verdict : This review expresses a BAD opinion.")

    print("\n" + "=" * 50)


# =============================================================================
# STEP 6 — Get user input from the console
# =============================================================================

def get_user_review() -> str:
    """
    Prompt the user to type a movie review and return the input string.

    Keeps prompting until the user types something non-empty — prevents
    the model from running on a blank input, which would return a meaningless
    prediction.
    """
    print("\n" + "=" * 50)
    print("  🎬  YentiFlix — Sentiment Predictor")
    print("=" * 50)
    print("\nType a movie review below and press Enter.")
    print("(Type 'quit' to exit)\n")

    while True:
        review = input("  Your review: ").strip()

        if review.lower() == "quit":
            print("\n  Goodbye! 👋\n")
            exit(0)

        if len(review) == 0:
            print("  ⚠️  Please enter a review before pressing Enter.\n")
            continue

        if len(review) < 5:
            print("  ⚠️  Review is too short. Please write at least a few words.\n")
            continue

        return review


# =============================================================================
# Entry point — runs the full prediction loop
# =============================================================================

def main():
    """
    Full prediction pipeline:
        Load artifacts → Get user input → Clean → Vectorize → Predict → Display

    Wrapped in a loop so the user can predict multiple reviews in one session
    without restarting the script (model loading only happens once).
    """

    # Load model and vectorizer once — reuse for every prediction in the loop
    print("\nLoading model and vectorizer ...")
    model, vectorizer = load_artifacts()
    print("✅ Model loaded successfully.\n")

    # Prediction loop — keeps running until the user types 'quit'
    while True:

        # Step 1 — Get input
        raw_review = get_user_review()

        # Step 2 — Clean the text using the same pipeline as training
        cleaned = preprocess_input(raw_review)

        # Edge case: if cleaning removed ALL words (e.g. input was only symbols),
        # the vectorizer would produce a zero vector and the prediction is meaningless.
        if len(cleaned.strip()) == 0:
            print("\n  ⚠️  After cleaning, your review had no meaningful words.")
            print("      Please try again with a real sentence.\n")
            continue

        # Step 3 — Convert to TF-IDF numeric vector
        review_vector = vectorize_input(cleaned, vectorizer)

        # Step 4 — Predict
        label = predict_sentiment(review_vector, model)

        # Step 5 — Show result
        display_result(label, raw_review)

        # Ask if the user wants to analyse another review
        print("\n  Analyse another review? (Press Enter to continue / type 'quit' to exit)")
        again = input("  → ").strip().lower()
        if again == "quit":
            print("\n  Goodbye! 👋\n")
            break


# Only run main() when this file is executed directly.
# If predict.py is imported by app.py, main() does NOT run automatically —
# only the individual functions (load_artifacts, predict_sentiment, etc.) are available.
if __name__ == "__main__":
    main()
