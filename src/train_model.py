# =============================================================================
# train_model.py — YentiFlix Sentiment Analysis
# Purpose : Load cleaned reviews, train a Logistic Regression classifier,
#           evaluate it, and save the model + vectorizer for use in Flask.
# Input   : data/processed/cleaned_reviews.csv
# Output  : models/sentiment_model.pkl
#           models/tfidf_vectorizer.pkl
# Run     : python src/train_model.py
# =============================================================================

# ── Imports ───────────────────────────────────────────────────────────────────

import os       # Builds file paths that work on all operating systems

import joblib   # Saves and loads Python objects (model, vectorizer) as .pkl files
                # Faster and more memory-efficient than pickle for NumPy arrays

import pandas as pd  # Loads the cleaned CSV into a DataFrame

# scikit-learn — the machine learning library
from sklearn.model_selection import train_test_split
# train_test_split : randomly splits data into training and test sets

from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer : converts raw text into a matrix of TF-IDF numeric features
# TF  = Term Frequency     — how often a word appears in THIS review
# IDF = Inverse Document Frequency — penalises words common across ALL reviews
# Result: rare but meaningful words get higher scores than common filler words

from sklearn.linear_model import LogisticRegression
# LogisticRegression : a fast, reliable binary classifier — ideal for text data
# Despite the name, it is a CLASSIFICATION algorithm, not regression

from sklearn.metrics import accuracy_score, classification_report
# accuracy_score       : percentage of correctly predicted labels
# classification_report: precision, recall, F1-score per class — a richer view
#                        of model performance than accuracy alone


# ── File paths ────────────────────────────────────────────────────────────────
# os.path.abspath(__file__) → absolute path to THIS script (src/train_model.py)
# dirname twice             → navigates up two levels to the YentiFlix/ root
# os.path.join              → safely builds paths for any operating system

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_PATH    = os.path.join(BASE_DIR, "data", "processed", "cleaned_reviews.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")


# =============================================================================
# STEP 1 — Load the cleaned dataset
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Read cleaned_reviews.csv into a pandas DataFrame.
    Validates that both required columns are present before continuing.
    """
    print(f"[1/7] Loading cleaned dataset from:\n      {path}\n")

    df = pd.read_csv(path, encoding="utf-8")

    print(f"      Rows loaded : {len(df):,}")
    print(f"      Columns     : {list(df.columns)}")

    # Safety check — fail early with a clear message if columns are wrong
    if "cleaned_review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError(
            "Expected columns 'cleaned_review' and 'sentiment' not found.\n"
            "Make sure you have run preprocess.py first."
        )

    # Drop any rows where the review text or label is missing
    before = len(df)
    df.dropna(subset=["cleaned_review", "sentiment"], inplace=True)
    if len(df) < before:
        print(f"      Dropped {before - len(df)} rows with missing values.")

    print(f"      Clean rows  : {len(df):,}\n")
    return df


# =============================================================================
# STEP 2 — Separate features (X) and labels (y), then split into train/test
# =============================================================================

def split_data(df: pd.DataFrame):
    """
    Split the dataset into training (80%) and test (20%) sets.

    X = the input the model learns from  → the review text
    y = the answer the model predicts    → 'positive' or 'negative'

    Why 80/20?
        80% gives the model enough examples to learn patterns.
        20% is held back so we can measure real-world performance on
        data the model has never seen. This prevents over-optimistic scores.

    random_state=42 ensures the split is identical every time you run the
    script — useful for reproducible experiments and debugging.

    stratify=y keeps the positive/negative class ratio the same in both
    train and test sets, which is good practice for balanced datasets.
    """
    print("[2/7] Splitting dataset into 80% train / 20% test ...")

    X = df["cleaned_review"]   # Series of review strings
    y = df["sentiment"]        # Series of 'positive' / 'negative' labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,       # 20% goes to the test set
        random_state=42,     # Fixed seed for reproducibility
        stratify=y           # Preserve class balance in both sets
    )

    print(f"      Training samples : {len(X_train):,}")
    print(f"      Test samples     : {len(X_test):,}\n")

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 3 — Convert text to numbers using TF-IDF
# =============================================================================

def build_tfidf(X_train, X_test):
    """
    Fit a TF-IDF vectorizer on the training set and transform both sets.

    Why TF-IDF instead of raw word counts?
        A word like 'movie' appears in nearly every review, so its raw count
        is high but it tells us nothing about sentiment. TF-IDF down-weights
        such frequent words and up-weights rare but meaningful words like
        'masterpiece' or 'unwatchable'.

    max_features=50000:
        Only keep the 50,000 most frequent unique words (vocabulary).
        This caps memory usage while retaining the most informative features.
        The IMDB corpus has ~100K+ unique words; capping avoids very rare
        typo-words from becoming features.

    ngram_range=(1, 2):
        Include both single words (unigrams) AND two-word phrases (bigrams).
        Example: 'not good' is captured as a feature instead of just 'not'
        and 'good' separately — this preserves negation context.

    CRITICAL RULE:
        fit_transform on TRAIN data only  → the vectorizer learns vocabulary here
        transform on TEST data only       → applies the SAME vocabulary, no re-learning
        If you fit on test data too, the model gets an unfair preview of test
        vocabulary — this is called data leakage and inflates accuracy scores.
    """
    print("[3/7] Building TF-IDF feature matrix ...")

    vectorizer = TfidfVectorizer(
        max_features=50000,   # Vocabulary size cap
        ngram_range=(1, 2),   # Unigrams + bigrams
        sublinear_tf=True     # Apply log(1 + tf) scaling — reduces impact of
                              # very high-frequency terms within a single review
    )

    # fit_transform: learns vocabulary from training text AND converts it to numbers
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # transform only: converts test text using the ALREADY-LEARNED vocabulary
    X_test_tfidf  = vectorizer.transform(X_test)

    print(f"      Vocabulary size  : {len(vectorizer.vocabulary_):,} terms")
    print(f"      Train matrix     : {X_train_tfidf.shape[0]:,} rows × {X_train_tfidf.shape[1]:,} features")
    print(f"      Test  matrix     : {X_test_tfidf.shape[0]:,} rows × {X_test_tfidf.shape[1]:,} features\n")

    return vectorizer, X_train_tfidf, X_test_tfidf


# =============================================================================
# STEP 4 — Train the Logistic Regression model
# =============================================================================

def train_model(X_train_tfidf, y_train):
    """
    Train a Logistic Regression classifier on the TF-IDF feature matrix.

    Why Logistic Regression for text?
        - Trains fast — handles sparse high-dimensional matrices well
        - Highly interpretable — you can inspect which words push predictions
          toward positive or negative
        - Consistently achieves 88–92% accuracy on IMDB with TF-IDF
        - Industry baseline for binary text classification

    Key parameters:
        max_iter=1000  — maximum optimisation iterations; the default (100)
                         often fails to converge on large text datasets.
        C=1.0          — regularisation strength (default). Lower C = stronger
                         regularisation = simpler model. 1.0 is a safe starting point.
        solver='lbfgs' — the optimisation algorithm. Works well for medium-sized
                         datasets. 'saga' is faster for very large datasets if needed.
        n_jobs=-1      — use ALL available CPU cores to speed up training.
    """
    print("[4/7] Training Logistic Regression model ...")
    print("      This typically takes 30–90 seconds on 40,000 samples ...\n")

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        n_jobs=-1
    )

    # .fit() is where the actual learning happens.
    # The model adjusts its internal weights to minimise prediction error
    # on the training set.
    model.fit(X_train_tfidf, y_train)

    print("      Model training complete.\n")
    return model


# =============================================================================
# STEP 5 — Evaluate model performance on the test set
# =============================================================================

def evaluate_model(model, X_test_tfidf, y_test) -> None:
    """
    Measure how well the trained model performs on data it has never seen.

    accuracy_score:
        (Correct predictions) / (Total predictions)
        e.g. 0.912 = 91.2% of test reviews classified correctly.

    classification_report breaks performance down per class:
        Precision — of all reviews predicted Positive, how many actually were?
        Recall    — of all actual Positive reviews, how many did we catch?
        F1-score  — harmonic mean of precision and recall (best single metric
                    when you care about both false positives and false negatives)
        Support   — number of actual samples in each class
    """
    print("[5/7] Evaluating model on test set ...")

    # Generate predictions for every review in the test set
    y_pred = model.predict(X_test_tfidf)

    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print(f"  TEST ACCURACY : {accuracy * 100:.2f}%")
    print("=" * 55)

    # Detailed per-class metrics
    print("\nClassification Report:")
    print("-" * 55)
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))


# =============================================================================
# STEP 6 & 7 — Save model and vectorizer to disk
# =============================================================================

def save_artifacts(model, vectorizer) -> None:
    """
    Persist the trained model and fitted vectorizer as .pkl files.

    Why save both files?
        When a user submits a new review in the Flask app, the raw text must
        be converted to a TF-IDF vector using EXACTLY the same vocabulary the
        model was trained on. Saving the vectorizer ensures this consistency.
        Loading and re-training on every request would be far too slow.

    joblib vs pickle:
        Both serialise Python objects. joblib is preferred for scikit-learn
        objects because it efficiently handles the large NumPy arrays inside
        the model and vectorizer using memory-mapped files.

    joblib.dump(object, path):
        Serialises the object and writes it to the given file path.
    """
    print("[6/7] Saving trained model ...")

    # Create the models/ directory if it doesn't exist yet
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"      Model saved      → {MODEL_PATH}")
    print(f"      Vectorizer saved → {VECTORIZER_PATH}\n")


# =============================================================================
# Entry point
# =============================================================================

def main():
    """
    Runs the full training pipeline in order:
        Load → Split → Vectorize → Train → Evaluate → Save
    """
    print("=" * 55)
    print("  YentiFlix — Model Training Pipeline")
    print("=" * 55 + "\n")

    # Step 1 — Load data
    df = load_data(CLEANED_PATH)

    # Step 2 — Split into train / test
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 3 — TF-IDF vectorization
    vectorizer, X_train_tfidf, X_test_tfidf = build_tfidf(X_train, X_test)

    # Step 4 — Train classifier
    model = train_model(X_train_tfidf, y_train)

    # Step 5 — Evaluate
    evaluate_model(model, X_test_tfidf, y_test)

    # Steps 6 & 7 — Save artifacts
    save_artifacts(model, vectorizer)

    # Final status
    print("[7/7] All done!")
    print("\n" + "=" * 55)
    print("  Training complete. Next step: run predict.py")
    print("=" * 55)


# Run main() only when this script is executed directly.
# Prevents auto-execution when imported by another module (e.g. app.py).
if __name__ == "__main__":
    main()
