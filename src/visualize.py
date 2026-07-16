# =============================================================================
# visualize.py — YentiFlix Sentiment Analysis
# Purpose : Load the cleaned dataset and generate charts that show the
#           sentiment distribution — both as a bar chart and a pie chart.
# Input   : data/processed/cleaned_reviews.csv
# Output  : static/charts/sentiment_distribution.png
#           static/charts/sentiment_pie.png
# Run     : python src/visualize.py
# =============================================================================

# ── Imports ───────────────────────────────────────────────────────────────────

import os                        # Builds cross-platform file paths and creates folders

import pandas as pd              # Loads the CSV dataset into a DataFrame

import matplotlib                # Core matplotlib library
matplotlib.use("Agg")            # Use non-interactive backend — IMPORTANT for servers
                                 # "Agg" renders charts to image files without needing
                                 # a display or GUI window. This prevents errors when
                                 # running on Linux servers or inside Flask later.

import matplotlib.pyplot as plt  # pyplot is the high-level plotting interface
                                 # All chart-drawing commands come from here


# ── File Paths ────────────────────────────────────────────────────────────────
# Build all paths from the project root so the script works regardless of
# which directory it is launched from.
#
# os.path.abspath(__file__)  → full path to THIS file  (src/visualize.py)
# dirname once               → src/  folder
# dirname twice              → YentiFlix/ project root

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input — cleaned dataset produced by preprocess.py
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "cleaned_reviews.csv")

# Output — folder where all chart images will be saved
CHARTS_DIR  = os.path.join(BASE_DIR, "static", "charts")

# Individual chart file paths
BAR_CHART_PATH = os.path.join(CHARTS_DIR, "sentiment_distribution.png")
PIE_CHART_PATH = os.path.join(CHARTS_DIR, "sentiment_pie.png")

# ── Colour Palette ────────────────────────────────────────────────────────────
# Defining colours as constants makes them easy to change in one place.
# These are standard matplotlib named colours — no external library needed.

COLOR_POSITIVE = "#2ecc71"   # Green  — represents positive sentiment
COLOR_NEGATIVE = "#e74c3c"   # Red    — represents negative sentiment
COLOR_BG       = "#f9f9f9"   # Light grey background for chart areas


# =============================================================================
# STEP 1 — Load the cleaned dataset
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Read cleaned_reviews.csv into a pandas DataFrame.

    Validates that the file exists and that both required columns are present
    before any plotting begins, giving a clear error message if not.

    Parameters
    ----------
    path : str — absolute path to cleaned_reviews.csv

    Returns
    -------
    pd.DataFrame with columns: cleaned_review, sentiment
    """
    print("Loading dataset ...")

    # Fail clearly if the file doesn't exist yet
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cleaned dataset not found at:\n  {path}\n"
            "Please run preprocess.py first to generate this file."
        )

    df = pd.read_csv(path, encoding="utf-8")

    # Validate columns
    if "sentiment" not in df.columns:
        raise ValueError(
            "Column 'sentiment' not found in the dataset.\n"
            "Expected columns: cleaned_review, sentiment"
        )

    # Drop rows with missing sentiment label — they would distort the charts
    df.dropna(subset=["sentiment"], inplace=True)

    print(f"  Rows loaded : {len(df):,}")
    print(f"  Sentiments  : {df['sentiment'].value_counts().to_dict()}\n")

    return df


# =============================================================================
# STEP 2 — Bar Chart: Sentiment Distribution
# =============================================================================

def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    """
    Create a bar chart showing the raw count of positive vs negative reviews
    and save it to static/charts/sentiment_distribution.png.

    Chart anatomy:
        - X axis : sentiment categories ('positive', 'negative')
        - Y axis : number of reviews in each category
        - Each bar is labelled with its exact count for easy reading
        - A horizontal grid makes it easier to read heights visually

    matplotlib steps:
        plt.figure()      — create a new blank canvas (figure)
        ax = fig.add_subplot() or plt.subplots() — add a drawing area (axes)
        ax.bar()          — draw the bars
        ax.set_*()        — configure title, axis labels, tick labels
        plt.tight_layout()— auto-adjusts spacing so nothing is clipped
        plt.savefig()     — write the image to disk
        plt.close()       — release memory (important in long-running apps)

    Parameters
    ----------
    df : pd.DataFrame — cleaned dataset with a 'sentiment' column
    """
    print("  Generating bar chart: sentiment_distribution.png ...")

    # Count how many reviews belong to each sentiment category
    # value_counts() returns a Series: index = category, value = count
    counts = df["sentiment"].value_counts()

    # Extract individual counts with .get() so missing categories default to 0
    positive_count = counts.get("positive", 0)
    negative_count = counts.get("negative", 0)

    categories = ["Positive", "Negative"]
    values     = [positive_count, negative_count]
    colors     = [COLOR_POSITIVE, COLOR_NEGATIVE]

    # ── Build the figure ──────────────────────────────────────────────────────
    # figsize=(width, height) in inches — 8×5 produces a readable landscape chart
    fig, ax = plt.subplots(figsize=(8, 5))

    # Set a light background colour on the axes area
    ax.set_facecolor(COLOR_BG)
    fig.patch.set_facecolor("white")   # Outer figure background stays white

    # Draw the bars
    # bar(x_positions, heights, color, edgecolor, linewidth, width)
    bars = ax.bar(
        categories,            # X-axis tick labels and bar positions
        values,                # Bar heights (the counts)
        color=colors,          # Fill colour for each bar
        edgecolor="white",     # White outline separates bars from background
        linewidth=1.2,
        width=0.5              # Bar width as a fraction of available space
    )

    # ── Add count labels on top of each bar ───────────────────────────────────
    # ax.text(x, y, text) places a text annotation at the given coordinates.
    # bar.get_x() + bar.get_width()/2  → horizontal centre of the bar
    # bar.get_height() + 200           → just above the top of the bar
    # ha='center'                      → horizontally centred text
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,   # X: centre of bar
            height + 200,                          # Y: slightly above bar top
            f"{height:,}",                         # Text: count with comma separator
            ha="center",                           # Horizontal alignment
            va="bottom",                           # Vertical alignment
            fontsize=12,
            fontweight="bold",
            color="#2c3e50"
        )

    # ── Labels, title, and grid ───────────────────────────────────────────────
    ax.set_title(
        "Sentiment Distribution — YentiFlix\nIMDB 50K Movie Reviews",
        fontsize=15,
        fontweight="bold",
        color="#2c3e50",
        pad=15              # Space between title and chart area
    )
    ax.set_xlabel("Sentiment",      fontsize=12, labelpad=10, color="#2c3e50")
    ax.set_ylabel("Number of Reviews", fontsize=12, labelpad=10, color="#2c3e50")

    # Format Y-axis tick labels with comma separators (25,000 not 25000)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    # Horizontal grid lines only — vertical grid adds visual clutter on bar charts
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="#cccccc")
    ax.set_axisbelow(True)    # Draw grid lines BEHIND the bars, not on top

    # Remove the top and right chart borders (spines) for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────────────
    # tight_layout() adjusts all padding automatically so titles/labels aren't cut off
    plt.tight_layout()
    plt.savefig(BAR_CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close()   # Free the figure from memory — always close after saving

    print(f"  Saved → {BAR_CHART_PATH}")


# =============================================================================
# STEP 3 — Pie Chart: Sentiment Percentage Split
# =============================================================================

def plot_sentiment_pie(df: pd.DataFrame) -> None:
    """
    Create a pie chart showing the percentage split between positive and
    negative reviews, and save it to static/charts/sentiment_pie.png.

    The pie chart complements the bar chart:
        - Bar chart answers "how many?"
        - Pie chart answers "what proportion?"

    matplotlib pie chart key parameters:
        sizes       — the numeric values that determine slice angles
        labels      — text displayed next to each slice
        colors      — fill colour for each slice
        autopct     — format string for percentage labels inside slices
                      '%1.1f%%' → one decimal place e.g. "49.8%"
        startangle  — rotates the chart; 140° puts positive on top-left
        explode     — pulls a slice slightly out for visual emphasis
        shadow      — adds a subtle drop shadow for depth

    Parameters
    ----------
    df : pd.DataFrame — cleaned dataset with a 'sentiment' column
    """
    print("  Generating pie chart: sentiment_pie.png ...")

    # Count reviews per class
    counts = df["sentiment"].value_counts()
    positive_count = counts.get("positive", 0)
    negative_count = counts.get("negative", 0)

    sizes  = [positive_count, negative_count]
    labels = [
        f"Positive\n({positive_count:,})",
        f"Negative\n({negative_count:,})"
    ]
    colors  = [COLOR_POSITIVE, COLOR_NEGATIVE]

    # explode slightly separates the first slice (Positive) from the pie
    # 0.05 = 5% of the radius — a subtle effect, not distracting
    explode = (0.05, 0)

    # ── Build the figure ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("white")

    # Draw the pie
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",     # Show percentage inside each slice
        startangle=140,        # Rotate so positive slice starts near top
        explode=explode,
        shadow=True,           # Subtle depth effect
        wedgeprops={           # Style each pie slice
            "edgecolor": "white",
            "linewidth": 2
        }
    )

    # Style the percentage text inside each slice
    for autotext in autotexts:
        autotext.set_fontsize(13)
        autotext.set_fontweight("bold")
        autotext.set_color("white")   # White text is readable on coloured slices

    # Style the outer labels
    for text in texts:
        text.set_fontsize(12)
        text.set_color("#2c3e50")

    # ── Title and legend ──────────────────────────────────────────────────────
    ax.set_title(
        "Sentiment Split — YentiFlix\nIMDB 50K Movie Reviews",
        fontsize=15,
        fontweight="bold",
        color="#2c3e50",
        pad=20
    )

    # Add a legend in the lower-right corner
    ax.legend(
        wedges,
        ["Positive 😊", "Negative 😡"],
        loc="lower right",
        fontsize=11
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(PIE_CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved → {PIE_CHART_PATH}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    """
    Runs the full visualization pipeline:
        1. Create output directory if needed
        2. Load cleaned dataset
        3. Generate bar chart
        4. Generate pie chart
    """
    print("=" * 55)
    print("  YentiFlix — Visualization Pipeline")
    print("=" * 55 + "\n")

    # Create static/charts/ folder if it doesn't exist yet.
    # exist_ok=True means no error is raised if the folder already exists.
    os.makedirs(CHARTS_DIR, exist_ok=True)
    print(f"Charts will be saved to:\n  {CHARTS_DIR}\n")

    # Step 1 — Load data
    df = load_data(DATA_PATH)

    # Step 2 — Generate charts
    print("Generating charts ...")
    plot_sentiment_distribution(df)
    plot_sentiment_pie(df)

    # Done
    print("\nCharts saved successfully ✅")
    print("\n" + "=" * 55)
    print("  Visualization complete.")
    print("  Next step: build app.py")
    print("=" * 55)


# Run main() only when this script is executed directly.
# When visualize.py is imported by app.py, main() does NOT auto-run.
if __name__ == "__main__":
    main()
