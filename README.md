# YentiFlix 🎬

A sentiment analysis web app that predicts whether a movie review is **Positive** or **Negative** using NLP and machine learning.

## Tech Stack

- **Python** — core language
- **Flask** — web framework
- **NLTK** — text preprocessing
- **Scikit-learn** — TF-IDF vectorization and classification
- **Matplotlib** — sentiment distribution charts

## Project Structure

```
YentiFlix/
├── data/
│   ├── raw/            # Original unmodified dataset
│   └── processed/      # Cleaned data ready for training
├── models/             # Saved model and vectorizer (.pkl)
├── src/                # Core scripts (preprocess, train, predict, visualize)
├── static/             # CSS, JS, images, and charts
├── templates/          # HTML templates (Jinja2)
├── notebooks/          # Jupyter notebooks for experimentation
└── app.py              # Flask application entry point
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/YentiFlix.git
cd YentiFlix
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### 5. Preprocess the data

```bash
python src/preprocess.py
```

### 6. Train the model

```bash
python src/train_model.py
```

### 7. Run the Flask app

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

## Usage

1. Open the app in your browser.
2. Type or paste a movie review into the input box.
3. Click **Analyze**.
4. The app returns a **Positive** or **Negative** prediction.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
