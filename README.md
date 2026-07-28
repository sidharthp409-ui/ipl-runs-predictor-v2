# IPL Ball Outcome Predictor 🏏

A broadcast-styled Streamlit app that predicts the probability of each ball
outcome (0, 1, 2, 3, 4, 6) using a Random Forest Classifier trained on
phase-specific (powerplay / middle / death) batsman strike rate and bowler
economy, with career averages as a fallback signal.

## Project structure

```
.
├── app.py                  # Main Streamlit app
├── encoders3.pkl           # Label encoders + phase/career stat lookups
├── player_maps.pkl         # Team -> available batsmen/bowlers maps
├── train_model3.pkl        # Trained RandomForestClassifier (ADD THIS — see below)
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Theme
└── .gitignore
```

> **⚠️ Missing file:** `train_model3.pkl` (the trained model) is **not** in
> this folder yet — it wasn't part of the original upload. Copy your trained
> model pickle into the project root before running or deploying, using
> exactly that filename (or update the `pickle.load(...)` path in `app.py`).

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Push to GitHub

```bash
cd ipl-runs-predictor
git init
git add .
git commit -m "Initial commit: IPL ball outcome predictor"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

`train_model3.pkl` — if it's large (check with `ls -lh`), GitHub's normal
100MB file limit applies. For anything over ~50MB, use
[Git LFS](https://git-lfs.com/):

```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add train_model3.pkl
git commit -m "Track model with Git LFS"
git push
```

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub (above).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick this repo/branch, and set **Main file path** to `app.py`.
4. Deploy. The `.streamlit/config.toml` theme and `requirements.txt` are
   picked up automatically.

## Notes

- The three `.pkl` files are loaded relative to the app's working directory,
  so keep them in the repo root alongside `app.py` (don't move them into a
  subfolder unless you also update the paths in `load_artifacts()`).
- If model loading fails on Streamlit Cloud with a `sklearn` version
  mismatch, pin `scikit-learn` in `requirements.txt` to the exact version
  used to train `train_model3.pkl` (check with
  `python -c "import sklearn; print(sklearn.__version__)"` in your training
  environment).
