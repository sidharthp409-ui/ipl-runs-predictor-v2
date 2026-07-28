# IPL Ball Outcome Predictor 🏏

🔴 **Live app:** https://ipl-runs-predictor-v2-isbnd2qvj8bmezdyjxb8kj.streamlit.app/

A broadcast-styled Streamlit app that predicts the probability of each ball
outcome (0, 1, 2, 3, 4, 6) using a Random Forest Classifier trained on
phase-specific (powerplay / middle / death) batsman strike rate and bowler
economy, with career averages as a fallback signal.

## Run locally

\`\`\`bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Deployment

Live on **Streamlit Community Cloud**:
👉 https://ipl-runs-predictor-v2-isbnd2qvj8bmezdyjxb8kj.streamlit.app/

Pushing to `main` on GitHub auto-triggers a redeploy — no manual step needed.

## Push updates to GitHub

\`\`\`bash
git add .
git commit -m "Update app"
git push origin main
\`\`\`
