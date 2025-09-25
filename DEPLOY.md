# Deploying the Autism Predictor (Streamlit)

## 1) Push to GitHub
Commit the whole `autism-predictor` folder to a GitHub repo.

## 2) Streamlit Community Cloud (fastest)
- Go to https://streamlit.io/cloud and sign in with GitHub.
- Click **New app** → pick your repo and branch.
- Set **Main file path** to `streamlit_app.py`.
- Add `requirements.txt` with:
  - streamlit, pandas, numpy, scikit-learn, matplotlib
- (Optional) Keep a small sample data file at `data/autism_data.csv`.

## 3) Hugging Face Spaces (also easy)
- Create a new Space → **Streamlit** template.
- Upload your files.
- In **README** or Space settings, set the main file to `streamlit_app.py`.

## Notes
- The app retrains models on the current dataset each run (simple and reproducible).
- If you want faster load, pretrain and save pickles (`.pkl`), then load them in the app.
