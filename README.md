# LLM Visualizer (Streamlit)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
- Push `app.py`, `backend.py`, and `requirements.txt` to a public GitHub repo
- In Streamlit Cloud, set the app entry point to `app.py`

## Notes
- Supports Basic exports (sheets like `brand_mentions`, `answers`)
- Supports Advanced exports (sheets like `Results`, `Mentions`, `Citations`, `SearchQueries`)
- Handles thousands of rows by using Streamlit caching and long-format explode tables
