# QualiGenix – Streamlit GenAI Demo

QualiGenix is a lightweight decision-support demo for pharmaceutical manufacturing teams. It couples structured analytics (historical batch data + pretrained CatBoost/LightGBM models) with a hybrid RAG assistant that rewrites grounded answers using OpenAI's `gpt-5-mini`.

> **Data provenance**: The demo dataset is derived from *“Big data collection in pharmaceutical manufacturing and its use for product quality predictions”* by Žagar & Mihelič (Scientific Data, 2022). DOI: [10.1038/s41597-022-01203-x](https://doi.org/10.1038/s41597-022-01203-x).

## Features
- **Data explorer**: Filter and chart key quality metrics inside the Streamlit UI.
- **ML predictions**: Serve pre-trained models stored in `data/processed/models/` for yield, dissolution and impurity forecasts.
- **Hybrid RAG assistant**: Deterministic analytics first, followed by an LLM rewrite with token/cost reporting.
- **Scenario tooling**: Optimisation and simulation helpers for "what-if" process tweaks.

## Project layout
```
.
├── data/processed/            # Master dataset + trained models (kept small for a demo)
├── src/
│   ├── genai_agent.py         # Hybrid analytics + LLM answer pipeline
│   ├── ml_modeling.py         # Offline training script (optional in the cloud app)
│   ├── data_integration.py    # Dataset preparation utilities
│   └── usage_examples.py      # CLI demo of the agent
├── streamlit_app.py           # Entry-point used by Streamlit Cloud
├── requirements.txt
├── .env.example               # Template for local secrets
└── README.md
```

## Local setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env        # then edit with your real OpenAI key
streamlit run streamlit_app.py
```
The hybrid assistant expects:
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (defaults to `gpt-5-mini`)
- `OPENAI_EMBED_MODEL` (defaults to `text-embedding-3-small`)

## Deploying on Streamlit Cloud (`share.streamlit.io`)
1. **Push this repo to GitHub** (`main` branch). All heavy files are under 100 MB to stay within Streamlit’s limits.
2. In Streamlit Cloud, create a new app pointing to `thisisarminahmadi/QualiGenix` and set the main file to `streamlit_app.py`.
3. Under *App settings → Secrets*, paste:
   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-5-mini"
   OPENAI_EMBED_MODEL = "text-embedding-3-small"
   ```
   (Never commit your real key; `.env` and `secrets.toml` are gitignored.)
4. Click **Deploy**. Build time is ~5–7 minutes because CatBoost/LightGBM wheels are compiled.

## Useful commands
| Description                     | Command |
|---------------------------------|---------|
| Lint with `ruff` (optional)     | `ruff check src` |
| Re-run agent smoke tests        | `PYTHONPATH=src python src/usage_examples.py` |
| Re-train ML models (offline)    | `PYTHONPATH=src python src/ml_modeling.py` |

## Troubleshooting
- **Missing API key**: the chat falls back to deterministic text and displays no token usage line—set the secrets correctly.
- **Large dependencies**: Streamlit Cloud caches wheels, but the first deploy may take a few minutes; future redeploys are faster.
- **Matplotlib cache warning**: harmless on Streamlit; you can ignore it or set `MPLCONFIGDIR` in `~/.streamlit/secrets.toml` if desired.

## License
Internal demo code © 2025 Armin Ahmadi. Review and adapt before any production deployment.
