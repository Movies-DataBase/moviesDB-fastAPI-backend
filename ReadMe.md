# Movie Recommendation API

FastAPI backend that generates movie recommendations using the **best available free model on OpenRouter**, selected automatically at runtime.

## How model selection works

`GET /free-models` and the internal `get_best_free_model()` call `https://openrouter.ai/api/v1/models`, filter to models whose ID ends in `:free` (OpenRouter's convention for zero-cost models) with `$0` prompt/completion pricing, then pick the one with the **largest context window** as a proxy for capability. Result is cached in memory for 30 minutes so you're not hitting `/models` on every request.

## Setup

```bash
cd movie-recs
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your OpenRouter key
```

## Run

```bash
uvicorn main:app --reload
```

Server starts at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

## Endpoints

### `GET /health`
Simple liveness check.

### `GET /free-models`
Lists every currently free model on OpenRouter, sorted by context length descending. Useful for debugging which model got picked.

### `POST /recommend`
Request body:
```json
{
  "favorite_movies": ["Inception", "Interstellar"],
  "genres": ["sci-fi", "thriller"],
  "mood": "mind-bending, cerebral",
  "language": "English",
  "exclude": ["Tenet"],
  "count": 5
}
```
All fields are optional — send `{}` for a generic set of picks.

Response:
```json
{
  "model_used": "meta-llama/llama-3.1-405b-instruct:free",
  "recommendations": [
    {
      "title": "Arrival",
      "year": "2016",
      "genre": "Sci-Fi",
      "reason": "Shares Interstellar's emotional, high-concept approach to sci-fi."
    }
  ]
}
```
If the model doesn't return clean JSON (free models can be inconsistent), `recommendations` will be empty and `raw_text` will carry the raw model output instead, so you never lose the response.

## Notes / next steps
- Add Redis or a DB-backed cache if you want the free-model list cached across restarts/instances.
- Add retry logic (`httpx` + `tenacity`) since free-tier models on OpenRouter are sometimes rate-limited or briefly unavailable — on failure you could fall back to the next model in the sorted free list instead of failing outright.
- Wire this up to your Angular/React frontend by pointing it at `/recommend`; CORS is already open (`allow_origins=["*"]`) for local dev — restrict this before deploying.