"""
Movie Recommendation API
-------------------------
FastAPI backend that talks to OpenRouter's free-tier LLMs to generate
movie recommendations based on user preferences.

Run:
    uvicorn main:app --reload

Env:
    OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
"""

import os
import time
import logging
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("movie-recs")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not set — set it in your environment or a .env file.")

app = FastAPI(title="Movie Recommendation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory cache for the "best free model" so we don't hit /models on every
# request. Refreshed every CACHE_TTL seconds.
# ---------------------------------------------------------------------------
_model_cache = {"model_id": None, "fetched_at": 0}
CACHE_TTL = 60 * 30  # 30 minutes


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    favorite_movies: list[str] = Field(default_factory=list, description="Movies the user likes")
    genres: list[str] = Field(default_factory=list, description="Preferred genres")
    mood: Optional[str] = Field(None, description="e.g. 'something light', 'mind-bending', 'feel-good'")
    language: Optional[str] = Field(None, description="Preferred language/industry, e.g. Hindi, Korean")
    exclude: list[str] = Field(default_factory=list, description="Movies to avoid recommending")
    count: int = Field(5, ge=1, le=10, description="How many recommendations to return")


class Recommendation(BaseModel):
    title: str
    year: Optional[str] = None
    genre: Optional[str] = None
    reason: str


class RecommendResponse(BaseModel):
    model_used: str
    recommendations: list[Recommendation]
    raw_text: Optional[str] = None  # fallback if JSON parsing fails


# ---------------------------------------------------------------------------
# OpenRouter helpers
# ---------------------------------------------------------------------------
async def get_best_free_model() -> str:
    """
    Fetch OpenRouter's model list, filter to $0-cost ("free") models,
    and pick the one with the largest context window as a proxy for
    "best" general-purpose free model.
    """
    now = time.time()
    if _model_cache["model_id"] and (now - _model_cache["fetched_at"] < CACHE_TTL):
        return _model_cache["model_id"]

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(f"{OPENROUTER_BASE_URL}/models")
        resp.raise_for_status()
        data = resp.json()["data"]

    free_models = []
    for m in data:
        pricing = m.get("pricing", {})
        try:
            prompt_cost = float(pricing.get("prompt", "1"))
            completion_cost = float(pricing.get("completion", "1"))
        except (TypeError, ValueError):
            continue
        # OpenRouter marks free models with ":free" suffix and $0 pricing
        if prompt_cost == 0 and completion_cost == 0 and m["id"].endswith(":free"):
            free_models.append(m)

    if not free_models:
        raise HTTPException(status_code=503, detail="No free models currently available on OpenRouter.")

    # Rank by context length (bigger context = generally more capable / flexible)
    free_models.sort(key=lambda m: m.get("context_length", 0), reverse=True)
    best = free_models[0]["id"]
    print(free_models)

    _model_cache["model_id"] = best
    _model_cache["fetched_at"] = now
    logger.info(f"Selected free model: {best}")
    return best


SYSTEM_PROMPT = """You are a knowledgeable, tasteful film recommendation expert.
Given a user's favorite movies, preferred genres, mood, and language preferences,
recommend movies they are highly likely to enjoy.

Rules:
- Recommend real, existing movies only. Do not invent titles.
- Never recommend anything in the user's exclude list.
- Prefer variety (different decades/directors) unless the user's taste is very narrow.
- Respond ONLY with a valid JSON array, no markdown fences, no commentary. Format:
[
  {"title": "...", "year": "YYYY", "genre": "...", "reason": "1-2 sentence reason tailored to this user"}
]
"""


def build_user_prompt(req: RecommendRequest) -> str:
    parts = []
    if req.favorite_movies:
        parts.append(f"Favorite movies: {', '.join(req.favorite_movies)}")
    if req.genres:
        parts.append(f"Preferred genres: {', '.join(req.genres)}")
    if req.mood:
        parts.append(f"Mood/vibe: {req.mood}")
    if req.language:
        parts.append(f"Preferred language/industry: {req.language}")
    if req.exclude:
        parts.append(f"Do NOT recommend: {', '.join(req.exclude)}")
    parts.append(f"Give exactly {req.count} recommendations.")
    return "\n".join(parts) if parts else "Recommend 5 well-loved movies across genres."


async def call_openrouter(model_id: str, system_prompt: str, user_prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not configured on the server.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter for analytics/rate-limit tiers
        "HTTP-Referer": "https://www.yasv.win",
        "X-Title": "Movie Recommendation API",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.8,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload)

    if resp.status_code != 200:
        logger.error(f"OpenRouter error {resp.status_code}: {resp.text}")
        raise HTTPException(status_code=502, detail=f"OpenRouter request failed: {resp.text}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="Unexpected response shape from OpenRouter.")


def parse_recommendations(raw: str) -> Optional[list[Recommendation]]:
    import json
    import re

    # Strip accidental markdown fences if the model ignores instructions
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        items = json.loads(cleaned)
        return [Recommendation(**item) for item in items]
    except Exception as e:
        logger.warning(f"Failed to parse model output as JSON: {e}")
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/free-models")
async def list_free_models():
    """Debug endpoint: see every free model OpenRouter currently offers."""
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(f"{OPENROUTER_BASE_URL}/models")
        resp.raise_for_status()
        data = resp.json()["data"]

    free_models = [
        {"id": m["id"], "context_length": m.get("context_length"), "name": m.get("name")}
        for m in data
        if m["id"].endswith(":free")
    ]
    free_models.sort(key=lambda m: m["context_length"] or 0, reverse=True)
    return {"count": len(free_models), "models": free_models}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    model_id = 'nvidia/nemotron-3.5-lightning:free'
    # model_id = await get_best_free_model()
    
    user_prompt = build_user_prompt(req)

    raw_output = await call_openrouter(model_id, SYSTEM_PROMPT, user_prompt)
    recommendations = parse_recommendations(raw_output)

    if recommendations is None:
        # Return raw text so the client isn't left with nothing
        return RecommendResponse(model_used=model_id, recommendations=[], raw_text=raw_output)

    return RecommendResponse(model_used=model_id, recommendations=recommendations)