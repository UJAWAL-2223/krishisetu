"""
KRISHI-SETU | FastAPI Backend
==============================
Architecture:
  Farmer query
      ↓
  FAISS semantic search (finds relevant schemes)
      ↓
  Knowledge Graph eligibility check (deterministic reasoning)
      ↓
  Gemini 1.5 Flash (generates farmer-friendly explanation)
      ↓
  Structured JSON response

Run:
  $env:GEMINI_API_KEY = "AIza-your-key"
  uvicorn main:app --reload --port 8000

Test:
  http://localhost:8000/health
  http://localhost:8000/docs
"""

import json
import os
import logging
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("krishi-api")

# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title="Krishi-Setu API",
    description="AI-driven agricultural scheme discovery for Indian farmers",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load FAISS + Knowledge Graph on startup ───────────────
import faiss
from sentence_transformers import SentenceTransformer
from knowledge_graph import KrishiSetuGraph

log.info("Loading FAISS vector store...")
INDEX = faiss.read_index("./vector_store/schemes.index")
with open("./vector_store/schemes_metadata.json", encoding="utf-8") as f:
    METADATA = json.load(f)
log.info(f"FAISS ready — {INDEX.ntotal} schemes")

log.info("Loading embedding model...")
EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
log.info("Embedding model ready")

log.info("Loading knowledge graph...")
KG = KrishiSetuGraph.load("./vector_store/knowledge_graph.pkl")
log.info("Knowledge graph ready")

# ── Gemini setup ──────────────────────────────────────────
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not set.\n"
        "Run: $env:GEMINI_API_KEY = 'your-key-here'"
    )
GEMINI = genai.Client(api_key=GEMINI_KEY)
log.info("Gemini 1.5 Flash ready")


# ── Request / Response models ─────────────────────────────

class FarmerProfile(BaseModel):
    state: str
    crop: str = ""
    land_size: float = 0.0
    purpose: str
    income_band: str = ""
    language: str = "en"

class SchemeCard(BaseModel):
    scheme_name: str
    why_relevant: str
    likely_eligible: bool
    eligibility_reasons: list[str]
    documents_needed: list[str]
    next_step: str
    official_link: str
    relevance_score: float

class RecommendResponse(BaseModel):
    summary: str
    recommendations: list[SchemeCard]
    farmer_profile: dict
    knowledge_graph_used: bool
    schemes_searched: int


# ── FAISS search ──────────────────────────────────────────

def vector_search(query: str, top_k: int = 8) -> list[dict]:
    """
    Converts query to vector, finds closest schemes in FAISS.
    Returns top_k most semantically similar schemes.
    """
    qvec = EMBED_MODEL.encode(
        [query], normalize_embeddings=True
    )[0].reshape(1, -1).astype(np.float32)
    distances, indices = INDEX.search(qvec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(METADATA):
            s = METADATA[idx].copy()
            s["relevance_score"] = round(float(1 / (1 + dist)), 3)
            results.append(s)
    return results


def build_query(profile: FarmerProfile) -> str:
    """Builds a rich search query from farmer profile."""
    parts = [profile.purpose]
    if profile.crop:
        parts.append(f"{profile.crop} crop farmer")
    if profile.land_size > 0:
        parts.append(f"{profile.land_size} acres")
    if profile.state:
        parts.append(profile.state)
    if profile.income_band:
        parts.append(profile.income_band)
    return " ".join(parts)


# ── Gemini call ───────────────────────────────────────────

def call_gemini(profile: FarmerProfile, kg_results: list[dict]) -> dict:
    """
    Sends farmer profile + graph-verified schemes to Gemini.
    Gemini generates human-friendly explanation and next steps.
    """
    schemes_context = ""
    for i, s in enumerate(kg_results[:5], 1):
        eligible_str = "LIKELY ELIGIBLE" if s["likely_eligible"] else "MAY NOT BE ELIGIBLE"
        reasons_for = ", ".join(s["reasons_eligible"]) or "General match"
        reasons_against = ", ".join(s["reasons_not_eligible"]) or "None"
        docs = ", ".join(s["required_documents"][:4]) or "Aadhaar, Land Record"
        schemes_context += f"""
Scheme {i}: {s['scheme_name']}
Status: {eligible_str}
Eligibility Score: {s['eligibility_score']}
Reasons Eligible: {reasons_for}
Reasons Not Eligible: {reasons_against}
Benefits: {s['benefits_preview'][:300]}
Documents: {docs}
Link: {s['official_link']}
---"""

    if profile.language == "hi":
        lang_note = (
            "The farmer prefers Hindi. Write 'why_relevant' and "
            "'next_step' in simple Hindi/Hinglish. Keep scheme names in English."
        )
    else:
        lang_note = "Write all explanations in simple English."

    prompt = f"""You are Krishi-Setu, an AI assistant helping Indian farmers find government schemes.

FARMER PROFILE:
- State: {profile.state}
- Crop: {profile.crop or 'Not specified'}
- Land Size: {profile.land_size} acres
- Need: {profile.purpose}
- Income: {profile.income_band or 'Not specified'}

SCHEMES VERIFIED BY KNOWLEDGE GRAPH:
{schemes_context}

TASK: Recommend TOP 3 most suitable schemes based on the knowledge graph analysis.
{lang_note}

RULES:
- Prioritize LIKELY ELIGIBLE schemes
- Be honest if farmer may not be eligible
- Keep explanations simple — farmer may have low literacy
- next_step must be ONE specific actionable thing

Return ONLY valid JSON, no markdown, no extra text:
{{
  "summary": "2-3 sentences overview in simple language",
  "recommendations": [
    {{
      "scheme_name": "exact name from above",
      "why_relevant": "1-2 sentences why this matches farmer need",
      "likely_eligible": true,
      "eligibility_reasons": ["reason1", "reason2"],
      "documents_needed": ["doc1", "doc2", "doc3"],
      "next_step": "one specific action to take",
      "official_link": "link from above"
    }}
  ]
}}"""

    response = GEMINI.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    raw = response.text.strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ── ENDPOINTS ─────────────────────────────────────────────

@app.get("/health")
def health():
    """Check if server is running correctly."""
    return {
        "status": "ok",
        "schemes_in_vector_store": INDEX.ntotal,
        "knowledge_graph_nodes": KG.graph.number_of_nodes(),
        "knowledge_graph_edges": KG.graph.number_of_edges(),
        "model": "gemini-2.0-flash",
        "embedding": "paraphrase-multilingual-MiniLM-L12-v2"
    }


@app.get("/stats")
def stats():
    """Returns knowledge base statistics."""
    kg_stats = KG.get_graph_stats()
    return {
        "vector_store": {"total_schemes": INDEX.ntotal},
        "knowledge_graph": kg_stats,
        "sample_schemes": [m.get("scheme_name", "") for m in METADATA[:5]]
    }


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(profile: FarmerProfile):
    """
    Main endpoint — takes farmer profile, returns scheme recommendations.

    Flow:
    1. Build semantic query from farmer profile
    2. Search FAISS vector store (RAG layer)
    3. Run eligibility check through Knowledge Graph
    4. Send top results to Gemini 2.0 Flash for explanation
    5. Return structured recommendations
    """
    if not profile.purpose.strip():
        raise HTTPException(
            status_code=400,
            detail="Please describe what you need help with"
        )

    log.info(f"Request — State: {profile.state} | Purpose: {profile.purpose[:50]}")

    # Step 1: FAISS semantic search
    query = build_query(profile)
    log.info(f"Searching FAISS: '{query}'")
    faiss_results = vector_search(query, top_k=8)
    log.info(f"FAISS: {len(faiss_results)} candidates")

    if not faiss_results:
        raise HTTPException(
            status_code=404,
            detail="No schemes found. Try describing your need differently."
        )

    # Step 2: Knowledge Graph eligibility check
    candidate_ids = [
        s.get("scheme_id", "")
        for s in faiss_results
        if s.get("scheme_id")
    ]
    log.info(f"Running knowledge graph on {len(candidate_ids)} candidates...")
    kg_results = KG.check_eligibility(candidate_ids, profile.dict())
    log.info(f"Knowledge Graph: {len(kg_results)} verified")

    # Merge FAISS relevance scores into KG results
    faiss_scores = {
        s.get("scheme_id"): s.get("relevance_score", 0.5)
        for s in faiss_results
    }
    for r in kg_results:
        r["relevance_score"] = faiss_scores.get(r["scheme_id"], 0.5)

    if not kg_results:
        raise HTTPException(
            status_code=404,
            detail="Could not verify eligibility for found schemes."
        )

    # Step 3: Gemini generates explanation
    log.info("Calling Gemini 2.0 Flash...")
    try:
        gemini_output = call_gemini(profile, kg_results)
    except json.JSONDecodeError as e:
        log.error(f"Gemini JSON error: {e}")
        raise HTTPException(
            status_code=500,
            detail="AI response format error. Please try again."
        )
    except Exception as e:
        log.error(f"Gemini error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"AI error: {str(e)}"
        )

    # Step 4: Build structured response
    recommendations = []
    for rec in gemini_output.get("recommendations", []):
        matching_kg = next(
            (r for r in kg_results
             if rec.get("scheme_name", "").lower()[:20]
             in r.get("scheme_name", "").lower()),
            kg_results[0] if kg_results else {}
        )
        recommendations.append(SchemeCard(
            scheme_name=rec.get("scheme_name", ""),
            why_relevant=rec.get("why_relevant", ""),
            likely_eligible=rec.get("likely_eligible", False),
            eligibility_reasons=rec.get("eligibility_reasons", []),
            documents_needed=rec.get(
                "documents_needed",
                matching_kg.get("required_documents", [])
            ),
            next_step=rec.get("next_step", ""),
            official_link=rec.get(
                "official_link",
                matching_kg.get("official_link", "https://www.myscheme.gov.in")
            ),
            relevance_score=matching_kg.get("relevance_score", 0.5),
        ))

    log.info(f"Returning {len(recommendations)} recommendations")

    return RecommendResponse(
        summary=gemini_output.get("summary", ""),
        recommendations=recommendations,
        farmer_profile=profile.dict(),
        knowledge_graph_used=True,
        schemes_searched=len(faiss_results),
    )


@app.get("/search")
async def search(q: str, top_k: int = 5):
    """Direct vector search — for testing FAISS without LLM."""
    if not q:
        raise HTTPException(status_code=400, detail="Query required")
    results = vector_search(q, top_k)
    return {
        "query": q,
        "results": [
            {
                "scheme_name": r.get("scheme_name"),
                "scheme_id": r.get("scheme_id"),
                "relevance_score": r.get("relevance_score"),
                "description_preview": r.get("description", "")[:200],
            }
            for r in results
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)