# matcher.py
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import numpy as np

# Load the embedding model once (cached by sentence-transformers)
_model = SentenceTransformer("all-MiniLM-L6-v2")

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize (safe for constant arrays)."""
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)

def match_resumes(job_description: str, resumes: list[dict], skills_list: list[str],
                  w_bm25: float = 0.3, w_dense: float = 0.4, w_skills: float = 0.3) -> list[dict]:
    """
    resumes: list of { 'name': str, 'text': str }
    returns: sorted list of dicts with bm25, dense, skills, missing, score
    """
    if not resumes:
        return []

    # ---------- BM25 (keyword relevance) ----------
    corpus_tokens = [r["text"].lower().split() for r in resumes]
    bm25 = BM25Okapi(corpus_tokens)
    jd_tokens = job_description.lower().split()
    bm25_scores = np.array(bm25.get_scores(jd_tokens), dtype=float)
    bm25_norm = _normalize(bm25_scores)

    # ---------- Embeddings (semantic relevance) ----------
    resume_embeddings = _model.encode([r["text"] for r in resumes], normalize_embeddings=True)
    jd_embedding = _model.encode([job_description], normalize_embeddings=True)
    dense_scores = util.cos_sim(jd_embedding, resume_embeddings)[0].cpu().numpy()
    dense_norm = _normalize(dense_scores)

    # ---------- Skills coverage ----------
    results = []
    skills_lower = [s.lower() for s in skills_list]
    for i, r in enumerate(resumes):
        text_lower = r["text"].lower()
        matched = [s for s in skills_list if s.lower() in text_lower]
        missing = [s for s in skills_list if s.lower() not in text_lower]
        coverage = (len(matched) / len(skills_list)) if skills_list else 0.0

        final = (w_bm25 * float(bm25_norm[i])) + (w_dense * float(dense_norm[i])) + (w_skills * coverage)

        results.append({
            "name": r["name"],
            "bm25": round(float(bm25_norm[i]), 2),
            "dense": round(float(dense_norm[i]), 2),
            "skills": matched,
            "missing": missing,
            "score": round(final * 100, 2)
        })

    # sort by final score, desc
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
