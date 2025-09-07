from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Lazy-load dense embedding model
dense_model = None

def get_dense_model():
    global dense_model
    if dense_model is None:
        dense_model = SentenceTransformer('all-MiniLM-L6-v2')
    return dense_model

def normalize_text(text):
    """Lowercase and remove simple punctuation"""
    return text.lower().replace(".", " ").replace(",", " ").replace("(", " ").replace(")", " ").strip()

def match_resumes(job_description, resumes, skills_list,
                  w_bm25=0.3, w_dense=0.4, w_skills=0.3):
    """
    Resume matching with:
    - BM25 + Dense scoring
    - Exact skill matching
    """

    results = []

    if not skills_list:
        raise ValueError("Skills list is empty. Please provide at least one skill.")

    # Normalize
    jd_norm = normalize_text(job_description)
    jd_tokens = jd_norm.split()
    skills_norm = [s.lower().strip() for s in skills_list]
    resume_texts_norm = [normalize_text(r["text"]) for r in resumes]

    # BM25
    tokenized_resumes = [txt.split() for txt in resume_texts_norm]
    bm25 = BM25Okapi(tokenized_resumes)

    # Dense embedding
    model = get_dense_model()
    jd_embedding = model.encode(job_description, convert_to_tensor=True)

    for idx, r in enumerate(resumes):
        text_norm = resume_texts_norm[idx]

        # ----- Exact skill matching -----
        matched_skills = [s for s, s_norm in zip(skills_list, skills_norm) if s_norm in text_norm]
        missing_skills = [s for s in skills_list if s not in matched_skills]

        if not matched_skills:
            matched_skills = ["None"]
        if not missing_skills:
            missing_skills = ["None"]

        # ----- BM25 scoring -----
        bm25_scores_all = bm25.get_scores(jd_tokens)
        bm25_score = bm25_scores_all[idx]
        # Normalize 0-1
        bm25_score = (bm25_score - np.min(bm25_scores_all)) / (np.max(bm25_scores_all) - np.min(bm25_scores_all) + 1e-6)

        # ----- Dense scoring -----
        resume_embedding = model.encode(r["text"], convert_to_tensor=True)
        dense_score = util.cos_sim(jd_embedding, resume_embedding).item()

        # ----- Weighted total -----
        skill_score = len([s for s in matched_skills if s != "None"]) / len(skills_list)
        total_score = w_skills * skill_score + w_bm25 * bm25_score + w_dense * dense_score

        # If all skills matched, cap at 100%
        if skill_score == 1.0:
            total_score = 1.0

        results.append({
            "name": r["name"],
            "bm25": round(bm25_score * 100, 2),
            "dense": round(dense_score * 100, 2),
            "skills": matched_skills,
            "missing": missing_skills,
            "score": round(total_score * 100, 2)
        })

    # Sort descending by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
