from flask import Flask, render_template, request
import os
from matcher import match_resumes  # your BM25 + Dense + exact skill matcher
import fitz  # PyMuPDF
import docx

app = Flask(__name__)

# ---- File Parsing ----
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    doc_obj = docx.Document(file)
    return "\n".join([p.text for p in doc_obj.paragraphs])

# ---- Routes ----
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/match", methods=["POST"])
def match():
    # Get job description
    job_description = request.form.get("job_description", "").strip()
    if not job_description:
        return "Error: Job description cannot be empty", 400

    # Get skills
    skills_input = request.form.get("skills", "").strip()
    if not skills_input:
        return "Error: Please enter at least one skill, comma separated.", 400

    skills_list = [s.strip() for s in skills_input.split(",") if s.strip()]

    # Get uploaded resumes
    uploaded_resumes = request.files.getlist("resumes")
    if not uploaded_resumes:
        return "Error: Please upload at least one resume.", 400

    resumes = []
    for f in uploaded_resumes:
        fname = f.filename or "resume"
        text = ""
        if fname.lower().endswith(".pdf"):
            text = extract_text_from_pdf(f)
        elif fname.lower().endswith(".docx"):
            text = extract_text_from_docx(f)
        else:
            continue  # skip unsupported files
        resumes.append({"name": fname, "text": text})

    # ----- Match resumes -----
    results = match_resumes(job_description, resumes, skills_list,
                            w_bm25=0.3, w_dense=0.7, w_skills=0.0)  # w_skills=0 if you only want BM25+Dense scoring
    print(results)
    return render_template("results.html", results=results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

