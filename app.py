from flask import Flask, render_template, request
import fitz  # PyMuPDF
import docx
from matcher import match_resumes

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/match", methods=["POST"])
def match():
    job_description = request.form["job_description"].strip()
    skills_input = request.form["skills"].strip()
    skills_list = [s.strip() for s in skills_input.split(",") if s.strip()]
    uploaded_resumes = request.files.getlist("resumes")

    resumes = []
    for f in uploaded_resumes:
        fname = f.filename or "resume"
        if fname.lower().endswith(".pdf"):
            text = extract_text_from_pdf(f)
        elif fname.lower().endswith(".docx"):
            text = extract_text_from_docx(f)
        else:
            # skip unsupported
            continue
        resumes.append({"name": fname, "text": text})

    # Weights: bm25=0.3, dense=0.4, skills=0.3 (tweak if you like)
    results = match_resumes(job_description, resumes, skills_list,
                            w_bm25=0.3, w_dense=0.4, w_skills=0.3)

    return render_template("results.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
