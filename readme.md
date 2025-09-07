# Resume Screener

A Flask web application that screens and ranks resumes against job requirements using both keyword matching (BM25) and semantic similarity (Sentence Transformers).

## Features

- Upload multiple resumes in PDF or DOCX format
- Enter job requirements and required skills
- Analyze resumes using:
  - BM25 keyword matching
  - Semantic similarity with MiniLM
- Visual results with interactive charts
- Displays matched and missing skills for each resume

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resume-screener.git
cd resume-screener
```

2. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload resumes and enter job requirements

4. View the analysis results with score breakdowns

## Requirements

- Python 3.8+
- Flask 2.3+
- PyMuPDF 1.23+
- python-docx 0.8.11+
- rank-bm25 0.2.2+
- sentence-transformers 2.2.2+
- numpy 1.24+
- gunicorn 20.1.0+

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Author
Abhyudaya Sharma