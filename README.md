# AI Resume Screener + AI Resume Feedback Generator

This repository contains an AI Resume Screening System and a separate Claude-powered resume feedback tool.

## Projects

### 1. Existing AI Resume Screener
The original Gradio application is in `app.py`. It uses TF-IDF and a Random Forest model to explore job-category prediction and resume/job-description similarity. Similarity scores are heuristic and should not be interpreted as actual ATS scores, hiring recommendations, or predictions of selection.

### 2. AI Resume Feedback Generator (Claude API)
The new Streamlit application is in `resume_feedback_app.py`. It accepts a resume (PDF, DOCX, or TXT) and a job description, then uses Anthropic's Claude API to produce structured, evidence-based feedback. The supporting logic is in `resume_feedback.py`.

Feedback includes relevant strengths, potential gaps, keyword alignment, actionable edits, questions for the applicant, and limitations. It must be reviewed by a human and does not guarantee ATS compatibility or hiring outcomes.

## Requirements

- Python 3.10+
- For Claude feedback: an Anthropic API key with API access and billing configured.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure Claude API

Copy `.env.example` to `.env` and set your key:

Windows:
```powershell
Copy-Item .env.example .env
```

macOS/Linux:
```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder for `ANTHROPIC_API_KEY`. Never commit your real `.env` file or expose your API key. The repository's `.gitignore` excludes it.

### Run the Claude-powered feedback app

```bash
streamlit run resume_feedback_app.py
```

### Run the original Gradio screener

```bash
python app.py
```

## Tests

```bash
pytest -q
```

The tests cover basic text extraction, file validation, prompt boundaries, and missing API-key handling. They do not call the live Anthropic API.

## Privacy and security

- When you request feedback, resume text and the job description are sent to Anthropic for processing. Review the provider's current privacy and retention terms before sharing sensitive information.
- The app does not intentionally store uploads in a database. Hosting, browser, logs, and API-provider handling may differ.
- Keep API keys in local environment variables, a local ignored `.env`, or your hosting provider's secrets manager.
- Do not commit resumes, personal data, or API keys.
- Uploads are limited to 5 MB; input text sent to the model is length-limited.
- Review all AI-generated suggestions and never add skills, qualifications, or achievements you cannot substantiate.

## License

MIT. See [LICENSE](LICENSE).
