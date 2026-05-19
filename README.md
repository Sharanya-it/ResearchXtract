# ResearchXtract

ResearchXtract is a Streamlit-based research paper insight extraction and structured summarization system. It helps users upload research PDFs, extract paper sections, generate structured summaries, identify key concepts, ask section-specific questions, extract visuals, recommend related papers, and export a full report.

## Features

- Upload and analyze research papers in PDF format
- Detect and browse paper sections
- Generate structured section summaries using Gemini
- Extract key concepts and important terms
- Ask questions about selected sections
- Extract tables and visual content from PDFs
- Recommend related research papers using Semantic Scholar with Gemini and Mistral fallback support
- Cache summaries, metadata, and recommendations for faster repeat analysis
- Export analyzed content as a PDF report

## Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini
- Mistral AI
- Semantic Scholar API
- PyMuPDF
- pdfplumber
- Camelot
- Plotly

## Project Structure

```text
ResearchXtract/
  app.py
  core/
    document_processor.py
    export_handler.py
    llm_pipeline.py
    recommendation_engine.py
    visual_extractor.py
  utils/
    prompt_templates.py
    text_cleaner.py
  requirements.txt
  run_app.bat
```

## Setup

Clone the repository:

```bash
git clone https://github.com/Sharanya-it/ResearchXtract.git
cd ResearchXtract
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
MISTRAL_API_KEY=your_mistral_api_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
```

`GOOGLE_API_KEY` is required for Gemini-powered summarization and question answering. `MISTRAL_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY` improve related-paper recommendations.

Do not commit `.env` to GitHub.

## Run the App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

You can also run the included batch file on Windows:

```powershell
.\run_app.bat
```

## Usage

1. Upload a research paper PDF from the sidebar.
2. Select a detected section of the paper.
3. View the generated summary and key concepts.
4. Generate visuals for supported PDF content.
5. Ask questions about the selected section.
6. Find related papers from the recommendation panel.
7. Download the generated full report.

## Git Workflow

After making changes:

```bash
git status
git add .
git commit -m "Describe your changes"
git push
```

## Notes

- Local cache files are stored in `.cache/` and should not be committed.
- Virtual environments such as `.venv/` and `venv/` should not be committed.
- Generated temporary data should stay out of Git.
