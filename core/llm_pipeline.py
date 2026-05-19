import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

def extract_authors_with_ai(author_zone_text):
    """Simple string return to prevent JSON parsing failure."""
    prompt = (
        "List ONLY the author names from this text as a comma-separated string. "
        "Exclude the title, emails, and university names:\n\n" + author_zone_text
    )
    try:
        res = llm.invoke(prompt).content
        return res.strip()
    except Exception as e:
        return f"Author Detection Error: {e}"
    


def summarize_section(title, content):
    """
    Summarizes the section and marks critical points with '===' 
    for the highlighter in app.py.
    """
    prompt = (
        f"Summarize the section '{title}' from this research paper.\n"
        "1. Provide a concise overview.\n"
        "2. Identify 2-3 most CRITICAL points or findings.\n"
        "3. Wrap these critical points in '===' (e.g., ===This is a key finding===).\n\n"
        f"Content:\n{content}"
    )
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Summary extraction failed: {e}"


def extract_keywords(content):
    """Extracts 5 technical keywords and definitions."""
    try:
        return llm.invoke(f"Extract 5 key concepts and definitions:\n\n{content}").content
    except Exception as e:
        return f"Keyword extraction failed: {e}"

def generate_graph_data(table_text):
    """Converts table text to Chart JSON for Plotly."""
    prompt = (
        "Extract numerical data for a bar chart. Return ONLY JSON: "
        "{'labels':[], 'series':{'Name':[]}, 'x_label':'X', 'y_label':'Y'}. "
        "If no metrics found, return 'NULL'.\n\nTable:\n" + table_text
    )
    try:
        res = llm.invoke(prompt).content
        if "NULL" in res: return None
        clean = res.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        return f"Graph generation failed: {e}"

def answer_paper_question(question, full_text):
    """
    NEW: Answers user questions based on paper context.
    Fixed Indentation: All lines below 'def' are shifted 4 spaces.
    """
    prompt = (
        f"You are an expert research assistant. Answer the question based ONLY on the paper text.\n"
        f"If the answer isn't there, say 'I cannot find that in the paper.'\n\n"
        f"Paper Text: {full_text[:15000]}\n\n"
        f"Question: {question}"
    )
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Question answering failed: {e}"