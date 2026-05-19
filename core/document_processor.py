import fitz  # PyMuPDF
import pdfplumber
import io
import re


def extract_metadata(pdf_bytes):
    """Detects Title via font size, Author Zone for AI, and Year via Regex."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    blocks = page.get_text("dict")["blocks"]

    # 1. Capture Author Zone (Top 35% of page)
    rect = page.rect
    author_zone = fitz.Rect(0, 0, rect.width, rect.height * 0.35)
    author_zone_text = page.get_text("text", clip=author_zone)

    # 2. Extract Title based on largest visual font
    spans = []
    for b in blocks:
        if "lines" in b:
            for l in b["lines"]:
                for s in l["spans"]:
                    if s["text"].strip() and s["size"] > 8:
                        spans.append(s)

    spans.sort(key=lambda x: x["size"], reverse=True)
    max_size = spans[0]["size"] if spans else 0
    title_parts = [s["text"] for s in spans if s["size"] >= (max_size - 1)]
    full_title = " ".join(title_parts).strip()

    # 3. Robust Year Detection (Scanning 202x pattern)
    year_match = re.search(r'\b202[0-9]\b', page.get_text())
    detected_year = year_match.group() if year_match else "2025"

    result = {
        "title": full_title,
        "author_zone_text": author_zone_text,
        "detected_year": detected_year
    }
    doc.close()
    return result


def extract_sections(pdf_bytes):
    """
    Surgical Fix: Strictly follows the 2-column flow and captures every 
    heading level (I, A, 1, etc.) while ignoring 'noise' numbers.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    sections = []
    
    # Matches: I. SECTION, A. Sub-section, 1) Topic, or 1.1 Topic
    header_pattern = r'^([IVXLCDM]+\.|[A-Z]\.|[0-9]+\.|[0-9]+\)|[0-9]+\.[0-9]+)\s+[A-Z]'
    
    for page in doc:
        width = page.rect.width
        # Get blocks with spatial metadata
        blocks = page.get_text("blocks")
        
        # COLUMN-FIRST SORTING:
        # Group by Column (Left then Right) and then by Vertical (Top to Bottom)
        blocks.sort(key=lambda b: (0 if b[0] < width/2 else 1, b[1]))
        
        for b in blocks:
            line = b[4].strip()
            
            # 1. Clean up line breaks within a single block
            line = line.replace('\n', ' ')
            
            # 2. Check for Section Keywords (Case Insensitive)
            is_keyword = any(k in line.upper() for k in [
                "ABSTRACT", "INTRODUCTION", "RELATED WORK", "METHODOLOGIES", 
                "IMPLEMENTATION", "RESULTS", "CONCLUSION", "REFERENCES"
            ])
            
            # 3. Check for Header Pattern
            is_pattern = re.match(header_pattern, line)
            
            # 4. Filter: Must be a header/keyword, short, and not just a page number
            if (is_pattern or is_keyword) and len(line) < 90:
                # Remove duplicate detection
                if line not in sections:
                    sections.append(line)
                    
    doc.close()
    return sections


def get_section_text(pdf_bytes, section_name, sections_list):
    """Slices text between headers using the same column-aware sorting."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    width = doc[0].rect.width
    ordered_text = []

    for page in doc:
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (0 if b[0] < width / 2 else 1, b[1]))
        ordered_text.extend([b[4].replace('\n', ' ') for b in blocks])

    full_content = " ".join(ordered_text)

    try:
        start_idx = full_content.find(section_name)
        if start_idx == -1:
            raise ValueError("Section not found in extracted content")

        curr_pos = sections_list.index(section_name)

        if curr_pos < len(sections_list) - 1:
            next_sec = sections_list[curr_pos + 1]
            end_idx = full_content.find(next_sec)
            result = full_content[start_idx:end_idx].strip()
        else:
            result = full_content[start_idx:].strip()
    except:
        result = "Content extraction failed."

    doc.close()
    return result


def extract_all_tables(pdf_bytes):
    """Identifies tables for visualization."""
    all_tables = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables({"vertical_strategy": "text", "horizontal_strategy": "text"})
            for table in tables:
                clean = [[str(c).strip() if c else "" for c in r] for r in table]
                if len(clean) > 1:
                    all_tables.append("\n".join([" | ".join(row) for row in clean]))
    return all_tables
