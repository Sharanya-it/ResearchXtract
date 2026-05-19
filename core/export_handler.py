import re
import fitz


def _clean_text(value):
    if not value:
        return ""
    text = str(value)
    text = text.replace("===", "")
    text = text.replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def _split_critical_points(text):
    if not text:
        return "", []

    parts = str(text).split("===")
    normal_parts = []
    critical_points = []

    for i, part in enumerate(parts):
        cleaned = _clean_text(part)
        if not cleaned:
            continue

        if i % 2 == 1:
            critical_points.append(cleaned)
        else:
            normal_parts.append(cleaned)

    normal_text = " ".join(normal_parts).strip()
    return normal_text, critical_points

def _split_key_concepts(text):
    if not text:
        return []

    raw_lines = str(text).splitlines()
    items = []
    current = ""

    for line in raw_lines:
        cleaned = line.strip()
        if not cleaned:
            continue

        if re.match(r"^\d+\.", cleaned):
            if current:
                items.append(current.strip())
            current = cleaned
        else:
            if current:
                current += " " + cleaned
            else:
                current = cleaned

    if current:
        items.append(current.strip())

    return items

def _parse_key_concept(item):
    cleaned = _clean_text(item)

    match = re.match(r"^(\d+\.)\s*(.*)$", cleaned)
    if match:
        number = match.group(1)
        body = match.group(2).strip()
    else:
        number = ""
        body = cleaned

    if "Definition:" in body:
        title, definition = body.split("Definition:", 1)
    else:
        title, definition = body, ""

    return number, title.strip(" :*-"), definition.strip()


def _write_wrapped_text(page, text, x, y, max_width, font_size=11, line_gap=16, fontname="helv"):
    words = _clean_text(text).split()
    if not words:
        return page, y

    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        text_width = fitz.get_text_length(test_line, fontsize=font_size, fontname=fontname)

        if text_width <= max_width:
            line = test_line
        else:
            if y > 720:
                page = page.parent.new_page()
                y = 60
            page.insert_text((x, y), line, fontsize=font_size, fontname=fontname)
            y += line_gap
            line = word

    if line:
        if y > 720:
            page = page.parent.new_page()
            y = 60
        page.insert_text((x, y), line, fontsize=font_size, fontname=fontname)
        y += line_gap

    return page, y



def _start_new_page(doc):
    page = doc.new_page()
    return page, 60

def _ensure_space(page, y, needed_height):
    if y + needed_height > 720:
        return page.parent.new_page(), 60
    return page, y



def build_report_pdf(paper_meta, section_cache, chat_history):
    doc = fitz.open()
    page, y = _start_new_page(doc)
    left_margin = 60
    content_indent = 78
    max_width = 452


    page.insert_text((left_margin, y), "ResearchXtract Report", fontsize=20, fontname="hebo")
    y += 30

    title = _clean_text(paper_meta.get("title", "Untitled Paper"))
    authors = _clean_text(paper_meta.get("authors", "Unknown Authors"))
    year = _clean_text(paper_meta.get("year", "Unknown Year"))

    page, y = _write_wrapped_text(page, f"Title: {title}", content_indent, y, max_width, font_size=12)
    y += 4
    page, y = _write_wrapped_text(page, f"Authors: {authors}", content_indent, y, max_width, font_size=12)
    y += 4
    page, y = _write_wrapped_text(page, f"Year: {year}", content_indent, y, max_width, font_size=12)
    y += 14

    y += 10
    page, y = _ensure_space(page, y, 30)
    page.insert_text((left_margin, y), "Analyzed Sections", fontsize=16, fontname="hebo")
    y += 24

    for section_name, content in section_cache.items():
        if y > 720:
            page, y = _start_new_page(doc)
        
        page, y = _ensure_space(page, y, 36)
        page.insert_text((left_margin, y), _clean_text(section_name), fontsize=14, fontname="hebo")
        y += 24

        raw_summary = content.get("summary", "")
        summary, critical_points = _split_critical_points(raw_summary)

        keywords = content.get("keywords", "")
        keyword_items = _split_key_concepts(keywords)


        if summary or critical_points:
            page, y = _ensure_space(page, y, 28)
            page.insert_text((content_indent, y), "Summary", fontsize=12, fontname="hebo")
            y += 18

            if summary:
                page, y = _write_wrapped_text(page, summary, content_indent, y, max_width, font_size=11)
                y += 10

            if critical_points:
                page, y = _ensure_space(page, y, 28)
                page.insert_text((content_indent, y), "Critical Points", fontsize=12, fontname="hebo")
                y += 18

                for point in critical_points:
                    page, y = _write_wrapped_text(page, f"- {point}", content_indent, y, max_width, font_size=11)
                    y += 8

                y += 6

            if keyword_items:
                page, y = _ensure_space(page, y, 28)
                page.insert_text((content_indent, y), "Key Concepts", fontsize=12, fontname="hebo")
                y += 18

                for item in keyword_items:
                    number, title, definition = _parse_key_concept(item)

                    if title:
                        page, y = _ensure_space(page, y, 24)
                        heading_text = f"{number} {title}".strip()
                        page.insert_text((content_indent, y), heading_text, fontsize=11, fontname="hebo")
                        y += 16

                    if definition:
                        page, y = _ensure_space(page, y, 24)
                        page.insert_text((content_indent, y), "Definition:", fontsize=11, fontname="hebo")
                        y += 16
                        page, y = _write_wrapped_text(page, definition, content_indent + 12, y, max_width - 12, font_size=11)
 
                    y += 10





    if chat_history:
        if y > 680:
            page, y = _start_new_page(doc)
        
        page, y = _ensure_space(page, y, 30)
        page.insert_text((left_margin, y), "Q&A History", fontsize=16, fontname="hebo")
        y += 24
        for chat in chat_history:
            if y > 720:
                page, y = _start_new_page(doc)

            question = _clean_text(chat.get("q", ""))
            answer = _clean_text(chat.get("a", ""))

            page, y = _ensure_space(page, y, 28)
            page.insert_text((content_indent, y), "Question", fontsize=12, fontname="hebo")
            y += 18
            page, y = _write_wrapped_text(page, question, content_indent, y, max_width, font_size=11)
            y += 8

            page, y = _ensure_space(page, y, 28)
            page.insert_text((content_indent, y), "Answer", fontsize=12, fontname="hebo")
            y += 18
            page, y = _write_wrapped_text(page, answer, content_indent, y, max_width, font_size=11)
            y += 18


    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

