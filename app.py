import streamlit as st
import plotly.graph_objects as go
from core.document_processor import *
from core.llm_pipeline import *
from core.export_handler import *
import hashlib
import time
import os
import json
from concurrent.futures import ThreadPoolExecutor
from core.recommendation_engine import get_recommendations


st.set_page_config(page_title="ResearchXtract", layout="wide")

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
SECTION_CACHE_FILE = os.path.join(CACHE_DIR, "section_llm_cache.json")
RECOMMENDATION_CACHE_FILE = os.path.join(CACHE_DIR, "recommendation_cache.json")
PAPER_META_CACHE_FILE = os.path.join(CACHE_DIR, "paper_meta_cache.json")
DAY_SECONDS = 24 * 60 * 60

CACHE_POLICY = {
    SECTION_CACHE_FILE: {"ttl_days": 30, "max_entries": 500},
    RECOMMENDATION_CACHE_FILE: {"ttl_days": 14, "max_entries": 200},
    PAPER_META_CACHE_FILE: {"ttl_days": 60, "max_entries": 300},
}


def _load_json_cache(path):
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_json_cache(path, payload):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"CACHE save failed for {path}: {e}")


def _now_ts():
    return int(time.time())


def _is_wrapped_cache_entry(value):
    return isinstance(value, dict) and "_meta" in value and "data" in value


def _wrap_cache_entry(data, created_at=None, last_accessed_at=None, access_count=0):
    now = _now_ts()
    return {
        "_meta": {
            "created_at": int(created_at or now),
            "last_accessed_at": int(last_accessed_at or now),
            "access_count": int(access_count or 0),
        },
        "data": data,
    }


def _get_cache_data(cache_dict, key):
    entry = cache_dict.get(key)
    if entry is None:
        return None
    if _is_wrapped_cache_entry(entry):
        entry["_meta"]["last_accessed_at"] = _now_ts()
        entry["_meta"]["access_count"] = int(entry["_meta"].get("access_count", 0)) + 1
        return entry.get("data")
    # Backward compatibility with old plain entries.
    cache_dict[key] = _wrap_cache_entry(entry)
    return entry


def _set_cache_data(cache_dict, key, data):
    cache_dict[key] = _wrap_cache_entry(data)


def _prune_cache(cache_dict, ttl_days, max_entries):
    now = _now_ts()
    ttl_seconds = ttl_days * DAY_SECONDS

    # Normalize old plain entries first.
    for key, entry in list(cache_dict.items()):
        if not _is_wrapped_cache_entry(entry):
            cache_dict[key] = _wrap_cache_entry(entry)

    # TTL cleanup.
    for key, entry in list(cache_dict.items()):
        meta = entry.get("_meta", {})
        created_at = int(meta.get("created_at", now))
        if now - created_at > ttl_seconds:
            cache_dict.pop(key, None)

    # Max entries cleanup (LRU by last_accessed_at).
    if len(cache_dict) > max_entries:
        items = sorted(
            cache_dict.items(),
            key=lambda kv: int(kv[1].get("_meta", {}).get("last_accessed_at", 0))
        )
        remove_count = len(cache_dict) - max_entries
        for key, _ in items[:remove_count]:
            cache_dict.pop(key, None)


def _is_llm_failure_text(value):
    text = str(value or "").lower()
    return (
        "resource_exhausted" in text
        or "quota exceeded" in text
        or "summary extraction failed:" in text
        or "keyword extraction failed:" in text
        or "question answering failed:" in text
    )

# --- NEW: Highlighting Helper Function ---
def apply_highlights(text):
    """
    Finds '===' markers from the AI and wraps them in 
    Yellow HTML background tags for high visibility.
    """
    # Using <mark> for yellow highlight with rounded corners
    highlight_tag = "<mark style='background-color: #FFFF00; color: black; padding: 2px; border-radius: 4px; font-weight: bold;'>"
    
    # Simple regex to replace pairs of === with the HTML tags
    parts = text.split("===")
    highlighted_text = ""
    for i, part in enumerate(parts):
        if i % 2 == 1: # This is the text inside the markers
            highlighted_text += f"{highlight_tag}{part}</mark>"
        else:
            highlighted_text += part
    return highlighted_text

def safe_filename(text, fallback="researchxtract_report"):
    cleaned = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in str(text))
    cleaned = "_".join(cleaned.strip().split())
    return cleaned[:60] or fallback

def get_recommendation_abstract(pdf_bytes, sections):
    for sec in sections:
        if "ABSTRACT" in sec.upper():
            text = get_section_text(pdf_bytes, sec, sections)
            if text and text != "Content extraction failed.":
                return text[:4000]

    for sec in sections[:3]:
        text = get_section_text(pdf_bytes, sec, sections)
        if text and text != "Content extraction failed.":
            return text[:4000]

    return ""



# State management for Quota protection and UI persistence
if "active_sec" not in st.session_state: st.session_state.active_sec = None
if "cache" not in st.session_state: st.session_state.cache = {}
if "paper_meta" not in st.session_state: st.session_state.paper_meta = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "visual_cache" not in st.session_state: st.session_state.visual_cache = {}
if "current_pdf_id" not in st.session_state: st.session_state.current_pdf_id = None
if "report_pdf" not in st.session_state: st.session_state.report_pdf = None
if "recommendations" not in st.session_state: st.session_state.recommendations = []
if "recommendation_error" not in st.session_state: st.session_state.recommendation_error = None
if "recommendation_loaded" not in st.session_state: st.session_state.recommendation_loaded = False
if "recommendation_cache" not in st.session_state: st.session_state.recommendation_cache = {}
if "last_recommendation_click_ts" not in st.session_state: st.session_state.last_recommendation_click_ts = 0
if "persistent_section_cache" not in st.session_state:
    st.session_state.persistent_section_cache = _load_json_cache(SECTION_CACHE_FILE)
if "persistent_recommendation_cache" not in st.session_state:
    st.session_state.persistent_recommendation_cache = _load_json_cache(RECOMMENDATION_CACHE_FILE)
if "persistent_paper_meta_cache" not in st.session_state:
    st.session_state.persistent_paper_meta_cache = _load_json_cache(PAPER_META_CACHE_FILE)

# One-time prune/migrate on startup.
_prune_cache(
    st.session_state.persistent_section_cache,
    CACHE_POLICY[SECTION_CACHE_FILE]["ttl_days"],
    CACHE_POLICY[SECTION_CACHE_FILE]["max_entries"],
)
_save_json_cache(SECTION_CACHE_FILE, st.session_state.persistent_section_cache)
_prune_cache(
    st.session_state.persistent_recommendation_cache,
    CACHE_POLICY[RECOMMENDATION_CACHE_FILE]["ttl_days"],
    CACHE_POLICY[RECOMMENDATION_CACHE_FILE]["max_entries"],
)
_save_json_cache(RECOMMENDATION_CACHE_FILE, st.session_state.persistent_recommendation_cache)
_prune_cache(
    st.session_state.persistent_paper_meta_cache,
    CACHE_POLICY[PAPER_META_CACHE_FILE]["ttl_days"],
    CACHE_POLICY[PAPER_META_CACHE_FILE]["max_entries"],
)
_save_json_cache(PAPER_META_CACHE_FILE, st.session_state.persistent_paper_meta_cache)

st.title("ResearchXtract 🔬")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # FIX: Define pdf_bytes immediately after upload so it is available globally
    pdf_bytes = uploaded_file.getvalue()
    pdf_id = hashlib.md5(pdf_bytes).hexdigest()

    if st.session_state.current_pdf_id != pdf_id:
        st.session_state.current_pdf_id = pdf_id
        st.session_state.active_sec = None
        st.session_state.cache = {}
        st.session_state.paper_meta = None
        st.session_state.chat_history = []
        st.session_state.visual_cache = {}
        st.session_state.report_pdf = None
        st.session_state.recommendations = []
        st.session_state.recommendation_error = None
        st.session_state.recommendation_loaded = False


    
    # Metadata Logic: Only runs if not already in session state
    if st.session_state.paper_meta is None:
        with st.spinner("Initializing Paper Details..."):
            cached_meta = _get_cache_data(st.session_state.persistent_paper_meta_cache, pdf_id)
            if cached_meta:
                st.session_state.paper_meta = cached_meta
                print(f"CACHE paper_meta_hit=1 pdf_id='{pdf_id[:12]}...'")
            else:
                # Step 1: Visual Title & Regex Year Extraction
                raw_meta = extract_metadata(pdf_bytes)

                # Step 2: AI Author Extraction
                authors = extract_authors_with_ai(raw_meta['author_zone_text'])

                computed_meta = {
                    "title": raw_meta['title'],
                    "authors": authors,
                    "year": raw_meta.get('detected_year', "2025")
                }
                st.session_state.paper_meta = computed_meta
                _set_cache_data(st.session_state.persistent_paper_meta_cache, pdf_id, computed_meta)
                _prune_cache(
                    st.session_state.persistent_paper_meta_cache,
                    CACHE_POLICY[PAPER_META_CACHE_FILE]["ttl_days"],
                    CACHE_POLICY[PAPER_META_CACHE_FILE]["max_entries"],
                )
                _save_json_cache(PAPER_META_CACHE_FILE, st.session_state.persistent_paper_meta_cache)
                print(f"CACHE paper_meta_hit=0 pdf_id='{pdf_id[:12]}...'")

    # SECTION DETECTION: Uses spatial sorting to ensure correct order (Abstract -> References)
    sections = extract_sections(pdf_bytes)

    with st.sidebar:
        st.subheader("📑 Paper Details")
        st.info(f"**Title:** {st.session_state.paper_meta['title']}")
        st.write(f"**Authors:** {st.session_state.paper_meta['authors']}")
        st.write(f"**Year:** {st.session_state.paper_meta['year']}")
        st.divider()
        if st.session_state.cache:
            if st.session_state.report_pdf is None:
                st.session_state.report_pdf = build_report_pdf(
                    paper_meta=st.session_state.paper_meta,
                    section_cache=st.session_state.cache,
                    chat_history=st.session_state.chat_history
                )

            st.download_button(
                "📄 Download Full Report (PDF)",
                data=st.session_state.report_pdf,
                file_name=f"{safe_filename(st.session_state.paper_meta['title'])}_report.pdf",
                mime="application/pdf",
                width="stretch"
            )
        else:
            st.caption("Analyze a section to enable PDF export.")

        
        st.write("### Table of Contents")
        # Display sections in the order they were detected on the page
        for sec in sections:
            if st.button(sec, key=f"sidebar_{sec}", width="stretch"):
                st.session_state.active_sec = sec
    

    # --- Main Content Tabs (Now with 4 Tabs) ---
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🔑 Key Concepts", "📊 Visuals", "💬 Q&A"])
    
    if st.session_state.active_sec:
        sec = st.session_state.active_sec
        
        # Ensure section is analyzed and stored in cache
        if sec not in st.session_state.cache:
            with st.spinner(f"Analyzing {sec}..."):
                section_t0 = time.perf_counter()
                text = get_section_text(pdf_bytes, sec, sections)
                text_t1 = time.perf_counter()
                section_key = f"{st.session_state.current_pdf_id}:{sec}"
                persisted_section = _get_cache_data(st.session_state.persistent_section_cache, section_key)

                # Backward compatibility: try legacy hash-based key once and migrate.
                if not persisted_section:
                    legacy_section_key = f"{st.session_state.current_pdf_id}:{sec}:{hashlib.md5(text.encode('utf-8')).hexdigest()}"
                    legacy_cached = _get_cache_data(st.session_state.persistent_section_cache, legacy_section_key)
                    if legacy_cached:
                        persisted_section = legacy_cached
                        _set_cache_data(st.session_state.persistent_section_cache, section_key, legacy_cached)
                        st.session_state.persistent_section_cache.pop(legacy_section_key, None)
                        _save_json_cache(SECTION_CACHE_FILE, st.session_state.persistent_section_cache)

                if (
                    persisted_section
                    and not _is_llm_failure_text(persisted_section.get("summary", ""))
                    and not _is_llm_failure_text(persisted_section.get("keywords", ""))
                ):
                    summary = persisted_section.get("summary", "")
                    keywords = persisted_section.get("keywords", "")
                    summary_t2 = time.perf_counter()
                    keywords_t3 = summary_t2
                    print(f"CACHE section_hit=1 key='{section_key[:36]}...'")
                else:
                    # Run independent LLM calls in parallel to reduce latency.
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        summary_future = executor.submit(summarize_section, sec, text)
                        keywords_future = executor.submit(extract_keywords, text)
                        summary = summary_future.result()
                        summary_t2 = time.perf_counter()
                        keywords = keywords_future.result()
                        keywords_t3 = time.perf_counter()
                    if not _is_llm_failure_text(summary) and not _is_llm_failure_text(keywords):
                        _set_cache_data(st.session_state.persistent_section_cache, section_key, {
                            "summary": summary,
                            "keywords": keywords
                        })
                        _prune_cache(
                            st.session_state.persistent_section_cache,
                            CACHE_POLICY[SECTION_CACHE_FILE]["ttl_days"],
                            CACHE_POLICY[SECTION_CACHE_FILE]["max_entries"],
                        )
                        _save_json_cache(SECTION_CACHE_FILE, st.session_state.persistent_section_cache)
                    else:
                        # Avoid polluting persistent cache with transient API failures.
                        st.session_state.persistent_section_cache.pop(section_key, None)
                    print(f"CACHE section_hit=0 key='{section_key[:36]}...'")
                st.session_state.cache[sec] = {
                    "summary": summary,
                    "keywords": keywords
                }
                print(
                    f"TIMING section='{sec}' "
                    f"extract_ms={(text_t1 - section_t0) * 1000:.0f} "
                    f"summary_ms={(summary_t2 - text_t1) * 1000:.0f} "
                    f"keywords_ms={(keywords_t3 - summary_t2) * 1000:.0f} "
                    f"total_ms={(keywords_t3 - section_t0) * 1000:.0f}"
                )
                st.session_state.report_pdf = None
                st.rerun()

        # --- TAB 1: Summary (With Highlighting) ---
        with tab1:
            raw_summary = st.session_state.cache[sec]["summary"]
            st.markdown(apply_highlights(raw_summary), unsafe_allow_html=True)
            
        # --- TAB 2: Key Concepts ---
        with tab2: 
            st.markdown(st.session_state.cache[sec]["keywords"])

        # --- TAB 3: Visuals ---
        with tab3:
            st.subheader("Automated Performance Visuals")

            if "tables" not in st.session_state.visual_cache:
                st.session_state.visual_cache["tables"] = extract_all_tables(pdf_bytes)

            if "charts" not in st.session_state.visual_cache:
                if st.button("📊 Generate Visuals", key="generate_visuals", width="stretch"):
                    charts = []
                    for t in st.session_state.visual_cache["tables"]:
                        g = generate_graph_data(t)
                        if isinstance(g, dict) and "labels" in g and "series" in g:
                            charts.append(g)
                    st.session_state.visual_cache["charts"] = charts
                    st.rerun()
            else:
                for g in st.session_state.visual_cache["charts"]:
                    fig = go.Figure([go.Bar(x=g["labels"], y=v, name=k) for k, v in g["series"].items()])
                    st.plotly_chart(fig, width="stretch")

        # --- TAB 4: Q&A Session ---
        with tab4:
            st.subheader("Ask anything about this paper")
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
                
            user_query = st.chat_input("Ask about the WhatsApp Gateway or Encryption...")
            
            if user_query:
                with st.spinner("Searching paper..."):
                    context_text = "\n".join([get_section_text(pdf_bytes, s, sections) for s in sections[:5]])
                    answer = answer_paper_question(user_query, context_text)
                    st.session_state.chat_history.append({"q": user_query, "a": answer})
                    st.session_state.report_pdf = None

            for chat in reversed(st.session_state.chat_history):
                st.write(f"**🧐 Question:** {chat['q']}")
                st.write(f"**🤖 Answer:** {chat['a']}")

        st.divider()
        st.markdown("<div id='recommended-papers'></div>", unsafe_allow_html=True)

        st.subheader("📚 Recommended Papers")
        st.caption("Discover related research based on the current paper.")

        if st.button("Find Related Papers", key="find_related_papers", width="stretch"):
            now_ts = time.time()
            if now_ts - st.session_state.last_recommendation_click_ts < 2.0:
                st.info("Recommendation request is already in progress. Please wait a moment.")
                st.stop()
            st.session_state.last_recommendation_click_ts = now_ts
            with st.spinner("Finding related research papers..."):
                cache_key = None
                rec_t0 = time.perf_counter()
                try:
                    abstract_text = get_recommendation_abstract(pdf_bytes, sections)
                    cache_key = f"{st.session_state.current_pdf_id}:{hashlib.md5(abstract_text.encode('utf-8')).hexdigest()}"
                    cached = st.session_state.recommendation_cache.get(cache_key)
                    if not cached:
                        cached = _get_cache_data(st.session_state.persistent_recommendation_cache, cache_key)

                    if cached:
                        st.session_state.recommendations = cached.get("recommendations", [])
                        st.session_state.recommendation_error = cached.get("error")
                        st.session_state.recommendation_cache[cache_key] = cached
                        rec_t1 = time.perf_counter()
                        print(
                            f"TIMING recommendations cache_hit=1 "
                            f"total_ms={(rec_t1 - rec_t0) * 1000:.0f}"
                        )
                    else:
                        result = get_recommendations(
                            title=st.session_state.paper_meta["title"],
                            abstract=abstract_text,
                            top_k=5,
                            fetch_limit=15
                        )
                        st.session_state.recommendations = result
                        st.session_state.recommendation_error = None
                        st.session_state.recommendation_cache[cache_key] = {
                            "recommendations": result,
                            "error": None
                        }
                        _set_cache_data(st.session_state.persistent_recommendation_cache, cache_key, {
                            "recommendations": result,
                            "error": None
                        })
                        _prune_cache(
                            st.session_state.persistent_recommendation_cache,
                            CACHE_POLICY[RECOMMENDATION_CACHE_FILE]["ttl_days"],
                            CACHE_POLICY[RECOMMENDATION_CACHE_FILE]["max_entries"],
                        )
                        _save_json_cache(RECOMMENDATION_CACHE_FILE, st.session_state.persistent_recommendation_cache)
                        rec_t1 = time.perf_counter()
                        print(
                            f"TIMING recommendations cache_hit=0 "
                            f"result_count={len(result)} "
                            f"total_ms={(rec_t1 - rec_t0) * 1000:.0f}"
                        )

                    st.session_state.recommendation_loaded = True
                except Exception as e:
                    st.session_state.recommendations = []
                    st.session_state.recommendation_error = str(e)
                    if cache_key:
                        st.session_state.recommendation_cache[cache_key] = {
                            "recommendations": [],
                            "error": str(e)
                        }
                        _set_cache_data(st.session_state.persistent_recommendation_cache, cache_key, {
                            "recommendations": [],
                            "error": str(e)
                        })
                        _prune_cache(
                            st.session_state.persistent_recommendation_cache,
                            CACHE_POLICY[RECOMMENDATION_CACHE_FILE]["ttl_days"],
                            CACHE_POLICY[RECOMMENDATION_CACHE_FILE]["max_entries"],
                        )
                        _save_json_cache(RECOMMENDATION_CACHE_FILE, st.session_state.persistent_recommendation_cache)
                    st.session_state.recommendation_loaded = True

        if st.session_state.recommendation_error:
            st.error(f"Recommendation loading failed: {st.session_state.recommendation_error}")
        elif st.session_state.recommendation_loaded and not st.session_state.recommendations:
            st.info("No related papers with accessible links were found.")

        if st.session_state.recommendations:
            for idx, paper in enumerate(st.session_state.recommendations, start=1):
                title = paper.get("title") or f"Recommended Paper {idx}"
                authors = paper.get("authors") or "Unknown authors"
                year = paper.get("year") or "Unknown"
                source = paper.get("source") or "Unknown source"
                retrieved_from = paper.get("retrieved_from") or "Unknown"
                snippet = paper.get("snippet") or "No abstract snippet available."
                reason = paper.get("reason") or "Recommended based on topic similarity."
                accessible_url = paper.get("accessible_url")
                pdf_url = paper.get("pdf_url")
                doi_url = paper.get("doi_url")

                with st.expander(f"{idx}. {title}"):
                    st.write(f"**Authors:** {authors}")
                    st.write(f"**Year:** {year}")
                    st.write(f"**Source:** {source}")
                    st.write(f"**Retrieved from:** {retrieved_from}")
                    st.write(f"**Snippet:** {snippet}")
                    st.write(f"**Why recommended:** {reason}")

                    if accessible_url:
                        st.markdown(f"[Open Paper]({accessible_url})")

                    if pdf_url and pdf_url != accessible_url:
                        st.markdown(f"[Open PDF]({pdf_url})")

                    if doi_url and doi_url != accessible_url:
                        st.markdown(f"[View DOI]({doi_url})")

       