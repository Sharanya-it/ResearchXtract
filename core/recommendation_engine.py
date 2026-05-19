import re
import time
import httpx
from difflib import SequenceMatcher
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from mistralai.client import Mistral

load_dotenv()

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

mistral_api_key = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=mistral_api_key) if mistral_api_key else None

semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# Semantic Scholar global throttle: keep >=1.1s between calls.
_last_s2_call = 0.0
_S2_MIN_INTERVAL = 1.1


# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

QUERY_GEN_PROMPT = """You are an academic search expert specializing in finding related research papers.

Given this paper's title and abstract, generate exactly 3 search queries for academic paper search.

Goal:
Generate queries that retrieve papers solving the SAME TASK or RESEARCH PROBLEM first.
Do not over-focus on generic method words alone.

Rules:
- Each query must be 4-7 keywords, not a full sentence
- Prioritize the paper's core task, problem, or application domain
- Include specific technical terms only if they help distinguish the task
- Avoid generic-only method queries such as:
  "transformer architecture self attention"
  "neural network architecture"
  "attention model deep learning"
- Prefer queries that combine:
  1. the task/problem
  2. the domain
  3. the distinctive method if useful
- Do NOT copy the exact paper title
- Do NOT use verbs, articles, or filler words
- Avoid overly broad terms like:
  deep learning, neural network, model, architecture, system, framework
  unless paired with the actual task

Return:
- Query 1: same task/problem
- Query 2: same domain + similar objective
- Query 3: same task + distinctive method

Return ONLY a valid JSON array of exactly 3 strings.
No explanation. No markdown. Only the JSON array.

Title: {title}
Abstract: {abstract}
"""


RERANK_PROMPT = """You are ranking academic papers by relevance to a source paper.

SOURCE PAPER:
Title: {title}
Abstract: {abstract}

TASK:
Rank the candidate papers by how closely they are related to the SOURCE PAPER.

Use this priority order:
1. HIGHEST: Papers solving the same core research problem or task
2. HIGH: Papers in the same domain/application that address a very similar goal
3. MEDIUM: Papers using a similar method but on a different task
4. LOW: Papers sharing only general method words or broad field vocabulary

Very important:
- Prefer SAME TASK over SAME METHOD
- Prefer SAME DOMAIN over generic transformer/attention similarity
- If the source paper is NLP, machine translation, language modeling, or sequence transduction,
  rank computer vision, image captioning, segmentation, and pixel prediction papers LOWER unless
  the candidate is clearly about the same task
- If the source paper is in computer vision, rank pure NLP papers LOWER unless the task matches closely
- Do not overvalue words like attention, transformer, network, architecture, model, or learning by themselves

RED FLAGS — rank these LAST:
- Different domain but same buzzwords
- Papers sharing only “attention”, “transformer”, or “neural network”
- Survey papers unless the source is also a survey
- Papers with broad topical overlap but a different research objective

CANDIDATE PAPERS:
{candidates}

Return ONLY a JSON array of candidate indices (0-based) in order from most to least relevant.
Example: [3, 0, 2, 4, 1]
No explanation. No markdown. Only the JSON array.
"""

TASK_PROFILE_PROMPT = """You are extracting a compact research task profile.

Given the paper title and abstract, identify:
1) Primary domain (one label only): NLP, COMPUTER_VISION, SPEECH_AUDIO, RECOMMENDER_SYSTEMS, SECURITY_PRIVACY, SYSTEMS_NETWORKING, HEALTHCARE_BIOMEDICAL, ROBOTICS, MULTIMODAL, OTHER
2) Core task/problem in one short phrase
3) 6-10 important task/domain terms (lowercase, no stopwords)
4) Optional domains to avoid during retrieval (labels from the same domain list)

Return ONLY valid JSON in this schema:
{
  "domain": "NLP",
  "task": "machine translation quality estimation",
  "task_terms": ["translation", "quality", "estimation"],
  "avoid_domains": ["COMPUTER_VISION"]
}

No markdown. No explanations.

Title: {title}
Abstract: {abstract}
"""



# ---------------------------------------------------------------------------
# TEXT UTILITIES
# ---------------------------------------------------------------------------

def normalize_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def similarity_score(a, b):
    return SequenceMatcher(
        None,
        normalize_text(a).lower(),
        normalize_text(b).lower()
    ).ratio()


def keyword_overlap_score(source_text, candidate_text):
    source_words = set(re.findall(r"[a-zA-Z]{4,}", normalize_text(source_text).lower()))
    candidate_words = set(re.findall(r"[a-zA-Z]{4,}", normalize_text(candidate_text).lower()))
    if not source_words or not candidate_words:
        return 0
    return len(source_words & candidate_words) / max(len(source_words), 1)


_STOPWORDS = {
    "this", "that", "with", "from", "have", "into", "their", "using",
    "based", "paper", "study", "approach", "method", "results",
    "system", "analysis", "model", "research", "proposed", "developed",
    "application", "applications", "work", "works", "data",
    "these", "those", "they", "them", "being", "often", "across",
    "through", "within", "where", "which", "while", "such", "than",
    "also", "can", "could", "would", "should", "about", "there",
    "here", "been", "were", "was", "are", "our", "your", "many",
    "most", "more", "less", "other", "some", "each", "both",
    "leveraging", "campus", "management",
}

_GENERIC_ML_TERMS = {
    "learning", "deep", "neural", "network", "networks", "transformer",
    "transformers", "attention", "architecture", "architectures", "model",
    "models", "framework", "frameworks", "approach", "approaches", "method",
    "methods", "performance", "accuracy", "dataset", "datasets",
}

_DOMAIN_KEYWORDS = {
    "NLP": {
        "text", "language", "translation", "summarization", "token", "tokens",
        "dialogue", "question", "answering", "qa", "corpus", "syntax",
    },
    "COMPUTER_VISION": {
        "image", "images", "vision", "segmentation", "detection", "pixel",
        "object", "video", "cnn", "captioning",
    },
    "SPEECH_AUDIO": {
        "speech", "audio", "asr", "transcription", "acoustic", "voice",
    },
    "RECOMMENDER_SYSTEMS": {
        "recommendation", "recommender", "ranking", "personalization",
        "collaborative", "clickthrough",
    },
    "SECURITY_PRIVACY": {
        "encryption", "cipher", "privacy", "malware", "attack", "secure",
        "cryptography", "threat",
    },
    "SYSTEMS_NETWORKING": {
        "networking", "protocol", "latency", "throughput", "distributed",
        "system", "systems", "gateway",
    },
    "HEALTHCARE_BIOMEDICAL": {
        "clinical", "patient", "medical", "biomedical", "diagnosis",
        "disease", "healthcare",
    },
    "ROBOTICS": {
        "robot", "robotics", "navigation", "control", "manipulation",
    },
    "MULTIMODAL": {
        "multimodal", "vision-language", "cross-modal", "audio-visual",
    },
}

_DOMAIN_CONFLICT_SIGNALS = {
    "NLP": {
        "eeg", "electroencephalogram", "brain-computer", "electromyography",
        "ecg", "electrocardiogram", "fmri", "neuroimaging", "pixel",
        "segmentation", "object detection", "image classification",
        "bounding box", "convolutional", "medical imaging", "radiology",
        "histology", "remote sensing", "satellite imagery", "optical flow",
        "pose estimation",
    },
    "COMPUTER_VISION": {
        "machine translation", "summarization", "question answering",
        "named entity recognition", "sentiment analysis", "parsing",
        "coreference", "text classification", "dialogue", "tokenization",
        "language model", "eeg", "electroencephalogram", "ecg",
    },
    "SPEECH_AUDIO": {
        "pixel", "segmentation", "object detection", "image classification",
        "eeg", "electroencephalogram", "brain-computer",
        "machine translation", "sentiment analysis",
    },
}


def _content_terms(text):
    words = re.findall(r"[a-zA-Z]{4,}", normalize_text(text).lower())
    return {
        w for w in words
        if w not in _STOPWORDS and w not in _GENERIC_ML_TERMS
    }


def _infer_domain(text):
    terms = _content_terms(text)
    if not terms:
        return "OTHER", 0

    scores = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        scores[domain] = len(terms & keywords)

    best_domain = max(scores, key=scores.get)
    best_score = scores.get(best_domain, 0)
    if best_score == 0:
        return "OTHER", 0
    return best_domain, best_score


def extract_topic_terms(title, abstract):
    title_words = re.findall(r"[a-zA-Z]{4,}", normalize_text(title).lower())
    abstract_words = re.findall(r"[a-zA-Z]{4,}", normalize_text(abstract).lower())

    title_terms = []
    for word in title_words:
        if word not in _STOPWORDS and word not in _GENERIC_ML_TERMS and word not in title_terms:
            title_terms.append(word)

    abstract_frequency = {}
    for word in abstract_words:
        if word in _STOPWORDS or word in _GENERIC_ML_TERMS or word in title_terms:
            continue
        abstract_frequency[word] = abstract_frequency.get(word, 0) + 1

    abstract_terms = [
        term for term, _ in sorted(
            abstract_frequency.items(), key=lambda x: (-x[1], x[0])
        )
    ]

    return (title_terms + abstract_terms)[:10]


def _s2_get(client, url, params):
    global _last_s2_call
    elapsed = time.monotonic() - _last_s2_call
    if elapsed < _S2_MIN_INTERVAL:
        time.sleep(_S2_MIN_INTERVAL - elapsed)
    headers = {"x-api-key": semantic_scholar_api_key} if semantic_scholar_api_key else {}
    response = client.get(url, params=params, headers=headers)
    _last_s2_call = time.monotonic()
    response.raise_for_status()
    return response


def extract_key_phrases(title, abstract):
    title_text = normalize_text(title).lower()
    abstract_text = normalize_text(abstract).lower()

    def collect_phrases(text):
        words = re.findall(r"[a-zA-Z]{4,}", text)
        phrases = []
        for i in range(len(words) - 1):
            first, second = words[i], words[i + 1]
            if first in _STOPWORDS or second in _STOPWORDS:
                continue
            phrase = f"{first} {second}"
            if phrase not in phrases:
                phrases.append(phrase)
        return phrases

    title_phrases = collect_phrases(title_text)
    abstract_phrases = collect_phrases(abstract_text)
    return (title_phrases + abstract_phrases)[:8]


def extract_query_keywords(title, abstract):
    text = f"{title} {abstract}".lower()
    words = re.findall(r"[a-zA-Z]{4,}", text)
    filtered = [w for w in words if w not in _STOPWORDS]
    seen = []
    for word in filtered:
        if word not in seen:
            seen.append(word)
    return seen[:6]


def build_search_query(title, abstract):
    key_phrases = extract_key_phrases(title, abstract)
    topic_terms = extract_topic_terms(title, abstract)

    query_parts = []
    seen_tokens = set()

    for phrase in key_phrases[:3]:
        cleaned_phrase = normalize_text(phrase).lower()
        phrase_tokens = tuple(cleaned_phrase.split())
        if cleaned_phrase and phrase_tokens not in seen_tokens:
            query_parts.append(cleaned_phrase)
            seen_tokens.add(phrase_tokens)

    used_words = set()
    for part in query_parts:
        used_words.update(part.split())

    for term in topic_terms[:5]:
        cleaned_term = normalize_text(term).lower()
        if cleaned_term and cleaned_term not in used_words:
            query_parts.append(cleaned_term)
            used_words.add(cleaned_term)

    return " ".join(query_parts) if query_parts else "research paper"


def build_fallback_query(title, abstract):
    topic_terms = extract_topic_terms(title, abstract)
    return " ".join(topic_terms[:4]) if topic_terms else "research paper"


# ---------------------------------------------------------------------------
# LLM: QUERY GENERATION
# ---------------------------------------------------------------------------

def generate_queries_with_gemini(title, abstract):
    prompt = QUERY_GEN_PROMPT.format(title=title, abstract=abstract)
    response = gemini_llm.invoke(prompt)
    raw = response.content.strip().replace("```json", "").replace("```", "").strip()
    queries = json.loads(raw)
    if not isinstance(queries, list):
        raise ValueError("Gemini did not return a list of queries.")
    return [normalize_text(q) for q in queries[:3] if normalize_text(q)]


def generate_queries_with_mistral(title, abstract):
    if not mistral_client:
        raise ValueError("Mistral client is not configured.")
    prompt = QUERY_GEN_PROMPT.format(title=title, abstract=abstract)
    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    queries = json.loads(raw)
    if not isinstance(queries, list):
        raise ValueError("Mistral did not return a list of queries.")
    return [normalize_text(q) for q in queries[:3] if normalize_text(q)]


def generate_queries_with_llm(title, abstract):
    try:
        return generate_queries_with_gemini(title, abstract)
    except Exception:
        pass
    try:
        return generate_queries_with_mistral(title, abstract)
    except Exception:
        pass
    return None


def extract_task_profile_with_gemini(title, abstract):
    prompt = TASK_PROFILE_PROMPT.format(title=title, abstract=abstract)
    response = gemini_llm.invoke(prompt)
    raw = response.content.strip().replace("```json", "").replace("```", "").strip()
    profile = json.loads(raw)
    if not isinstance(profile, dict):
        raise ValueError("Gemini did not return a valid task profile object.")
    return profile


def extract_task_profile_with_mistral(title, abstract):
    if not mistral_client:
        raise ValueError("Mistral client is not configured.")
    prompt = TASK_PROFILE_PROMPT.format(title=title, abstract=abstract)
    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    profile = json.loads(raw)
    if not isinstance(profile, dict):
        raise ValueError("Mistral did not return a valid task profile object.")
    return profile


def extract_task_profile_with_llm(title, abstract):
    try:
        profile = extract_task_profile_with_gemini(title, abstract)
    except Exception:
        profile = None

    if profile is None:
        try:
            profile = extract_task_profile_with_mistral(title, abstract)
        except Exception:
            profile = None

    if not profile:
        inferred_domain, _ = _infer_domain(f"{title} {abstract}")
        return {
            "domain": inferred_domain,
            "task": normalize_text(title)[:120] or "research task",
            "task_terms": extract_topic_terms(title, abstract)[:8],
            "avoid_domains": [],
        }

    domain = normalize_text(profile.get("domain", "OTHER")).upper()
    task = normalize_text(profile.get("task", "")) or "research task"
    task_terms = [
        normalize_text(t).lower()
        for t in profile.get("task_terms", [])
        if normalize_text(t)
    ]
    avoid_domains = [
        normalize_text(d).upper()
        for d in profile.get("avoid_domains", [])
        if normalize_text(d)
    ]

    if domain not in _DOMAIN_KEYWORDS and domain != "OTHER":
        domain = "OTHER"

    if not task_terms:
        task_terms = extract_topic_terms(title, abstract)[:8]

    return {
        "domain": domain,
        "task": task,
        "task_terms": task_terms[:10],
        "avoid_domains": avoid_domains,
    }


def deduplicate_queries(queries):
    """
    Remove near-duplicate queries before hitting OpenAlex.
    Prevents wasting fetch quota on queries that produce the same result set.
    """
    unique = []
    for q in queries:
        q_norm = normalize_text(q).lower()
        is_dup = any(
            SequenceMatcher(None, q_norm, normalize_text(u).lower()).ratio() > 0.65
            for u in unique
        )
        if not is_dup:
            unique.append(q)
    return unique


# ---------------------------------------------------------------------------
# LLM: RERANKING
# ---------------------------------------------------------------------------

def format_candidates_for_rerank(candidates):
    lines = []
    for idx, paper in enumerate(candidates):
        title = normalize_text(paper.get("title", ""))
        # Cap snippet length to reduce rerank prompt latency.
        snippet = normalize_text(paper.get("snippet", ""))[:320]
        source = normalize_text(paper.get("source", ""))
        lines.append(
            f"{idx}. Title: {title}\n"
            f"   Source: {source}\n"
            f"   Abstract: {snippet}\n"
        )
    return "\n".join(lines)


def rerank_with_gemini(title, abstract, candidates):
    candidate_block = format_candidates_for_rerank(candidates)
    prompt = RERANK_PROMPT.format(
        title=title,
        abstract=abstract,
        candidates=candidate_block
    )
    response = gemini_llm.invoke(prompt)
    raw = response.content.strip().replace("```json", "").replace("```", "").strip()
    ranked_indices = json.loads(raw)
    if not isinstance(ranked_indices, list):
        raise ValueError("Gemini did not return a list of ranked indices.")
    return [int(i) for i in ranked_indices]


def rerank_with_mistral(title, abstract, candidates):
    if not mistral_client:
        raise ValueError("Mistral client is not configured.")
    candidate_block = format_candidates_for_rerank(candidates)
    prompt = RERANK_PROMPT.format(
        title=title,
        abstract=abstract,
        candidates=candidate_block
    )
    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    ranked_indices = json.loads(raw)
    if not isinstance(ranked_indices, list):
        raise ValueError("Mistral did not return a list of ranked indices.")
    return [int(i) for i in ranked_indices]


def rerank_candidates_with_llm(title, abstract, candidates):
    try:
        return rerank_with_gemini(title, abstract, candidates)
    except Exception:
        pass
    try:
        return rerank_with_mistral(title, abstract, candidates)
    except Exception:
        pass
    return None


def normalize_semantic_scholar_candidate(paper):
    authors = ", ".join(
        author.get("name", "")
        for author in paper.get("authors", [])
        if author.get("name")
    )

    title = normalize_text(paper.get("title"))
    abstract = normalize_text(paper.get("abstract"))
    has_abstract = bool(abstract)

    open_access_pdf = paper.get("openAccessPdf") or {}
    pdf_url = open_access_pdf.get("url")
    paper_url = paper.get("url")

    external_ids = paper.get("externalIds") or {}
    doi = external_ids.get("DOI")
    doi_url = f"https://doi.org/{doi}" if doi else None
    accessible_url = pdf_url or paper_url or doi_url

    snippet_clean = abstract[:800] if abstract else title
    source_name = normalize_text(paper.get("venue")) or "Semantic Scholar"

    return {
    "title": title,
    "authors": authors or "Unknown authors",
    "year": paper.get("year") or "Unknown",
    "snippet": snippet_clean,
    "has_abstract": has_abstract,
    "source": source_name,
    "retrieved_from": "Semantic Scholar",
    "paper_url": paper_url,
    "pdf_url": pdf_url,
    "doi_url": doi_url,
    "accessible_url": accessible_url,
    "priority_boost": 2,
}



def search_semantic_scholar(query, limit=15, client=None):
    if not semantic_scholar_api_key:
        raise ValueError("Semantic Scholar API key is not configured.")

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,authors,url,venue,openAccessPdf,externalIds",
    }
    if client is None:
        with httpx.Client(timeout=20.0) as local_client:
            response = _s2_get(local_client, url, params)
            response.raise_for_status()
            data = response.json()
    else:
        response = _s2_get(client, url, params)
        response.raise_for_status()
        data = response.json()

    return data.get("data", [])

# ---------------------------------------------------------------------------
# OPENALEX FETCH
# ---------------------------------------------------------------------------

def search_openalex(query, limit=15, client=None):
    """
    BM25 full-text search on OpenAlex.
    sort=relevance_score:desc prevents citation-count bias from
    burying niche-but-relevant papers under prolific tangential ones.
    """
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": limit,
        "sort": "relevance_score:desc",
    }
    if client is None:
        with httpx.Client(timeout=20.0) as local_client:
            response = local_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
    else:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    return data.get("results", [])


def search_openalex_batch(queries, limit=15, max_workers=4):
    """
    Run OpenAlex keyword queries in parallel without changing retrieval logic.
    Returns a flat list of raw OpenAlex paper objects.
    """
    if not queries:
        return []

    results = []
    if max_workers <= 1:
        with httpx.Client(timeout=20.0) as client:
            for query in queries:
                try:
                    papers = search_openalex(query, limit=limit, client=client)
                    results.extend(papers)
                except Exception as e:
                    print(f"DEBUG openalex_sequential_failed query='{query}': {e}")
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(search_openalex, query, limit): query
                for query in queries
            }
            for future in as_completed(future_map):
                query = future_map[future]
                try:
                    papers = future.result()
                    results.extend(papers)
                except Exception as e:
                    print(f"DEBUG openalex_parallel_failed query='{query}': {e}")
    return results


def fetch_openalex_related(title, client=None):
    """
    Attempt to locate the source paper itself in OpenAlex and retrieve
    its curated related_works list (computed from co-citation and
    bibliographic coupling — far more precise than BM25 queries).

    Returns a list of raw OpenAlex work objects, or [] if not found.
    """
    url = "https://api.openalex.org/works"

    # Step 1: find the source paper
    params = {"search": title, "per-page": 3, "sort": "relevance_score:desc"}
    try:
        if client is None:
            with httpx.Client(timeout=20.0) as local_client:
                response = local_client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
        else:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return []

    results = data.get("results", [])
    if not results:
        return []

    # Pick the closest title match rather than blindly taking index 0
    best = max(
        results,
        key=lambda p: similarity_score(
            normalize_text(p.get("display_name", "")),
            title
        )
    )

    related_ids = best.get("related_works", [])[:12]
    if not related_ids:
        return []

    # Step 2: batch-fetch those related works
    ids_filter = "|".join(related_ids)
    params2 = {
        "filter": f"openalex_id:{ids_filter}",
        "per-page": 12,
    }
    try:
        if client is None:
            with httpx.Client(timeout=20.0) as local_client:
                r2 = local_client.get(url, params=params2)
                r2.raise_for_status()
                d2 = r2.json()
        else:
            r2 = client.get(url, params=params2)
            r2.raise_for_status()
            d2 = r2.json()
        return d2.get("results", [])
    except Exception:
        return []


def normalize_candidate(paper):
    authorships = paper.get("authorships", [])
    authors = ", ".join(
        author.get("author", {}).get("display_name", "")
        for author in authorships
        if author.get("author", {}).get("display_name")
    )

    primary_location = paper.get("primary_location") or {}
    source = primary_location.get("source") or {}
    source_name = source.get("display_name") or "OpenAlex"

    landing_page_url = primary_location.get("landing_page_url")
    pdf_url = primary_location.get("pdf_url")
    doi = paper.get("doi")
    doi_url = doi if doi else None
    accessible_url = pdf_url or landing_page_url or doi_url

    # Reconstruct abstract from inverted index
    abstract = paper.get("abstract_inverted_index")
    snippet = ""
    has_abstract = False
    if abstract:
        word_positions = []
        for word, positions in abstract.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        snippet = " ".join(word for _, word in word_positions)
        has_abstract = True

    # Increase to 800 chars for better scoring signal downstream.
    # Fall back to title text if abstract is truly absent so the paper
    # isn't invisible to the relevance filter.
    snippet_clean = normalize_text(snippet)[:800]
    if not snippet_clean:
        snippet_clean = normalize_text(paper.get("display_name", ""))

    return {
    "title": normalize_text(paper.get("display_name")),
    "authors": authors or "Unknown authors",
    "year": paper.get("publication_year") or "Unknown",
    "snippet": snippet_clean,
    "has_abstract": has_abstract,
    "source": source_name,
    "retrieved_from": "OpenAlex",
    "paper_url": landing_page_url,
    "pdf_url": pdf_url,
    "doi_url": doi_url,
    "accessible_url": accessible_url,
    "priority_boost": 0,
}

    
    
def search_and_normalize_candidates(query, limit=15):
    semantic_candidates = []
    try:
        semantic_results = search_semantic_scholar(query, limit=limit)
        semantic_candidates = [
            normalize_semantic_scholar_candidate(paper)
            for paper in semantic_results
            if paper
        ]
    except Exception:
        semantic_candidates = []

    semantic_accessible = [
        paper for paper in semantic_candidates if paper.get("accessible_url")
    ]

    if len(semantic_accessible) >= 5:
        return semantic_accessible

    openalex_candidates = []
    try:
        openalex_results = search_openalex(query, limit=limit)
        openalex_candidates = [
            normalize_candidate(paper)
            for paper in openalex_results
            if paper
        ]
    except Exception:
        openalex_candidates = []

    combined = semantic_accessible + [
        paper for paper in openalex_candidates if paper.get("accessible_url")
    ]
    return combined
  


# ---------------------------------------------------------------------------
# CANDIDATE FILTERING & SELECTION
# ---------------------------------------------------------------------------

def filter_relevant_candidates(profile, candidates, min_overlap=2):
    """
    Keep candidates that share at least `min_overlap` topic terms with
    the source paper, OR match at least one key phrase.

    If the stricter threshold leaves fewer than 5 candidates (too aggressive),
    automatically relaxes to overlap >= 1 to preserve recall.
    """
    topic_terms = set(extract_topic_terms(
        profile.get("title", ""), profile.get("abstract", "")
    ))
    key_phrases = extract_key_phrases(
        profile.get("title", ""), profile.get("abstract", "")
    )

    if not topic_terms:
        return candidates

    def passes(item, threshold):
        candidate_text = (
            f"{item.get('title', '')} {item.get('snippet', '')}".lower()
        )
        candidate_words = _content_terms(candidate_text)
        overlap = topic_terms & candidate_words
        phrase_match = any(phrase in candidate_text for phrase in key_phrases[:5])
        return len(overlap) >= threshold or phrase_match

    filtered = [item for item in candidates if passes(item, min_overlap)]

    # Relax if too few survive
    if len(filtered) < 5:
        filtered = [item for item in candidates if passes(item, 1)]

    return filtered


def deduplicate_candidates(candidates):
    unique = []
    seen = set()
    seen_titles = []
    for item in candidates:
        title = normalize_text(item.get("title", "")).lower()
        accessible_url = normalize_text(item.get("accessible_url", "")).lower()
        doi_url = normalize_text(item.get("doi_url", "")).lower()
        key = (title, doi_url or accessible_url)

        if key in seen:
            continue
        is_title_dup = any(
            SequenceMatcher(None, title, existing).ratio() > 0.90
            for existing in seen_titles
        )
        if is_title_dup:
            continue
        seen.add(key)
        seen_titles.append(title)
        unique.append(item)
    return unique


def exclude_source_paper(source_title, candidates):
    source_norm = normalize_text(source_title).lower()
    if not source_norm:
        return candidates
    filtered = []
    for paper in candidates:
        cand_title = normalize_text(paper.get("title", "")).lower()
        if not cand_title:
            filtered.append(paper)
            continue
        # Drop exact or near-exact self match.
        if SequenceMatcher(None, source_norm, cand_title).ratio() >= 0.93:
            continue
        filtered.append(paper)
    return filtered


def filter_domain_candidates(profile, candidates):
    source_domain = normalize_text(profile.get("domain", "OTHER")).upper()
    if not source_domain or source_domain == "OTHER":
        return candidates

    avoid_domains = set(profile.get("avoid_domains", []))
    conflicts = set(_DOMAIN_CONFLICT_SIGNALS.get(source_domain, set()))
    not_about = set()
    for domain_text in avoid_domains:
        for token in re.findall(r"[a-zA-Z]{3,}", normalize_text(domain_text).lower()):
            not_about.add(token)
    all_conflicts = conflicts | not_about
    if not all_conflicts:
        return candidates

    filtered = []
    for item in candidates:
        candidate_text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
        hit_count = sum(1 for signal in all_conflicts if signal in candidate_text)
        if hit_count >= 2:
            continue
        filtered.append(item)
    if len(filtered) < 10:
        return candidates
    return filtered


def select_rerank_pool(profile, candidates, pool_size=15):
    """
    Lightweight word-overlap pre-selection for the LLM rerank pool.

    Intentionally avoids the full heuristic scorer so that papers scoring
    poorly on surface features (short abstract, unusual vocabulary) are not
    buried before the LLM ever evaluates them.
    """
    source_words = _content_terms(f"{profile['title']} {profile['abstract']}")
    task_terms = {
        normalize_text(t).lower() for t in profile.get("task_terms", [])
    }

    scored = []
    for paper in candidates:
        cand_words = _content_terms(f"{paper['title']} {paper['snippet']}")
        overlap = len(source_words & cand_words)
        task_overlap = len(task_terms & cand_words)
        citation_bonus = paper.get("priority_boost", 0)
        score = (2 * task_overlap) + overlap + citation_bonus
        scored.append((score, paper))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:pool_size]]


# ---------------------------------------------------------------------------
# HEURISTIC RANKING (fallback only)
# ---------------------------------------------------------------------------

def rank_candidates(profile, candidates):
    """
    Heuristic scorer used ONLY when LLM reranking fails entirely.
    Not used to pre-filter before the LLM reranker.
    """
    ranked = []

    source_title = profile.get("title", "")
    source_abstract = profile.get("abstract", "")
    source_topic_terms = extract_topic_terms(source_title, source_abstract)

    title_terms = _content_terms(source_title)
    abstract_terms = _content_terms(source_abstract)
    source_topic_set = set(source_topic_terms)
    task_terms = {
        normalize_text(t).lower() for t in profile.get("task_terms", [])
    }

    for item in candidates:
        candidate_text = f"{item['title']} {item['snippet']}"
        candidate_words = _content_terms(candidate_text)

        title_score = similarity_score(source_title, item["title"])
        abstract_score = similarity_score(
            source_abstract[:1500], item["snippet"][:1500]
        )

        topic_overlap_ratio = 0
        if source_topic_set:
            topic_overlap_ratio = (
                len(source_topic_set & candidate_words) / len(source_topic_set)
            )

        title_overlap_count = len(title_terms & candidate_words)
        abstract_overlap_count = len(abstract_terms & candidate_words)
        topic_overlap_count = len(source_topic_set & candidate_words)
        task_overlap_count = len(task_terms & candidate_words)

        diversity_bonus = 0
        if topic_overlap_count >= 3:
            diversity_bonus += 0.15
        elif topic_overlap_count == 2:
            diversity_bonus += 0.08
        if title_overlap_count >= 2:
            diversity_bonus += 0.08
        if abstract_overlap_count >= 3:
            diversity_bonus += 0.05
        if task_overlap_count >= 2:
            diversity_bonus += 0.10

        # Small penalty for papers with no real abstract
        abstract_penalty = 0 if item.get("has_abstract", True) else 0.05

        score = (
            (0.18 * title_score)
            + (0.27 * abstract_score)
            + (0.40 * topic_overlap_ratio)
            + diversity_bonus
            - abstract_penalty
        )

        item["score"] = score
        ranked.append(item)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# REASON BUILDER
# ---------------------------------------------------------------------------

def build_reason(profile, candidate):
    source_title = normalize_text(profile.get("title", "")).lower()
    source_abstract = normalize_text(profile.get("abstract", "")).lower()
    candidate_title = normalize_text(candidate.get("title", "")).lower()
    candidate_snippet = normalize_text(candidate.get("snippet", "")).lower()

    title_words = _content_terms(source_title)
    abstract_words = _content_terms(source_abstract)
    candidate_words = _content_terms(f"{candidate_title} {candidate_snippet}")

    title_overlap = title_words & candidate_words - _STOPWORDS
    abstract_overlap = abstract_words & candidate_words - _STOPWORDS

    if len(title_overlap) >= 2:
        matched = ", ".join(sorted(list(title_overlap))[:3])
        return (
            f"Recommended because it shares strong title-level topics "
            f"such as: {matched}."
        )
    if len(abstract_overlap) >= 2:
        matched = ", ".join(sorted(list(abstract_overlap))[:3])
        return (
            f"Recommended because its abstract overlaps with the uploaded "
            f"paper on: {matched}."
        )
    return "Recommended because its overall topic is closely related to the uploaded paper."


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def get_recommendations(title, abstract, top_k=5, fetch_limit=15):
    """
    Pipeline order:
      1. LLM query generation (Gemini -> Mistral fallback)
      2. Deduplicate queries
      3. Semantic Scholar search with OpenAlex fallback
      4. OpenAlex related_works enrichment
      5. Deduplicate candidates
      6. Relevance filter (threshold=2, auto-relaxes to 1)
      7. Lightweight overlap pre-selection for rerank pool (pool_size=15)
      8. LLM reranking (Gemini -> Mistral fallback)
      9. Heuristic ranking applied ONLY to candidates outside rerank pool,
         or as full fallback if LLM reranking fails entirely
     10. Build reason strings and return top_k
    """
    total_t0 = time.perf_counter()
    profile = {
        "title": normalize_text(title),
        "abstract": normalize_text(abstract),
    }
    phase_t0 = time.perf_counter()
    task_profile = extract_task_profile_with_llm(profile["title"], profile["abstract"])
    profile["domain"] = task_profile.get("domain", "OTHER")
    profile["task"] = task_profile.get("task", "")
    profile["task_terms"] = task_profile.get("task_terms", [])
    profile["avoid_domains"] = task_profile.get("avoid_domains", [])
    phase_t1 = time.perf_counter()
    print("DEBUG task_profile:", task_profile)
    print(f"TIMING rec_phase=task_profile ms={(phase_t1 - phase_t0) * 1000:.0f}")

    # Step 1: Generate search queries
    phase_t0 = time.perf_counter()
    topic_terms = extract_topic_terms(profile["title"], profile["abstract"])
    print("DEBUG topic_terms:", topic_terms)

    llm_queries = generate_queries_with_llm(profile["title"], profile["abstract"])
    phase_t1 = time.perf_counter()
    print("DEBUG llm_queries (raw):", llm_queries)
    print(f"TIMING rec_phase=query_gen ms={(phase_t1 - phase_t0) * 1000:.0f}")

    # Step 2: Retrieve candidates
    phase_t0 = time.perf_counter()
    s2_results_norm = []
    oa_keyword_results = []

    with httpx.Client(timeout=20.0) as shared_client:
        if llm_queries:
            llm_queries = deduplicate_queries(llm_queries)
            print("DEBUG llm_queries (deduped):", llm_queries)
            primary_query = llm_queries[0]
            try:
                s2_results = search_semantic_scholar(primary_query, limit=fetch_limit, client=shared_client)
                s2_results_norm.extend(
                    normalize_semantic_scholar_candidate(p)
                    for p in s2_results if p
                )
            except Exception:
                pass
            oa_results = search_openalex_batch(llm_queries, limit=fetch_limit, max_workers=min(4, max(1, len(llm_queries))))
            oa_keyword_results.extend(normalize_candidate(p) for p in oa_results if p)
        else:
            query = build_search_query(profile["title"], profile["abstract"])
            print("DEBUG heuristic_query:", query)
            try:
                s2_results = search_semantic_scholar(query, limit=fetch_limit, client=shared_client)
                s2_results_norm.extend(
                    normalize_semantic_scholar_candidate(p)
                    for p in s2_results if p
                )
            except Exception:
                pass
            oa_results = search_openalex(query, limit=fetch_limit, client=shared_client)
            oa_keyword_results.extend(normalize_candidate(p) for p in oa_results if p)

        if len(s2_results_norm) + len(oa_keyword_results) < 5:
            fallback_query = build_fallback_query(profile["title"], profile["abstract"])
            print("DEBUG fallback_query:", fallback_query)
            fallback_oa = search_openalex(fallback_query, limit=fetch_limit, client=shared_client)
            oa_keyword_results.extend(normalize_candidate(p) for p in fallback_oa if p)

        # Step 3: OpenAlex related works enrichment
        phase_t0 = time.perf_counter()
        related_works = fetch_openalex_related(profile["title"], client=shared_client)
        print("DEBUG related_works_count:", len(related_works))
        oa_related_results = [normalize_candidate(paper) for paper in related_works]
        phase_t1 = time.perf_counter()
        print(f"TIMING rec_phase=oa_related ms={(phase_t1 - phase_t0) * 1000:.0f}")

    phase_t1 = time.perf_counter()
    print("DEBUG raw_results_count:", len(s2_results_norm) + len(oa_keyword_results))
    print(
        "DEBUG source_counts_retrieval:",
        {
            "semantic_scholar": len(s2_results_norm),
            "openalex_keyword": len(oa_keyword_results),
        },
    )
    print(f"TIMING rec_phase=retrieval ms={(phase_t1 - phase_t0) * 1000:.0f}")

    # Source-priority merge for dedupe: S2 > OA related > OA keyword
    phase_t0 = time.perf_counter()
    raw_results = s2_results_norm + oa_related_results + oa_keyword_results
    print(
        "DEBUG source_counts_merged:",
        {
            "semantic_scholar": len(s2_results_norm),
            "openalex_related": len(oa_related_results),
            "openalex_keyword": len(oa_keyword_results),
            "total_merged": len(raw_results),
        },
    )

    # Step 4: Deduplicate
    accessible = [p for p in raw_results if p.get("accessible_url")]
    accessible = deduplicate_candidates(accessible)
    accessible = exclude_source_paper(profile["title"], accessible)
    print("DEBUG accessible_after_dedup:", len(accessible))

    # Step 4.5: Hard domain pre-filter
    domain_filtered = filter_domain_candidates(profile, accessible)
    if domain_filtered:
        accessible = domain_filtered
    print("DEBUG accessible_after_domain_filter:", len(accessible))

    # Step 5: Relevance filter
    filtered = filter_relevant_candidates(profile, accessible, min_overlap=2)
    if filtered:
        accessible = filtered
    print("DEBUG accessible_after_filter:", len(accessible))
    phase_t1 = time.perf_counter()
    print(f"TIMING rec_phase=filtering ms={(phase_t1 - phase_t0) * 1000:.0f}")

    # Step 6: Lightweight pre-selection for rerank pool
    phase_t0 = time.perf_counter()
    rerank_pool = select_rerank_pool(profile, accessible, pool_size=15)

    # Step 7: LLM reranking
    reranked_indices = rerank_candidates_with_llm(
        profile["title"],
        profile["abstract"],
        rerank_pool
    )
    print("DEBUG reranked_indices:", reranked_indices)
    phase_t1 = time.perf_counter()
    print(f"TIMING rec_phase=rerank ms={(phase_t1 - phase_t0) * 1000:.0f}")

    if reranked_indices:
        reranked = []
        seen = set()

        for idx in reranked_indices:
            if isinstance(idx, int) and 0 <= idx < len(rerank_pool) and idx not in seen:
                reranked.append(rerank_pool[idx])
                seen.add(idx)

        for idx, paper in enumerate(rerank_pool):
            if idx not in seen:
                reranked.append(paper)

        pool_titles = {normalize_text(p["title"]).lower() for p in rerank_pool}
        remainder = [
            p for p in accessible
            if normalize_text(p["title"]).lower() not in pool_titles
        ]
        remainder_ranked = rank_candidates(profile, remainder)
        final_ordered = reranked + remainder_ranked
    else:
        print("DEBUG LLM reranking failed - using heuristic ranking")
        final_ordered = rank_candidates(profile, accessible)

    # Step 8: Build reasons and return
    final_results = []
    for paper in final_ordered[:top_k]:
        paper["reason"] = build_reason(profile, paper)
        final_results.append(paper)

    total_t1 = time.perf_counter()
    print(f"TIMING rec_phase=total ms={(total_t1 - total_t0) * 1000:.0f}")
    return final_results
