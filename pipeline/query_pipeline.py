# pipeline/query_pipeline.py

"""
Phase 7: Parse user query into structured filters + search query.
Supports:
- Natural language parsing (date, event)
- UI filters override (date, location, actor)
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta


# -------------------------------
# 📦 INTENT OBJECT
# -------------------------------
@dataclass
class QueryIntent:
    raw_query:     str
    search_text:   str
    date_from:     str | None = None
    date_to:       str | None = None
    location:      str | None = None
    actor:         str | None = None
    event_type:    str | None = None


# -------------------------------
# 📅 MONTH MAP
# -------------------------------
MONTH_MAP = {
    "january": "01", "february": "02", "march": "03",
    "april": "04",   "may": "05",      "june": "06",
    "july": "07",    "august": "08",   "september": "09",
    "october": "10", "november": "11", "december": "12",
    "jan": "01",     "feb": "02",      "mar": "03",
    "apr": "04",                        "jun": "06",
    "jul": "07",     "aug": "08",      "sep": "09",
    "oct": "10",     "nov": "11",      "dec": "12",
}


# -------------------------------
# ⚡ EVENT KEYWORDS
# -------------------------------
EVENT_KEYWORDS = {
    "conflict": ["conflict", "attack", "war", "strike", "bombing", "military", "operation"],
    "protest":  ["protest", "demonstration", "rally", "riot", "unrest"],
    "diplomacy":["diplomacy", "treaty", "negotiation", "agreement", "summit"],
    "election": ["election", "vote", "ballot", "polling"],
}


# -------------------------------
# 📅 DATE EXTRACTION
# -------------------------------
def extract_date_range(query: str):
    query_lower = query.lower()

    # "May 2025"
    for month_name, month_num in MONTH_MAP.items():
        pattern = rf"\b{month_name}\s+(\d{{4}})\b"
        match = re.search(pattern, query_lower)
        if match:
            year = match.group(1)

            # 🔥 better month end handling
            date_from = f"{year}-{month_num}-01"

            # simple last day logic
            last_day = "31"
            if month_num in ["04", "06", "09", "11"]:
                last_day = "30"
            elif month_num == "02":
                last_day = "28"

            date_to = f"{year}-{month_num}-{last_day}"
            return date_from, date_to

    # "in 2025"
    year_match = re.search(r"\b(\d{4})\b", query_lower)
    if year_match:
        year = year_match.group(1)
        return f"{year}-01-01", f"{year}-12-31"

    # "last 7 days"
    if "last" in query_lower and "days" in query_lower:
        days = int(re.search(r"(\d+)", query_lower).group(1))
        today = datetime.today()
        past = today - timedelta(days=days)
        return past.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    return None, None


# -------------------------------
# ⚡ EVENT TYPE EXTRACTION
# -------------------------------
def extract_event_type(query: str):
    query_lower = query.lower()
    for event_type, keywords in EVENT_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return event_type
    return None


# -------------------------------
# 🌍 LOCATION EXTRACTION (simple)
# -------------------------------
def extract_location(query: str):
    # simple heuristic (can improve later with NER)
    match = re.search(r"in ([A-Za-z\s]+)", query)
    if match:
        return match.group(1).strip()
    return None


# -------------------------------
# 👤 ACTOR EXTRACTION (basic)
# -------------------------------
def extract_actor(query: str):
    # simple: detect capitalized words (improvable)
    words = query.split()
    actors = [w for w in words if w.istitle()]
    return " ".join(actors) if actors else None


# -------------------------------
# 🧠 MAIN PARSER
# -------------------------------
def parse_query(raw_query: str, filters: dict | None = None) -> QueryIntent:
    """
    Combine:
    - NLP parsing
    - UI filters (override)
    """

    filters = filters or {}

    # 🔥 Extract from query
    date_from, date_to = extract_date_range(raw_query)
    event_type = extract_event_type(raw_query)
    location = extract_location(raw_query)
    actor = extract_actor(raw_query)

    # 🔥 Override with UI filters (IMPORTANT)
    date_from = filters.get("date_from") or date_from
    date_to   = filters.get("date_to")   or date_to
    location  = filters.get("location")  or location
    actor     = filters.get("actor")     or actor

    # 🔥 Clean query (important for embedding)
    search_text = raw_query

    for month_name in MONTH_MAP:
        search_text = re.sub(rf"\b{month_name}\s+\d{{4}}\b", "", search_text, flags=re.IGNORECASE)

    search_text = re.sub(r"\b\d{4}\b", "", search_text)
    search_text = re.sub(r"\s+", " ", search_text).strip()

    return QueryIntent(
        raw_query=raw_query,
        search_text=search_text or raw_query,
        date_from=date_from,
        date_to=date_to,
        location=location,
        actor=actor,
        event_type=event_type,
    )


# -------------------------------
# 🧪 TEST
# -------------------------------
if __name__ == "__main__":
    queries = [
        "conflicts in India in May 2025",
        "protests in USA last 7 days",
        "Russia Ukraine war 2024",
    ]

    for q in queries:
        intent = parse_query(q)
        print("\n---")
        print("Query:", intent.raw_query)
        print("Search:", intent.search_text)
        print("Date:", intent.date_from, "→", intent.date_to)
        print("Location:", intent.location)
        print("Actor:", intent.actor)
        print("Event:", intent.event_type)