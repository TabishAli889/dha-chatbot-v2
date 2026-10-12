"""
DHA Lahore Property Chatbot — all 14 phases, live data (no RAG).
FastAPI + LLM tool-calling (Gemini by default, or Groq) over the public dhaconnects.com APIs.

Run locally:   uvicorn main:app --reload
Env vars:      LLM_PROVIDER = gemini (default) | groq
               GEMINI_API_KEY  or  GROQ_API_KEY
               LLM_MODEL (optional, overrides the default model)
"""

import json
import os
import re
import time
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

# Read keys from a local .env file (ignored on Render, which uses its own env vars).
load_dotenv()

BASE = "https://api.dhaconnects.com/api"
SITE = "https://dhaconnects.com"
IMG_BASE = "https://api.dhaconnects.com/"          # NOT /api/
CACHE_TTL = 600                                    # seconds
# Both providers speak the OpenAI chat-completions format, so one client handles either.
# Each provider lists models in order; if one hits its quota (429) the next is tried.
PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY",
               ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]),
    "groq":   ("https://api.groq.com/openai/v1", "GROQ_API_KEY",
               ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]),
}
PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
if PROVIDER not in PROVIDERS:
    PROVIDER = "gemini"
BASE_URL, KEY_ENV, DEFAULT_MODELS = PROVIDERS[PROVIDER]
# LLM_MODEL can be one model or a comma-separated fallback list.
MODELS = [m.strip() for m in os.getenv("LLM_MODEL", ",".join(DEFAULT_MODELS)).split(",") if m.strip()]
MODEL = MODELS[0]
_cooldown: dict[str, float] = {}  # model -> time it can be tried again
COOLDOWN_SECONDS = 60

# ---------------------------------------------------------------------------
# Phases — slugs copied exactly (capital L in phase 3, "brodway" misspelt)
# ---------------------------------------------------------------------------
PHASES: dict[str, dict[str, Any]] = {
    "dha-phase-1-lahore":             {"name": "DHA Phase 1",                     "aliases": ["phase 1", "p1"]},
    "dha-phase-2-lahore":             {"name": "DHA Phase 2",                     "aliases": ["phase 2", "p2"]},
    "dha-phase-3-Lahore":             {"name": "DHA Phase 3",                     "aliases": ["phase 3", "p3"]},
    "dha-lahore-phase-4":             {"name": "DHA Phase 4",                     "aliases": ["phase 4", "p4"]},
    "dha-phase-5-lahore":             {"name": "DHA Phase 5",                     "aliases": ["phase 5", "p5"]},
    "dha-lahore-phase-6":             {"name": "DHA Phase 6",                     "aliases": ["phase 6", "p6"]},
    "dha-lahore-phase-7":             {"name": "DHA Phase 7",                     "aliases": ["phase 7", "p7"]},
    "dha-lahore-phase-8":             {"name": "DHA Phase 8",                     "aliases": ["phase 8", "p8"]},
    "dha-lahore-phase-8-ivy-green":   {"name": "DHA Phase 8 Ivy Green",           "aliases": ["ivy green", "phase 8 ivy green"]},
    "dha-phase-8-brodway-commercial": {"name": "DHA Phase 8 Broadway Commercial", "aliases": ["broadway", "brodway", "phase 8 broadway"]},
    "dha-phase-9-town-lahore":        {"name": "DHA Phase 9 Town",                "aliases": ["phase 9 town", "9 town", "town"]},
    "dha-phase-9-prism-lahore":       {"name": "DHA Phase 9 Prism",               "aliases": ["phase 9 prism", "prism"]},
    "dha-phase-11-halloki-garden":    {"name": "DHA Phase 11 Halloki Garden",     "aliases": ["halloki", "halloki garden"]},
    "dha-phase-11-rehbar":            {"name": "DHA Phase 11 Rehbar",             "aliases": ["rehbar", "phase 11 rehbar"]},
}


def resolve_phase(text: str) -> Optional[str]:
    """Turn 'phase 9 prism', 'Prism', or an exact slug into the real slug."""
    if not text:
        return None
    if text in PHASES:
        return text
    t = text.lower().strip()
    for slug in PHASES:
        if slug.lower() == t:
            return slug
    # accept "8", "phase-8", "9-prism", "dha_phase_7"
    t = re.sub(r"[-_]+", " ", t)
    if re.match(r"^\d", t):
        t = "phase " + t
    # longest alias first so "phase 8 ivy green" beats "phase 8";
    # word boundaries so "phase 11" never matches "phase 1"
    best, best_len = None, 0
    for slug, info in PHASES.items():
        for a in info["aliases"] + [info["name"].lower()]:
            if re.search(rf"\b{re.escape(a)}\b", t) and len(a) > best_len:
                best, best_len = slug, len(a)
    return best


# "phase 9" / "phase 11" alone mean both sub-phases
GROUPS = {
    "phase 9": ["dha-phase-9-town-lahore", "dha-phase-9-prism-lahore"],
    "phase 11": ["dha-phase-11-halloki-garden", "dha-phase-11-rehbar"],
}


def resolve_phases(text: str) -> list[str]:
    slug = resolve_phase(text)
    if slug:
        return [slug]
    t = re.sub(r"[-_]+", " ", (text or "").lower().strip())
    if re.match(r"^\d", t):
        t = "phase " + t
    for g, slugs in GROUPS.items():
        if re.search(rf"\b{g}\b", t):
            return slugs
    return []


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[float, Any]] = {}
http = httpx.Client(timeout=20)


def _get(url: str, params: Optional[dict] = None) -> Any:
    key = url + json.dumps(params or {}, sort_keys=True)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    r = http.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    _cache[key] = (time.time(), data)
    return data


def to_num(v: Any) -> Optional[float]:
    """Prices/areas arrive as text like '56000000' or '5,000'."""
    if v is None:
        return None
    try:
        n = float(str(v).replace(",", "").strip())
        return n if n > 0 else None
    except ValueError:
        return None


def format_price(v: Any) -> str:
    n = to_num(v)
    if not n:
        return "Price on request"
    if n >= 10_000_000:
        return f"PKR {n / 10_000_000:.2f} Crore"
    if n >= 100_000:
        return f"PKR {n / 100_000:.2f} Lakh"
    return f"PKR {n / 1000:.2f} Thousand"


def area_in_marla(area: Any, unit: Any) -> Optional[float]:
    n = to_num(area)
    if n is None:
        return None
    u = str(unit or "").lower()
    if "kanal" in u:
        return n * 20
    if "sq" in u and ("ft" in u or "feet" in u):
        return n / 225
    if "yard" in u:
        return n / 25
    return n  # assume marla


def fix_image(path: Any) -> Optional[str]:
    if not path:
        return None
    p = str(path).replace("\\", "/").lstrip("/")
    return p if p.startswith("http") else IMG_BASE + p


def normalize(p: dict, phase_slug: str) -> dict:
    mapped = p.get("lat") is not None and p.get("lng") is not None
    return {
        "title": p.get("title"),
        "phase": PHASES[phase_slug]["name"],
        "category": p.get("category"),
        "type": p.get("type") or p.get("propertyType") or p.get("subCategory"),
        "price": format_price(p.get("price")),
        "price_pkr": to_num(p.get("price")),
        "area": f"{p.get('area')} {p.get('areaUnit') or ''}".strip(),
        "area_marla": area_in_marla(p.get("area"), p.get("areaUnit")),
        "block": p.get("block") or p.get("sector"),
        "plot": p.get("plot") or p.get("plotnum"),
        "on_map": mapped,
        "lat": p.get("lat"),
        "lng": p.get("lng"),
        "image": fix_image(p.get("thumbnailImage")),
        "slug": p.get("slug"),
        "url": f"{SITE}/properties/{p['slug']}" if p.get("slug") else None,
    }


def _looks_like_listing(x: Any) -> bool:
    return isinstance(x, dict) and any(k in x for k in ("title", "slug", "price"))


def collect_listings(raw: Any) -> list[dict]:
    """Gather ads from every list in the response, not just `data`.
    Ads whose plot didn't match DHA's survey (no lat/lng) may be returned
    under a separate key, e.g. Phase 2/3 have ads but 0 mappable."""
    if isinstance(raw, list):
        return [x for x in raw if _looks_like_listing(x)]
    if not isinstance(raw, dict):
        return []
    found: list[dict] = []
    stack = [raw]
    while stack:  # walk nested dicts like {"data": {"matched": [...], "unmatched": [...]}}
        node = stack.pop()
        for v in node.values():
            if isinstance(v, list) and v and _looks_like_listing(v[0]):
                found += [x for x in v if _looks_like_listing(x)]
            elif isinstance(v, dict) and not _looks_like_listing(v):
                stack.append(v)
    seen, unique = set(), []
    for x in found:  # de-duplicate if an ad appears in two lists
        key = x.get("slug") or x.get("id") or x.get("_id") or json.dumps(x, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(x)
    return unique


def fetch_listings(phase_slug: str, category: Optional[str] = None) -> dict:
    params = {"category": category} if category in ("Sell", "Rent") else None
    raw = _get(f"{BASE}/plots/public/{phase_slug}/listings", params)
    meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
    items = collect_listings(raw)
    return {"items": [normalize(p, phase_slug) for p in items], "meta": meta}


# ---------------------------------------------------------------------------
# Tools the LLM can call
# ---------------------------------------------------------------------------
def tool_list_phases() -> dict:
    return {"phases": [{"slug": s, "name": i["name"]} for s, i in PHASES.items()]}


def tool_search_listings(
    phase: Optional[str] = None,
    category: Optional[str] = None,
    keyword: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_marla: Optional[float] = None,
    max_marla: Optional[float] = None,
    sort: Optional[str] = None,
    limit: int = 10,
) -> dict:
    if phase:
        slugs = resolve_phases(phase)
        if not slugs:
            return {"error": f"Unknown phase '{phase}'.", "valid_phases": [i["name"] for i in PHASES.values()]}
    else:
        slugs = list(PHASES)

    results, failed, hidden = [], [], {}
    for s in slugs:
        try:
            d = fetch_listings(s, category)
            results += d["items"]
            n = hidden_count(d["meta"])
            if n:
                hidden[PHASES[s]["name"]] = n
        except Exception:
            failed.append(PHASES[s]["name"])

    kw = (keyword or "").lower()

    def ok(x: dict) -> bool:
        if kw and kw not in json.dumps(x).lower():
            return False
        p, a = x["price_pkr"], x["area_marla"]
        if min_price and (p is None or p < min_price): return False
        if max_price and (p is None or p > max_price): return False
        if min_marla and (a is None or a < min_marla): return False
        if max_marla and (a is None or a > max_marla): return False
        return True

    matches = [x for x in results if ok(x)]
    big = float("inf")
    if sort == "price_asc":
        matches.sort(key=lambda x: x["price_pkr"] or big)
    elif sort == "price_desc":
        matches.sort(key=lambda x: -(x["price_pkr"] or 0))
    elif sort == "area_desc":
        matches.sort(key=lambda x: -(x["area_marla"] or 0))

    limit = max(1, min(int(limit or 5), 8))
    # One compact line per listing — keeps token use (and free-tier limits) low.
    lines = [compact(m) for m in matches[:limit]]
    out = {"total": len(matches), "showing": len(lines), "listings": lines}
    if hidden:
        out["ads_without_details"] = hidden
        out["note"] = HIDDEN_NOTE
    if failed:
        out["failed"] = failed
    return out


HIDDEN_NOTE = ("These phases have more ads on dhaconnects.com whose details this assistant cannot see "
               "(their plot numbers didn't match DHA survey records). Their size, price and type are UNKNOWN, "
               "so never say they match the user's filters (e.g. don't call them '4 kanal plots'). Say the phase "
               "has N other ads of unknown size/price and suggest https://dhaconnects.com. "
               "Never say a phase has no listings if it has these.")


def hidden_count(meta: dict) -> int:
    """Ads the API counts but doesn't return (unmatched to DHA survey)."""
    try:
        return int(meta.get("unmatched") or 0)
    except (TypeError, ValueError):
        return 0


def compact(m: dict) -> str:
    """'10 Marla Plot | DHA Phase 8 | Sell | PKR 5.60 Crore | 10 Marla | Blk T Plot 12 | slug=abc'"""
    loc = " ".join(x for x in [f"Blk {m['block']}" if m.get("block") else "", f"Plot {m['plot']}" if m.get("plot") else ""] if x)
    parts = [m.get("title") or "Property", m["phase"], m.get("category") or "", m["price"], m["area"], loc,
             f"slug={m['slug']}" if m.get("slug") else ""]
    return " | ".join(p for p in parts if p)


def tool_phase_stats(phase: str) -> dict:
    slug = resolve_phase(phase)
    if not slug:
        multi = resolve_phases(phase)
        if multi:
            return {"phases": [tool_phase_stats(s) for s in multi]}
        return {"error": f"Unknown phase '{phase}'."}
    d = fetch_listings(slug)
    items = d["items"]
    prices = [x["price_pkr"] for x in items if x["price_pkr"]]
    return {
        "phase": PHASES[slug]["name"],
        "total_ads": int(d["meta"].get("total") or len(items)) if isinstance(d["meta"], dict) else len(items),
        "ads_with_details": len(items),
        "ads_without_details": hidden_count(d["meta"]),
        "for_sale": sum(1 for x in items if x["category"] == "Sell"),
        "for_rent": sum(1 for x in items if x["category"] == "Rent"),
        "cheapest": format_price(min(prices)) if prices else None,
        "most_expensive": format_price(max(prices)) if prices else None,
        "note": (HIDDEN_NOTE if hidden_count(d["meta"]) else None) if items or hidden_count(d["meta"])
                else "No properties listed in this phase yet.",
    }


def tool_get_property_details(slug: str) -> dict:
    try:
        raw = _get(f"{BASE}/property/public/{slug}")
    except httpx.HTTPStatusError:
        return {"error": f"No property found with slug '{slug}'."}
    p = raw.get("data", raw) if isinstance(raw, dict) else raw
    if not isinstance(p, dict):
        return {"error": "Unexpected response"}

    def pick(*keys):
        for k in keys:
            v = p.get(k)
            if v not in (None, "", []):
                return v
        return None

    agent = p.get("agent") or p.get("user") or p.get("owner") or {}
    if not isinstance(agent, dict):
        agent = {"name": agent}
    desc = str(pick("description", "details") or "")
    # Only the fields the bot needs — the raw record is too big for the token budget.
    out = {
        "title": pick("title"),
        "price": format_price(pick("price")),
        "area": f"{pick('area') or ''} {pick('areaUnit') or ''}".strip(),
        "category": pick("category"),
        "type": pick("type", "propertyType", "subCategory"),
        "block": pick("block", "sector"),
        "plot": pick("plot", "plotnum"),
        "bedrooms": pick("bedrooms", "beds"),
        "bathrooms": pick("bathrooms", "baths"),
        "description": desc[:500] + ("…" if len(desc) > 500 else ""),
        "agent_name": agent.get("name") or pick("agentName", "contactName"),
        "agent_phone": agent.get("phone") or agent.get("mobile") or pick("agentPhone", "phone", "contactNumber", "mobile"),
        "url": f"{SITE}/properties/{slug}",
    }
    return {k: v for k, v in out.items() if v not in (None, "", [])}


TOOL_FUNCS = {
    "search_listings": tool_search_listings,
    "phase_stats": tool_phase_stats,
    "get_property_details": tool_get_property_details,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "search_listings",
        "description": "Search live DHA Lahore property ads. Omit phase to search all phases. Prices in PKR, areas in marla.",
        "parameters": {"type": "object", "properties": {
            "phase": {"type": "string", "description": "e.g. 'phase 8', 'prism', 'ivy green', 'broadway'"},
            "category": {"type": "string", "enum": ["Sell", "Rent"]},
            "keyword": {"type": "string", "description": "e.g. 'house', 'commercial', 'corner'"},
            "min_price": {"type": "number"}, "max_price": {"type": "number"},
            "min_marla": {"type": "number"}, "max_marla": {"type": "number"},
            "sort": {"type": "string", "enum": ["price_asc", "price_desc", "area_desc"]},
            "limit": {"type": "integer", "description": "default 5, max 8"},
        }}}},
    {"type": "function", "function": {
        "name": "phase_stats", "description": "Ad counts and price range for a phase.",
        "parameters": {"type": "object", "properties": {"phase": {"type": "string"}}, "required": ["phase"]}}},
    {"type": "function", "function": {
        "name": "get_property_details",
        "description": "Description, agent name and phone for one property, by slug from search results.",
        "parameters": {"type": "object", "properties": {"slug": {"type": "string"}}, "required": ["slug"]}}},
]

PHASE_NAMES = ", ".join(i["name"].replace("DHA ", "") for i in PHASES.values())

SYSTEM_PROMPT = f"""You are the DHA Lahore Property Assistant. Phases covered: {PHASE_NAMES}.
Use ONLY tool data; never invent listings, prices or phone numbers. Call a tool before stating any of them.
Money: 1 crore = 10000000, 1 lakh = 100000. Area: 1 kanal = 20 marla (pass marla to tools).
Each search result is one line ending in slug=...; link it as https://dhaconnects.com/properties/<slug>.
Show title, phase, price, area, block/plot, link. Only say a phase has no ads if the tool reports none at all
(e.g. Phase 11). If a tool reports ads_without_details, say those ads exist but you can't show their details, and
point to https://dhaconnects.com.
Be short and use bullet lists. Reply in the user's language (English, Urdu or Roman Urdu)."""

# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------
_client: Optional[OpenAI] = None


def llm() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv(KEY_ENV)
        if not key:
            raise HTTPException(500, f"{KEY_ENV} is not set")
        _client = OpenAI(api_key=key, base_url=BASE_URL)
    return _client


def is_quota_error(e: Exception) -> bool:
    t = str(e)
    return getattr(e, "status_code", None) in (413, 429) or "rate_limit" in t or "RESOURCE_EXHAUSTED" in t


def call_llm(msgs: list[dict]):
    """Try each model in order, skipping ones that recently hit their quota."""
    now = time.time()
    ready = [m for m in MODELS if _cooldown.get(m, 0) <= now] or MODELS
    last: Optional[Exception] = None
    for model in ready:
        try:
            return llm().chat.completions.create(model=model, messages=msgs, tools=TOOLS, tool_choice="auto")
        except Exception as e:
            print(f"[llm] {model} failed: {str(e)[:400]}", flush=True)
            last = e
            if is_quota_error(e) or getattr(e, "status_code", None) == 404:
                _cooldown[model] = now + COOLDOWN_SECONDS
                continue  # try the next model
            raise
    raise last  # every model is out of quota


def run_chat(message: str, history: list[dict]) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Keep history small: last 6 turns, each trimmed — keeps requests small.
    for h in history[-6:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            msgs.append({"role": h["role"], "content": str(h["content"])[:700]})
    msgs.append({"role": "user", "content": message})

    for _ in range(5):  # max tool rounds
        resp = call_llm(msgs)
        m = resp.choices[0].message
        if not m.tool_calls:
            return m.content or ""
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
        for tc in m.tool_calls:
            fn = TOOL_FUNCS.get(tc.function.name)
            try:
                args = json.loads(tc.function.arguments or "{}")
                out = fn(**args) if fn else {"error": "unknown tool"}
            except Exception as e:
                out = {"error": str(e)}
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": json.dumps(out, default=str, ensure_ascii=False)[:3500]})
    return "Sorry, I couldn't finish that request. Please try rephrasing it."


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="DHA Lahore Property Chatbot", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatIn(BaseModel):
    message: str
    history: list[dict] = []


@app.get("/")
def root():
    return {"status": "ok", "phases": len(PHASES), "llm": PROVIDER, "models": MODELS, "endpoints": ["/chat", "/phases", "/listings/{phase}", "/property/{slug}"]}


@app.post("/chat")
def chat(body: ChatIn):
    if not body.message.strip():
        raise HTTPException(400, "message is empty")
    try:
        return {"reply": run_chat(body.message, body.history)}
    except HTTPException:
        raise
    except Exception as e:  # LLM errors: bad key, retired model, rate limit
        if is_quota_error(e):
            daily = "per day" in str(e).lower() or "PerDay" in str(e)
            raise HTTPException(429, "Daily free limit reached for the AI model — please try again later."
                                if daily else "Too many requests right now — please wait a minute and ask again.")
        raise HTTPException(502, f"LLM error: {e}")


@app.get("/phases")
def phases():
    return tool_list_phases()


@app.get("/listings/{phase}")
def listings(phase: str, category: Optional[str] = None):
    slugs = resolve_phases(phase)
    if not slugs:
        raise HTTPException(404, f"Unknown phase '{phase}'")
    out = []
    for s in slugs:
        d = fetch_listings(s, category)
        out.append({"phase": PHASES[s]["name"], "meta": d["meta"], "count": len(d["items"]), "data": d["items"]})
    return out[0] if len(out) == 1 else {"phases": out}


@app.get("/property/{slug}")
def property_details(slug: str):
    return tool_get_property_details(slug)