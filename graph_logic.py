"""
LangGraph pipeline that turns one topic into four pieces of content:
Facebook post, X/Twitter thread, LinkedIn post, SEO meta tags.

Resilience: tries a list of LLM providers in priority order. If the first
provider fails (rate limit, 503, outage, etc.), it automatically falls back
to the next one instead of failing the whole generation.

Efficiency: all four pieces of content are produced in a SINGLE model call
(the model returns structured JSON), not four separate calls. This matters
a lot on free-tier quotas — e.g. Gemini's free tier caps requests PER DAY,
not per minute, so four parallel calls burn four times the daily quota for
one generation. One call keeps the same output while using 1/4 of the quota.
"""

import os
import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ContentState(TypedDict, total=False):
    topic: str
    tone: str
    audience: str
    facebook_post: str
    twitter_thread: str
    linkedin_post: str
    seo_meta: str


# ---------------------------------------------------------------------------
# Provider chain — built once from env vars set by run_pipeline(), then
# reused by the node. Order = fallback priority.
# ---------------------------------------------------------------------------
def _build_provider_chain():
    """Returns a list of (label, invoke_fn) tuples, in priority order."""
    chain = []
    configs = json.loads(os.environ.get("PROVIDERS_JSON", "[]"))

    for cfg in configs:
        provider = cfg["provider"]
        key = cfg["api_key"]

        if provider == "google" and key:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=key, temperature=0.8)
            chain.append(("Google Gemini", llm))

        elif provider == "anthropic" and key:
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(model="claude-sonnet-5", anthropic_api_key=key, temperature=0.8)
            chain.append(("Anthropic Claude", llm))

        elif provider == "openai" and key:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o-mini", api_key=key, temperature=0.8)
            chain.append(("OpenAI ChatGPT", llm))

    if not chain:
        raise ValueError("No LLM provider configured. Provide at least one API key.")

    return chain


def invoke_with_fallback(prompt: str) -> str:
    """
    Tries each configured provider in order. Returns the first successful
    response. Raises the last error if every provider fails.
    """
    chain = _build_provider_chain()
    last_error = None

    for label, llm in chain:
        try:
            resp = llm.invoke(prompt)
            return resp.content
        except Exception as e:
            last_error = e
            continue  # try the next provider in the chain

    raise RuntimeError(f"All providers failed. Last error: {last_error}")


def _context(state: ContentState) -> str:
    tone = state.get("tone", "professional")
    audience = state.get("audience", "a general online audience")
    return f'Topic: "{state["topic"]}"\nTone: {tone}\nAudience: {audience}'


# ---------------------------------------------------------------------------
# Single combined node — one API call produces all four pieces of content.
# ---------------------------------------------------------------------------
def generate_all_content(state: ContentState) -> dict:
    prompt = (
        f"{_context(state)}\n\n"
        "Produce four pieces of content for this topic. Return ONLY a valid JSON "
        "object with exactly these four keys and nothing else (no markdown fences, "
        "no commentary):\n\n"
        '{\n'
        '  "facebook_post": "Facebook post, 100-200 words, conversational and warm, '
        'short paragraphs, 1-2 relevant emojis, ends with a question or call-to-action",\n'
        '  "twitter_thread": "A 5-tweet thread, numbered 1/5 to 5/5, each tweet under '
        '280 characters, hook hard on tweet 1",\n'
        '  "linkedin_post": "LinkedIn post, 150-250 words, strong first line, short '
        'paragraphs, ends with a question to invite comments",\n'
        '  "seo_meta": "Plain text with labeled lines: Title Tag (max 60 chars), '
        'Meta Description (max 155 chars), Slug (url-friendly), Keywords (8-10 comma-separated)"\n'
        '}'
    )
    raw = invoke_with_fallback(prompt)

    # Models sometimes wrap JSON in ```json fences despite instructions — strip those.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to showing the raw response somewhere visible rather than crashing.
        return {
            "facebook_post": raw,
            "twitter_thread": "(voir l'onglet Facebook — le modèle n'a pas renvoyé un JSON structuré)",
            "linkedin_post": "",
            "seo_meta": "",
        }

    return {
        "facebook_post": data.get("facebook_post", ""),
        "twitter_thread": data.get("twitter_thread", ""),
        "linkedin_post": data.get("linkedin_post", ""),
        "seo_meta": data.get("seo_meta", ""),
    }


# ---------------------------------------------------------------------------
# Build the graph: a single node, START -> generate_all -> END
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(ContentState)
    graph.add_node("generate_all", generate_all_content)
    graph.add_edge(START, "generate_all")
    graph.add_edge("generate_all", END)
    return graph.compile()


def run_pipeline(topic: str, tone: str, audience: str, providers: list) -> ContentState:
    """
    providers: list of dicts in fallback priority order, e.g.
        [{"provider": "google", "api_key": "..."},
         {"provider": "anthropic", "api_key": "..."}]
    Entries with an empty api_key are ignored.
    """
    active = [p for p in providers if p.get("api_key")]
    os.environ["PROVIDERS_JSON"] = json.dumps(active)

    app = build_graph()
    result = app.invoke({"topic": topic, "tone": tone, "audience": audience})
    return result
