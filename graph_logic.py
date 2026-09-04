"""
LangGraph pipeline that fans out one topic into several pieces of content
in parallel (blog post, X/Twitter thread, LinkedIn post, SEO meta tags),
then fans back in to a single result.

Resilience: each node tries a list of LLM providers in priority order.
If the first provider fails (rate limit, 503, outage, etc.), it automatically
falls back to the next one instead of failing the whole generation.
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
# reused by every node. Order = fallback priority.
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
# Parallel nodes — each one is independent, so LangGraph runs them concurrently
# ---------------------------------------------------------------------------
def generate_facebook(state: ContentState) -> dict:
    prompt = (
        f"{_context(state)}\n\n"
        "Write a Facebook post (100-200 words). Conversational and warm tone, "
        "short paragraphs, use 1-2 relevant emojis naturally, end with a question "
        "or call-to-action that invites comments or shares."
    )
    return {"facebook_post": invoke_with_fallback(prompt)}


def generate_twitter(state: ContentState) -> dict:
    prompt = (
        f"{_context(state)}\n\n"
        "Write a 5-tweet thread (X/Twitter). Number each tweet (1/5 ... 5/5), "
        "keep each under 280 characters, hook hard on tweet 1."
    )
    return {"twitter_thread": invoke_with_fallback(prompt)}


def generate_linkedin(state: ContentState) -> dict:
    prompt = (
        f"{_context(state)}\n\n"
        "Write a LinkedIn post (150-250 words). Strong first line (it's what shows "
        "before 'see more'), short paragraphs, end with a question to invite comments."
    )
    return {"linkedin_post": invoke_with_fallback(prompt)}


def generate_seo(state: ContentState) -> dict:
    prompt = (
        f"{_context(state)}\n\n"
        "Produce SEO metadata as plain text with these labeled lines:\n"
        "Title Tag: (max 60 chars)\n"
        "Meta Description: (max 155 chars)\n"
        "Slug: (url-friendly)\n"
        "Keywords: (comma-separated, 8-10 keywords)"
    )
    return {"seo_meta": invoke_with_fallback(prompt)}


# ---------------------------------------------------------------------------
# Build the graph: START fans out to all four nodes, all four fan back into END
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(ContentState)

    graph.add_node("facebook", generate_facebook)
    graph.add_node("twitter", generate_twitter)
    graph.add_node("linkedin", generate_linkedin)
    graph.add_node("seo", generate_seo)

    for node in ("facebook", "twitter", "linkedin", "seo"):
        graph.add_edge(START, node)
        graph.add_edge(node, END)

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
