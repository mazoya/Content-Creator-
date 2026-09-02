"""
LangGraph pipeline that fans out one topic into several pieces of content
in parallel (blog post, X/Twitter thread, LinkedIn post, SEO meta tags),
then fans back in to a single result.

Model: Google Gemini via langchain-google-genai.
"""

import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ContentState(TypedDict, total=False):
    topic: str
    tone: str
    audience: str
    blog_post: str
    twitter_thread: str
    linkedin_post: str
    seo_meta: str


# ---------------------------------------------------------------------------
# LLM factory (created lazily so the API key can come from the UI at runtime)
# ---------------------------------------------------------------------------
def get_llm(api_key: str | None = None, model: str = "gemini-2.0-flash") -> ChatGoogleGenerativeAI:
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "No Google API key found. Set GOOGLE_API_KEY or pass one in from the UI."
        )
    return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0.8)


def _context(state: ContentState) -> str:
    tone = state.get("tone", "professional")
    audience = state.get("audience", "a general online audience")
    return f'Topic: "{state["topic"]}"\nTone: {tone}\nAudience: {audience}'


# ---------------------------------------------------------------------------
# Parallel nodes — each one is independent, so LangGraph runs them concurrently
# ---------------------------------------------------------------------------
def generate_blog(state: ContentState, config: dict) -> dict:
    llm = get_llm(config["configurable"].get("api_key"))
    prompt = (
        f"{_context(state)}\n\n"
        "Write a well-structured blog post (400-600 words) with a headline, "
        "a short intro, 2-3 subheadings, and a closing takeaway."
    )
    resp = llm.invoke(prompt)
    return {"blog_post": resp.content}


def generate_twitter(state: ContentState, config: dict) -> dict:
    llm = get_llm(config["configurable"].get("api_key"))
    prompt = (
        f"{_context(state)}\n\n"
        "Write a 5-tweet thread (X/Twitter). Number each tweet (1/5 ... 5/5), "
        "keep each under 280 characters, hook hard on tweet 1."
    )
    resp = llm.invoke(prompt)
    return {"twitter_thread": resp.content}


def generate_linkedin(state: ContentState, config: dict) -> dict:
    llm = get_llm(config["configurable"].get("api_key"))
    prompt = (
        f"{_context(state)}\n\n"
        "Write a LinkedIn post (150-250 words). Strong first line (it's what shows "
        "before 'see more'), short paragraphs, end with a question to invite comments."
    )
    resp = llm.invoke(prompt)
    return {"linkedin_post": resp.content}


def generate_seo(state: ContentState, config: dict) -> dict:
    llm = get_llm(config["configurable"].get("api_key"))
    prompt = (
        f"{_context(state)}\n\n"
        "Produce SEO metadata as plain text with these labeled lines:\n"
        "Title Tag: (max 60 chars)\n"
        "Meta Description: (max 155 chars)\n"
        "Slug: (url-friendly)\n"
        "Keywords: (comma-separated, 8-10 keywords)"
    )
    resp = llm.invoke(prompt)
    return {"seo_meta": resp.content}


# ---------------------------------------------------------------------------
# Build the graph: START fans out to all four nodes, all four fan back into END
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(ContentState)

    graph.add_node("blog", generate_blog)
    graph.add_node("twitter", generate_twitter)
    graph.add_node("linkedin", generate_linkedin)
    graph.add_node("seo", generate_seo)

    for node in ("blog", "twitter", "linkedin", "seo"):
        graph.add_edge(START, node)
        graph.add_edge(node, END)

    return graph.compile()


def run_pipeline(topic: str, tone: str, audience: str, api_key: str) -> ContentState:
    app = build_graph()
    result = app.invoke(
        {"topic": topic, "tone": tone, "audience": audience},
        config={"configurable": {"api_key": api_key}},
    )
    return result
