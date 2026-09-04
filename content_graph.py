import os
import streamlit as st

from graph_logic import run_pipeline

st.set_page_config(page_title="Parallel Content Agent", page_icon="⚡", layout="wide")

st.title("⚡ Parallel AI Content Generator")
st.caption(
    "One topic in → Facebook post, X thread, LinkedIn post, and SEO metadata out — "
    "generated in parallel by a LangGraph agent. If your primary AI provider is "
    "overloaded or down, it automatically falls back to the next one."
)

with st.sidebar:
    st.header("Providers (fallback order)")
    st.caption("Fill in at least one. If the first one fails, the app tries the next automatically.")

    st.subheader("1️⃣ Google Gemini")
    google_key = st.text_input(
        "Google API key",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        type="password",
        help="Get one free at https://aistudio.google.com/apikey",
        key="google_key",
    )

    st.subheader("2️⃣ Anthropic Claude")
    anthropic_key = st.text_input(
        "Anthropic API key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Get one at https://console.anthropic.com/settings/keys",
        key="anthropic_key",
    )

    st.subheader("3️⃣ OpenAI ChatGPT")
    openai_key = st.text_input(
        "OpenAI API key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="Get one at https://platform.openai.com/api-keys",
        key="openai_key",
    )

    st.divider()
    tone = st.selectbox(
        "Tone",
        ["professional", "casual/friendly", "witty", "authoritative", "inspirational"],
    )
    audience = st.text_input("Target audience", value="small business owners")

topic = st.text_area("Topic / brief", placeholder="e.g. Why farmers should track weekly crop market prices")

generate = st.button("Generate content", type="primary", use_container_width=True)

if generate:
    providers = [
        {"provider": "google", "api_key": google_key},
        {"provider": "anthropic", "api_key": anthropic_key},
        {"provider": "openai", "api_key": openai_key},
    ]
    if not any(p["api_key"] for p in providers):
        st.error("Please provide at least one API key in the sidebar.")
    elif not topic.strip():
        st.error("Please enter a topic.")
    else:
        with st.spinner("Running parallel agents (Facebook, thread, LinkedIn, SEO)..."):
            try:
                result = run_pipeline(topic, tone, audience, providers)
            except Exception as e:
                st.error(f"Generation failed on every configured provider: {e}")
                result = None

        if result:
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📘 Facebook", "🐦 X Thread", "💼 LinkedIn", "🔍 SEO"]
            )
            with tab1:
                st.markdown(result.get("facebook_post", "_No content generated_"))
            with tab2:
                st.markdown(result.get("twitter_thread", "_No content generated_"))
            with tab3:
                st.markdown(result.get("linkedin_post", "_No content generated_"))
            with tab4:
                st.code(result.get("seo_meta", "No content generated"), language=None)

            combined = (
                f"# Facebook Post\n\n{result.get('facebook_post','')}\n\n"
                f"# X Thread\n\n{result.get('twitter_thread','')}\n\n"
                f"# LinkedIn Post\n\n{result.get('linkedin_post','')}\n\n"
                f"# SEO Metadata\n\n{result.get('seo_meta','')}\n"
            )
            st.download_button(
                "⬇️ Download all as Markdown",
                combined,
                file_name="generated_content.md",
                mime="text/markdown",
                use_container_width=True,
            )
