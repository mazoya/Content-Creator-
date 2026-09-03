import os
import streamlit as st

from graph_logic import run_pipeline

st.set_page_config(page_title="Parallel Content Agent", page_icon="⚡", layout="wide")

st.title("⚡ Parallel AI Content Generator")
st.caption("One topic in → blog post, X thread, LinkedIn post, and SEO metadata out — generated in parallel by a LangGraph agent running on Gemini.")

with st.sidebar:
    st.header("Settings")
    default_key = os.environ.get("GOOGLE_API_KEY", "")
    api_key = st.text_input(
        "Google API key (Gemini)",
        value=default_key,
        type="password",
        help="Get one free at https://aistudio.google.com/apikey. "
             "Set GOOGLE_API_KEY as an env var to pre-fill this.",
    )
    tone = st.selectbox(
        "Tone",
        ["professional", "casual/friendly", "witty", "authoritative", "inspirational"],
    )
    audience = st.text_input("Target audience", value="small business owners")

topic = st.text_area("Topic / brief", placeholder="e.g. Why farmers should track weekly crop market prices")

generate = st.button("Generate content", type="primary", use_container_width=True)

if generate:
    if not api_key:
        st.error("Please provide a Google API key in the sidebar.")
    elif not topic.strip():
        st.error("Please enter a topic.")
    else:
        with st.spinner("Running parallel agents (blog, thread, LinkedIn, SEO)..."):
            try:
                result = run_pipeline(topic, tone, audience, api_key)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                result = None

        if result:
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📝 Blog Post", "🐦 X Thread", "💼 LinkedIn", "🔍 SEO"]
            )
            with tab1:
                st.markdown(result.get("blog_post", "_No content generated_"))
            with tab2:
                st.markdown(result.get("twitter_thread", "_No content generated_"))
            with tab3:
                st.markdown(result.get("linkedin_post", "_No content generated_"))
            with tab4:
                st.code(result.get("seo_meta", "No content generated"), language=None)

            combined = (
                f"# Blog Post\n\n{result.get('blog_post','')}\n\n"
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
