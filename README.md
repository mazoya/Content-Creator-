# Parallel AI Content Generator (LangGraph + Gemini + Streamlit)

Give it one topic and it fans out into four pieces of content **in parallel**
using LangGraph, then shows them all in a Streamlit UI:

- Blog post
- X/Twitter thread
- LinkedIn post
- SEO metadata (title tag, meta description, slug, keywords)

## How the parallelism works

`content_graph.py` builds a LangGraph `StateGraph` where `START` fans out to
four independent nodes (`blog`, `twitter`, `linkedin`, `seo`) that each write
to their own key in shared state, then all four converge to `END`. LangGraph
runs nodes with no dependency on each other concurrently, so all four API
calls to Gemini fire at the same time instead of one after another.

## 1. Get a free Gemini API key

https://aistudio.google.com/apikey

## 2. Run locally

```bash
cd parallel-content-agent
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key_here   # optional, can also paste it in the UI
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## 3. Deploy to Google Cloud Run

Requires the `gcloud` CLI, logged in, with a project selected.

```bash
cd parallel-content-agent

# Build and push the container, then deploy — one command:
gcloud run deploy parallel-content-agent \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY=your_key_here
```

`gcloud run deploy --source .` builds the `Dockerfile` in Cloud Build and
deploys it automatically — no manual `docker build`/`push` needed. When it
finishes it prints a public `*.run.app` URL.

**Note on the API key:** setting it with `--set-env-vars` is the quickest way
to get running, but it's visible in plain text in your Cloud Run service
config. For anything beyond a personal demo, store it in **Secret Manager**
instead and reference it with `--set-secrets GOOGLE_API_KEY=your-secret:latest`.

## Extending it

- Add more parallel nodes (e.g. Instagram caption, email newsletter) by
  copying the pattern of `generate_blog` in `content_graph.py` and wiring it
  into `build_graph()`.
- Swap `gemini-2.0-flash` for a stronger Gemini model in `get_llm()` if you
  want higher quality at the cost of latency/price.
- Add a review/critique node after the parallel ones (edges from each content
  node into a single "editor" node before `END`) if you want a pass that
  checks brand voice consistency across all four outputs.
