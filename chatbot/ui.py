# import streamlit as st
# import sys
# import os

# # ✅ MUST be the FIRST Streamlit command
# st.set_page_config(page_title="GDELT RAG Chatbot", layout="wide")

# # Fix path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from sentence_transformers import SentenceTransformer
# from pipeline.vector_store import load_index
# from pipeline.retrieval import retrieve
# from pipeline.rerank import load_reranker, rerank
# from pipeline.rag import generate_answer
# from pipeline.query_pipeline import QueryIntent
# from config import EMBED_MODEL

# # -----------------------
# # Load once (cached)
# # -----------------------
# @st.cache_resource
# def load_components():
#     embed_model = SentenceTransformer(EMBED_MODEL)
#     reranker = load_reranker()
#     faiss_index, chunk_id_map = load_index()
#     return embed_model, reranker, faiss_index, chunk_id_map

# embed_model, reranker, faiss_index, chunk_id_map = load_components()

# # -----------------------
# # UI
# # -----------------------
# st.title("🌍 GDELT RAG Chatbot")

# # Sidebar filters
# st.sidebar.header("🔍 Filters")

# year = st.sidebar.selectbox("Year", ["All", "2025"])
# country = st.sidebar.text_input("Country (e.g., Kenya, India)")
# event_type = st.sidebar.selectbox(
#     "Event Type",
#     ["All", "Conflict", "Cooperation"]
# )

# # User query
# query = st.text_input("Ask your question:")

# # -----------------------
# # Run pipeline
# # -----------------------
# if st.button("Search") and query:

#     # Build intent
#     intent = QueryIntent(
#         raw_query=query,
#         search_text=query,
#         date_from=None,
#         date_to=None,
#         location=None,
#         actor=None,
#         event_type=None
#     )

#     # Apply filters
#     if year != "All":
#         intent.date_from = f"{year}-01-01"
#         intent.date_to = f"{year}-12-31"

#     if country:
#         intent.location = country

#     if event_type != "All":
#         intent.event_type = event_type

#     # Retrieve
#     chunks = retrieve(intent, embed_model, faiss_index, chunk_id_map)

#     if not chunks:
#         st.warning("No results found.")
#     else:
#         # Rerank
#         top_chunks = rerank(query, chunks, reranker)

#         # Generate answer
#         result = generate_answer(query, top_chunks)

#         # Show answer
#         st.subheader("🤖 Answer")
#         st.write(result["answer"])

#         # Show sources
#         st.subheader("🔗 Sources")
#         for i, url in enumerate(result["sources"], 1):
#             st.write(f"{i}. {url}")
import streamlit as st
import sys
import os
import time

# ✅ FIRST
st.set_page_config(page_title="GDELT RAG Chatbot", layout="wide")

# -----------------------
# 🎨 LIGHT UI
# -----------------------
st.markdown("""
<style>
body, .main { background-color: #f8fafc; }

h1 { color: #1e293b; font-weight: 700; }

.card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    margin-bottom: 15px;
}

.answer-box {
    background: #ffffff;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #2563eb;
    font-size: 15px;
    line-height: 1.7;
    color: #1e293b;
}

.step {
    padding: 10px;
    border-radius: 8px;
    background: #eef2ff;
    margin-bottom: 6px;
    font-size: 14px;
    border-left: 4px solid #3b82f6;
}

.chunk-box {
    background: #ffffff;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    margin-bottom: 10px;
}

.chunk-title {
    font-weight: 600;
    color: #1e293b;
}

.chunk-score {
    font-size: 12px;
    color: #64748b;
}

.source-box {
    background: #f1f5f9;
    padding: 8px;
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 13px;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
from pipeline.vector_store import load_index
from pipeline.retrieval import retrieve
from pipeline.rerank import load_reranker, rerank
from pipeline.rag import generate_answer
from pipeline.query_pipeline import QueryIntent
from config import EMBED_MODEL

# -----------------------
# LOAD
# -----------------------
@st.cache_resource
def load_components():
    embed_model = SentenceTransformer(EMBED_MODEL)
    reranker = load_reranker()
    faiss_index, chunk_id_map = load_index()
    return embed_model, reranker, faiss_index, chunk_id_map

embed_model, reranker, faiss_index, chunk_id_map = load_components()

# -----------------------
# HEADER
# -----------------------
st.title("🌍 GDELT RAG Chatbot")


# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.header("🔍 Filters")

year = st.sidebar.selectbox("Year", ["All", "2025"])
country = st.sidebar.text_input("Country (e.g., India, Kenya)")
event_type = st.sidebar.selectbox("Event Type", ["All", "Conflict", "Cooperation"])

query = st.text_input("Ask your question:")

# -----------------------
# PIPELINE
# -----------------------
if st.button("Search") and query:

    start_time = time.time()

    intent = QueryIntent(
        raw_query=query,
        search_text=query,
        date_from=None,
        date_to=None,
        location=None,
        actor=None,
        event_type=None
    )

    if year != "All":
        intent.date_from = f"{year}-01-01"
        intent.date_to = f"{year}-12-31"

    if country:
        intent.location = country

    if event_type != "All":
        intent.event_type = event_type

    # STEP 1
    st.markdown('<div class="step">🧠 Retrieving relevant data...</div>', unsafe_allow_html=True)
    chunks = retrieve(intent, embed_model, faiss_index, chunk_id_map)

    if not chunks:
        st.warning("No results found.")
    else:
        st.success(f"Retrieved {len(chunks)} chunks")

        # STEP 2
        st.markdown('<div class="step">⚡ Reranking chunks...</div>', unsafe_allow_html=True)
        top_chunks = rerank(query, chunks, reranker)

        # STEP 3
        st.markdown('<div class="step">🤖 Generating answer...</div>', unsafe_allow_html=True)
        result = generate_answer(query, top_chunks)

        end_time = time.time()

        # -----------------------
        # TABS
        # -----------------------
        tab1, tab2, tab3 = st.tabs(["Answer", "Chunks", "Sources"])

        # -----------------------
        # ANSWER
        # -----------------------
        with tab1:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("Final Answer")

            st.markdown(
                f'<div class="answer-box">{result["answer"]}</div>',
                unsafe_allow_html=True
            )

            st.markdown(f"""
            
            ⏱ Time: {round(end_time - start_time, 2)} sec  
            📄 Chunks used: {len(top_chunks)}
            """)

            st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------
        # CHUNKS (FIXED HERE)
        # -----------------------
        with tab2:
            st.subheader("Retrieved Chunks")

            for i, chunk in enumerate(top_chunks, 1):

                # 🔥 Robust extraction (handles ANY format)
                if isinstance(chunk, dict):
                    text = (
                        chunk.get("text")
                        or chunk.get("content")
                        or chunk.get("chunk")
                        or chunk.get("document")
                        or chunk.get("page_content")
                        or str(chunk)
                    )
                    score = chunk.get("score", None)
                else:
                    text = str(chunk)
                    score = None

                st.markdown(f"""
                <div class="chunk-box">
                    <div class="chunk-title">Chunk {i}</div>
                    <div class="chunk-score">Score: {round(score,3) if score else "-"}</div>
                    <p>{text if text else "⚠️ No content found"}</p>
                </div>
                """, unsafe_allow_html=True)

        # -----------------------
        # SOURCES
        # -----------------------
        with tab3:
            st.subheader("Sources")

            if result.get("sources"):
                for src in result["sources"]:
                    st.markdown(
                        f'<div class="source-box">{src}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.write("No sources available")