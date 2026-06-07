import os
import time
import logging
import pandas as pd
import streamlit as st
import requests
from vector_store import VectorStore
from updater import get_updater
import config

# Set page configuration with a modern, responsive layout
st.set_page_config(
    page_title="Upgraded Dynamic KB Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark Theme, Glassmorphism, Metrics)
st.markdown("""
<style>
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Elegant gradients and headers */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B 0%, #8A2387 50%, #E94057 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 1.5rem;
    }
    
    /* Source Citations Box */
    .source-box {
        background-color: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #8A2387;
        padding: 10px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    
    /* Metric Cards Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 75, 75, 0.4);
    }
    
    /* Scrollbar for logs */
    .log-container {
        max-height: 180px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.8rem;
        background: #111;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #333;
        color: #00ff00;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RESOURCE INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_resource
def initialize_system():
    """
    Initializes vector store and background updater.
    Uses st.cache_resource to ensure it only initializes once.
    """
    db = VectorStore()
    updater = get_updater(db)
    
    # Run an initial scan to populate any existing documents on launch
    updater.scan_and_update()
    
    # Start the background polling thread
    updater.start()
    
    return db, updater

# Load database engine and updater
db, updater = initialize_system()

# -----------------------------------------------------------------------------
# APP STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------------------------
# SIDEBAR CONTROL PANEL
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/bot.png", width=80)
    st.markdown("## Knowledge Base Control Center")
    st.markdown("---")
    
    # LLM settings including OpenAI GPT support
    st.markdown("### ⚙️ LLM Integration")
    model_mode = st.radio(
        "Response Model Mode:",
        ["Extractive Search (Local, Free)", "OpenAI GPT RAG (Generative)", "Gemini RAG (Generative)"],
        index=1,  # Set OpenAI GPT RAG as the default mode
        help="Select 'OpenAI GPT' or 'Gemini' for conversational responses, or 'Extractive' for local comparison."
    )
    
    # Initialize model lists in session state if not present
    if "openai_models" not in st.session_state:
        st.session_state.openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    if "gemini_models" not in st.session_state:
        st.session_state.gemini_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

    openai_key = ""
    gemini_key = ""
    openai_model = st.session_state.openai_models[0]
    gemini_model = st.session_state.gemini_models[0]
    
    if model_mode == "OpenAI GPT RAG (Generative)":
        openai_key = st.text_input(
            "Enter OpenAI API Key:",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            placeholder="sk-..."
        )
        openai_model = st.selectbox(
            "Select GPT Model:",
            st.session_state.openai_models,
            index=0,
            help="Choose the GPT model to call. This list updates dynamically when you test connection."
        )
        if not openai_key:
            st.warning("⚠️ Please provide an OpenAI API Key to use GPT Mode.")
        else:
            if st.button("🔌 Test OpenAI Connection", use_container_width=True):
                with st.spinner("Testing connection..."):
                    try:
                        headers = {
                            "Authorization": f"Bearer {openai_key.strip()}"
                        }
                        res = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
                        if res.status_code == 200:
                            models_data = res.json().get("data", [])
                            model_names = [m["id"] for m in models_data if "gpt" in m["id"] or "chatgpt" in m["id"]]
                            if model_names:
                                # Sort to prioritize models containing 'mini' or '4o'
                                model_names.sort(key=lambda x: ("4o" in x or "mini" in x, x), reverse=True)
                                st.session_state.openai_models = model_names
                            st.success(f"✅ Connected! Selected: {st.session_state.openai_models[0]}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Connection Failed (HTTP {res.status_code}): {res.text}")
                    except Exception as e:
                        st.error(f"❌ Connection Failed: {e}")
            
    elif model_mode == "Gemini RAG (Generative)":
        gemini_key = st.text_input(
            "Enter Gemini API Key:",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            placeholder="AIzaSy..."
        )
        gemini_model = st.selectbox(
            "Select Gemini Model:",
            st.session_state.gemini_models,
            index=0,
            help="Choose the Gemini model to call. This list updates dynamically when you test connection."
        )
        if not gemini_key:
            st.warning("⚠️ Please provide a Gemini API Key to use Gemini Mode.")
        else:
            if st.button("🔌 Test Gemini Connection", use_container_width=True):
                with st.spinner("Testing connection..."):
                    try:
                        # Try v1 first
                        url = f"https://generativelanguage.googleapis.com/v1/models?key={gemini_key.strip()}"
                        res = requests.get(url, timeout=10)
                        if res.status_code != 200:
                            # Fallback to v1beta
                            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key.strip()}"
                            res = requests.get(url, timeout=10)
                            
                        if res.status_code == 200:
                            models_data = res.json().get("models", [])
                            model_names = [m["name"].replace("models/", "") for m in models_data if "generateContent" in m.get("supportedGenerationMethods", [])]
                            if model_names:
                                # Sort to prioritize "flash" models (generous free tier), then newer versions
                                model_names.sort(key=lambda x: ("flash" in x.lower(), "1.5" in x or "2.0" in x or "2.5" in x, x), reverse=True)
                                st.session_state.gemini_models = model_names
                            st.success(f"✅ Connected! Selected: {st.session_state.gemini_models[0]}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Connection Failed (HTTP {res.status_code}): {res.text}")
                    except Exception as e:
                        st.error(f"❌ Connection Failed: {e}")
            
    # FEATURE 1: Two-Stage Re-ranking Control
    st.markdown("### 🔍 Search Pipeline Settings")
    use_rerank = st.toggle(
        "Two-Stage Re-ranking",
        value=False,
        help="Enable this to use a local Cross-Encoder model. It will retrieve the top 10 vector results and re-rank them, picking the top 3 most semantically aligned chunks. Improves RAG accuracy."
    )
    
    st.markdown("---")
    
    # File Ingestion Section
    st.markdown("### 📥 Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload TXT, MD, or PDF documents:",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        help="Upload new files here to add them dynamically to the docs folder."
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            target_path = os.path.join(config.DOCS_DIR, uploaded_file.name)
            
            # Check if file already exists with same size to avoid redundant writes
            write_file = True
            if os.path.exists(target_path):
                if os.path.getsize(target_path) == uploaded_file.size:
                    write_file = False
                    
            if write_file:
                with open(target_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.toast(f"📥 Saved '{uploaded_file.name}' to docs folder!")
                
        # Force a database update scan after upload
        with st.spinner("Indexing uploaded files..."):
            res = updater.scan_and_update()
            if res["added"] or res["updated"]:
                st.success(f"Indexed: {', '.join(res['added'] + res['updated'])}")
                st.rerun()

    st.markdown("---")
    
    # DB Stats & Operations
    st.markdown("### 📊 Database Health")
    stats = db.get_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <small>Total Files</small><br>
            <strong style="font-size:1.4rem; color:#FF4B4B;">{stats['total_files']}</strong>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <small>Total Chunks</small><br>
            <strong style="font-size:1.4rem; color:#8A2387;">{stats['total_chunks']}</strong>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Force Scan Trigger
    if st.button("🔄 Force Database Scan", use_container_width=True):
        with st.spinner("Scanning docs/ folder..."):
            res = updater.scan_and_update()
            st.toast("Scan complete!")
            st.rerun()

    # FEATURE 2: Full Database & Docs Reset Control
    if st.button("🚨 Wipe Database & Docs", use_container_width=True, type="secondary"):
        with st.spinner("Wiping index and documents..."):
            res = db.wipe_database()
            updater.scan_and_update()
            st.toast("Cleaned vector database and cleared docs folder!")
            time.sleep(1)
            st.rerun()
            
    # Indexed Documents List (with deletion buttons)
    st.markdown("### 📂 Indexed Documents")
    if stats["files"]:
        for file in stats["files"]:
            c_count = stats["file_chunk_counts"].get(file, 0)
            col_name, col_btn = st.columns([4, 1])
            with col_name:
                st.caption(f"📄 {file} ({c_count} chunks)")
            with col_btn:
                if st.button("🗑️", key=f"del_{file}", help=f"Remove {file} from knowledge base"):
                    file_path = os.path.join(config.DOCS_DIR, file)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    updater.scan_and_update()
                    st.toast(f"Removed '{file}' successfully!")
                    st.rerun()
    else:
        st.info("No documents indexed yet.")
        
    st.markdown("---")
    
    # System Log Window
    st.markdown("### 📜 System Activity Log")
    log_path = os.path.join(config.DB_DIR, "updater.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            last_lines = lines[-15:] if len(lines) > 15 else lines
            log_content = "".join(last_lines)
            
        st.markdown(f'<div class="log-container">{log_content.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN VIEW: TAB LAYOUT
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Upgraded Knowledge Base Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Equipped with Two-Stage Re-ranking, Analytics Visualizers, and Advanced Chunking.</div>', unsafe_allow_html=True)

tab_chat, tab_explorer = st.tabs(["💬 Conversational Chatbot", "🔍 Database Explorer & Search Tuner"])

# =============================================================================
# TAB 1: CONVERSATIONAL CHATBOT
# =============================================================================
with tab_chat:
    # If DB is empty, show welcoming tips
    if stats["total_chunks"] == 0:
        st.info("""
        👋 **Welcome! Your chatbot knowledge base is currently empty.**
        
        1. Drop some documents (`.txt`, `.md`, `.pdf`) inside the **`docs/`** directory in your project folder, or upload them using the **sidebar drag-and-drop**.
        2. Once uploaded, the background database monitor will automatically parse, chunk, embed, and index them.
        3. Type your question in the input box at the bottom of the screen!
        """)

    # Render Conversation bubbles
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sources" in message and message["sources"]:
                with st.expander("🔍 View Retrieved Source Snippets"):
                    for src in message["sources"]:
                        # Show rerank score if available
                        score_label = f"| **Rerank Score:** {src['rerank_score']:.4f}" if src.get('rerank_score') is not None else ""
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source:</strong> {src['source']} | <strong>Chunk:</strong> {src['chunk_index']} | <strong>Bi-Encoder Distance:</strong> {src['distance']:.4f} {score_label}
                            <br>
                            <em>"{src['text']}"</em>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat submission block
    if prompt := st.chat_input("Ask a question about your knowledge base..."):
        # User message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Assistant Response with spinner
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            with st.spinner("Retrieving facts & reasoning..."):
                # Query database (with optional Cross-Encoder Re-ranking!)
                hits = db.query(prompt, use_reranking=use_rerank)
                
                answer = ""
                sources = []
                api_error = None
                
                if not hits:
                    answer = "I couldn't find any relevant details in my database. Please make sure you have documents indexed."
                else:
                    # Parse retrieved chunks
                    for hit in hits:
                        sources.append({
                            "source": hit["metadata"]["source"],
                            "chunk_index": hit["metadata"]["chunk_index"],
                            "text": hit["text"],
                            "distance": hit["distance"],
                            "rerank_score": hit["rerank_score"]
                        })
                    
                    context_str = "\n\n".join([
                        f"--- Source: {hit['metadata']['source']} (Chunk {hit['metadata']['chunk_index']}) ---\n{hit['text']}" 
                        for hit in hits
                    ])
                    
                    # OpenAI Mode
                    if model_mode == "OpenAI GPT RAG (Generative)":
                        if openai_key:
                            try:
                                headers = {
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {openai_key.strip()}"
                                }
                                payload = {
                                    "model": openai_model,
                                    "messages": [
                                        {
                                            "role": "system",
                                            "content": (
                                                "You are a helpful knowledge base chatbot. Your job is to answer the user's question "
                                                "using ONLY the provided text snippets from the knowledge base. Be concise, precise, "
                                                "and cite the sources if relevant. If the snippets do not contain enough information to answer "
                                                "the question, clearly state that you couldn't find the answer in the current knowledge base."
                                            )
                                        },
                                        {
                                            "role": "user",
                                            "content": f"CONTEXT SNIPPETS:\n{context_str}\n\nUSER QUESTION: {prompt}"
                                        }
                                    ]
                                }
                                res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
                                if res.status_code == 200:
                                    answer = res.json()["choices"][0]["message"]["content"]
                                else:
                                    raise Exception(f"HTTP Status {res.status_code}: {res.text}")
                            except Exception as e:
                                api_error = f"OpenAI GPT API Error: {e}"
                                model_mode = "Extractive Search (Local, Free)"
                        else:
                            # User selected OpenAI but did not enter key
                            answer = "⚠️ **OpenAI API Key Missing**: You selected **OpenAI GPT RAG** mode but did not enter an API Key. Please input your OpenAI API Key in the sidebar control center to get conversational responses.\n\n"
                            model_mode = "Extractive Search (Local, Free)"
                    
                    # Gemini Mode
                    elif model_mode == "Gemini RAG (Generative)":
                        if gemini_key:
                            try:
                                model_name = gemini_model
                                if not model_name.startswith("models/"):
                                    model_name = f"models/{model_name}"
                                
                                headers = {"Content-Type": "application/json"}
                                system_instruction_text = (
                                    "You are a helpful knowledge base chatbot. Your job is to answer the user's question "
                                    "using ONLY the provided text snippets from the knowledge base. Be concise, precise, "
                                    "and cite the sources if relevant. If the snippets do not contain enough information to answer "
                                    "the question, clearly state that you couldn't find the answer in the current knowledge base."
                                )
                                payload = {
                                    "contents": [
                                        {
                                            "role": "user",
                                            "parts": [
                                                {
                                                    "text": f"CONTEXT SNIPPETS:\n{context_str}\n\nUSER QUESTION: {prompt}"
                                                }
                                            ]
                                        }
                                    ]
                                }
                                # gemini-pro (1.0) does not support systemInstruction, so we only add it for 1.5/2.0/2.5 models
                                if "1.5" in model_name or "2.0" in model_name or "2.5" in model_name:
                                    payload["systemInstruction"] = {
                                        "parts": [
                                            {
                                                "text": system_instruction_text
                                            }
                                        ]
                                    }
                                else:
                                    payload["contents"][0]["parts"][0]["text"] = f"INSTRUCTION:\n{system_instruction_text}\n\n{payload['contents'][0]['parts'][0]['text']}"

                                # Try v1 endpoint first (recommended for stable models like gemini-1.5-flash)
                                url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={gemini_key.strip()}"
                                res = requests.post(url, headers=headers, json=payload, timeout=30)
                                if res.status_code != 200:
                                    # Fall back to v1beta endpoint
                                    url_beta = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={gemini_key.strip()}"
                                    res = requests.post(url_beta, headers=headers, json=payload, timeout=30)

                                if res.status_code == 200:
                                    res_json = res.json()
                                    try:
                                        answer = res_json["candidates"][0]["content"]["parts"][0]["text"]
                                    except (KeyError, IndexError) as e:
                                        raise Exception(f"Failed to parse Gemini response: {res_json}. Error: {e}")
                                else:
                                    raise Exception(f"HTTP Status {res.status_code}: {res.text}")
                            except Exception as e:
                                api_error = f"Gemini API Error: {e}"
                                model_mode = "Extractive Search (Local, Free)"
                        else:
                            # User selected Gemini but did not enter key
                            answer = "⚠️ **Gemini API Key Missing**: You selected **Gemini RAG** mode but did not enter an API Key. Please input your Gemini API Key in the sidebar control center to get conversational responses.\n\n"
                            model_mode = "Extractive Search (Local, Free)"
                            
                    # Extractive Match Fallback
                    if model_mode == "Extractive Search (Local, Free)":
                        # Clean query words to extract key terms
                        import re
                        stop_words = {"what", "who", "is", "the", "of", "and", "a", "to", "in", "for", "on", "are", "with", "project", "question", "details", "information"}
                        query_words = set(re.findall(r'\b\w+\b', prompt.lower())) - stop_words
                        
                        sentences_extracted = []
                        for hit in hits:
                            text = hit["text"]
                            # Split chunk into sentences
                            sentences = re.split(r'(?<=[.!?])\s+', text)
                            for sentence in sentences:
                                sentence_clean = sentence.strip()
                                if not sentence_clean:
                                    continue
                                # Check matching keywords
                                sentence_words = set(re.findall(r'\b\w+\b', sentence_clean.lower()))
                                matches = query_words.intersection(sentence_words)
                                if len(matches) > 0:
                                    sentences_extracted.append((len(matches), sentence_clean, hit["metadata"]["source"]))
                                    
                        # Sort matching sentences by relevance score descending
                        sentences_extracted.sort(key=lambda x: x[0], reverse=True)
                        
                        if not sentences_extracted:
                            # Fallback if no matching sentences (just preview top chunk)
                            answer = f"🤖 **Local AI Assistant (Extractive Search):**\n\nI found matching records in **{hits[0]['metadata']['source']}**:\n\n> *\"{hits[0]['text'][:300]}...\"*"
                        else:
                            # Build a structured list of key facts
                            seen = set()
                            summary_bullets = []
                            for score, sentence, source in sentences_extracted:
                                if sentence.lower() not in seen:
                                    seen.add(sentence.lower())
                                    summary_bullets.append(f"- **{sentence}** *(Source: {source})*")
                                    if len(summary_bullets) >= 4:
                                        break
                                        
                            answer = (
                                "🤖 **Local AI Assistant (Extractive Summary):**\n\n"
                                "Here are the facts found in your database matching your question:\n\n"
                            )
                            answer += "\n".join(summary_bullets)
                            
                        # Append the billing reminder to help the user configure paid credentials
                        if not openai_key and not gemini_key:
                            answer += "\n\n*(To get rich conversational AI responses, configure your API credentials and switch modes in the sidebar.)*"
                        else:
                            answer += "\n\n*(Note: Your selected API provider returned billing/quota errors, so we generated this local summary fallback.)*"
                            if api_error:
                                answer += f"\n\n**Raw API Error Details**:\n```\n{api_error}\n```"
                
                response_placeholder.markdown(answer)
                
                if sources:
                    with st.expander("🔍 View Retrieved Source Snippets"):
                        for src in sources:
                            score_label = f"| **Rerank Score:** {src['rerank_score']:.4f}" if src.get('rerank_score') is not None else ""
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source:</strong> {src['source']} | <strong>Chunk:</strong> {src['chunk_index']} | <strong>Bi-Encoder Distance:</strong> {src['distance']:.4f} {score_label}
                                <br>
                                <em>"{src['text']}"</em>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                st.rerun()

    # FEATURE 3: Export Chat History (Markdown download)
    if st.session_state.messages:
        st.markdown("---")
        chat_md = "# Chat Transcript - Upgraded KB Chatbot\n\n"
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_md += f"### **{role}**:\n{msg['content']}\n\n"
            if "sources" in msg and msg["sources"]:
                chat_md += "*Sources cited:*\n"
                for src in msg["sources"]:
                    score_info = f" | Rerank: {src['rerank_score']:.4f}" if src.get('rerank_score') is not None else ""
                    chat_md += f"- `{src['source']}` (Chunk {src['chunk_index']}) | Distance: {src['distance']:.4f}{score_info}\n"
                chat_md += "\n"
                
        st.download_button(
            label="📥 Export Chat History (Markdown)",
            data=chat_md,
            file_name=f"chat_transcript_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True
        )

# =============================================================================
# TAB 2: DATABASE EXPLORER & SEARCH TUNER (Fulfilling RAG Analytics / "Train Dataset")
# =============================================================================
with tab_explorer:
    st.markdown("### 🛠️ Interactive Search Tester & Latency Visualizer")
    st.caption("Type a test query to measure exact search speeds, cosine vector distances, and Cross-Encoder score distributions.")
    
    test_col1, test_col2 = st.columns([3, 1])
    with test_col1:
        test_query = st.text_input("Enter a query to test retrieval:", key="test_q_input", placeholder="Type a concept (e.g. project alpha, leaves, python)...")
    with test_col2:
        test_rerank_toggle = st.checkbox("Apply Cross-Encoder Re-ranking", value=True, help="Enabling this shows both vector similarity and deep semantic scoring.")

    if test_query:
        # Measure search latency
        t_start = time.time()
        test_hits = db.query(test_query, use_reranking=test_rerank_toggle, n_results=5)
        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000
        
        # Display performance stats
        st.markdown(f"⏱️ **Search Latency**: `{latency_ms:.2f} ms` | **Candidates Evaluated**: `{len(test_hits)}`")
        
        # Render matching chunks inside cards with distance scores
        for idx, hit in enumerate(test_hits):
            score_col1, score_col2 = st.columns([1, 4])
            with score_col1:
                # Color code metrics based on distance/score
                st.markdown(f"**Match #{idx+1}**")
                st.metric(
                    label="Vector Distance", 
                    value=f"{hit['distance']:.4f}", 
                    help="Lower is closer. Distances below 1.0 indicate strong vector similarity."
                )
                if hit['rerank_score'] is not None:
                    st.metric(
                        label="Rerank Score", 
                        value=f"{hit['rerank_score']:.4f}", 
                        delta=f"{hit['rerank_score'] - hit['distance']:.2f}" if hit['rerank_score'] is not None else None,
                        help="Calculated by the Cross-Encoder. Higher values represent better semantic matches."
                    )
            with score_col2:
                st.markdown(f"📄 **Source File**: `{hit['metadata']['source']}` | **Chunk Index**: `{hit['metadata']['chunk_index']}`")
                st.info(hit["text"])
                
    st.markdown("---")
    st.markdown("### 📂 Entire Database Document Inventory")
    st.caption("Here is the spreadsheet representation of all segmented text chunks currently stored inside your vector database.")
    
    try:
        # Fetch inventory
        collection_data = db.collection.get(include=["documents", "metadatas"])
        
        if collection_data and collection_data["documents"]:
            inventory_rows = []
            for cid, doc, meta in zip(collection_data["ids"], collection_data["documents"], collection_data["metadatas"]):
                inventory_rows.append({
                    "Chunk ID": cid,
                    "Source File": meta.get("source", "unknown"),
                    "Chunk Index": meta.get("chunk_index", 0),
                    "Character Count": len(doc),
                    "Content Preview": doc[:120] + "..." if len(doc) > 120 else doc
                })
            
            # Convert to pandas DataFrame for premium table rendering
            df_inv = pd.DataFrame(inventory_rows)
            
            # Interactive search and filter table
            st.dataframe(
                df_inv, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Content Preview": st.column_config.TextColumn(width="large")
                }
            )
        else:
            st.info("No records inside ChromaDB yet. Index files using the sidebar to view them here.")
    except Exception as e:
        st.error(f"Failed to load database inventory: {e}")
