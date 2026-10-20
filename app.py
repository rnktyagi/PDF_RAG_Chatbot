import streamlit as st
import requests
import uuid

BASE_URL = "https://lightrt-pdf-rag.hf.space"

defaults = {
    "page": "landing",
    "thread_id": str(uuid.uuid4()),
    "user_id": str(uuid.uuid4()),
    "messages": [],
    "suggestions": [],
    "pending_question": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

on_landing = st.session_state.page == "landing"

st.set_page_config(
    page_title="CRM Question Answering Chatbot",
    page_icon="💬",
    layout="wide" if on_landing else "centered",
    initial_sidebar_state="collapsed" if on_landing else "expanded",
)


def go_to(page: str):
    st.session_state.page = page


def ask_followup(question: str):
    st.session_state.pending_question = question


def render_landing():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
        .block-container { max-width: 1100px; padding-top: 2.5rem; }

        .hero { text-align: center; padding: 2.5rem 1rem 1rem; }
        .badge {
            display: inline-block; padding: 6px 16px; border-radius: 999px;
            background: rgba(99,102,241,0.15); color: #818cf8;
            font-size: 0.85rem; font-weight: 600; letter-spacing: 0.04em;
            border: 1px solid rgba(99,102,241,0.35);
        }
        .hero h1 {
            font-size: 3.2rem; line-height: 1.15; margin: 1.2rem 0 0.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .hero p {
            font-size: 1.15rem; max-width: 720px; margin: 0 auto;
            opacity: 0.8; line-height: 1.65;
        }

        .section-title { text-align: center; font-size: 1.9rem; font-weight: 700; margin: 3.5rem 0 0.4rem; }
        .section-sub { text-align: center; opacity: 0.7; margin-bottom: 1.8rem; }

        .card {
            border: 1px solid rgba(128,128,128,0.25); border-radius: 16px;
            padding: 1.4rem; height: 100%;
            background: rgba(128,128,128,0.06);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .card:hover { transform: translateY(-4px); border-color: #818cf8; }
        .card .icon { font-size: 2rem; }
        .card h4 { margin: 0.6rem 0 0.4rem; font-size: 1.1rem; }
        .card p { margin: 0; opacity: 0.75; font-size: 0.95rem; line-height: 1.55; }

        .step {
            text-align: center; padding: 1.2rem 0.8rem; border-radius: 16px;
            border: 1px dashed rgba(129,140,248,0.5); height: 100%;
        }
        .step .num {
            width: 38px; height: 38px; line-height: 38px; margin: 0 auto 0.6rem;
            border-radius: 50%; font-weight: 700; color: white;
            background: linear-gradient(135deg, #6366f1, #a855f7);
        }
        .step h4 { margin: 0.2rem 0; font-size: 1rem; }
        .step p { margin: 0; opacity: 0.75; font-size: 0.88rem; }

        .stat { text-align: center; padding: 1rem; }
        .stat .big { font-size: 2.2rem; font-weight: 800; color: #818cf8; }
        .stat .small { opacity: 0.7; font-size: 0.9rem; }

        .chips { text-align: center; }
        .chip {
            display: inline-block; margin: 5px; padding: 7px 16px; border-radius: 999px;
            border: 1px solid rgba(128,128,128,0.35); font-size: 0.9rem;
            background: rgba(128,128,128,0.08);
        }

        .cta {
            text-align: center; padding: 2.5rem 1rem; margin-top: 3rem; border-radius: 20px;
            background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(236,72,153,0.18));
            border: 1px solid rgba(129,140,248,0.35);
        }
        .cta h2 { margin: 0 0 0.4rem; }
        .cta p { opacity: 0.75; margin-bottom: 0; }

        .footer { text-align: center; opacity: 0.55; font-size: 0.85rem; margin: 2.5rem 0 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <span class="badge">AI-POWERED · RAG · CRM</span>
            <h1>Question Answering Chatbot<br>for Customer Relationship Management</h1>
            <p>
                Upload your CRM documents, such as customer policies, support guides, contracts, and
                reports, and get instant, accurate answers that are grounded in your own files
                and come with source citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1.2, 1, 1.2])
    with mid:
        st.write("")
        st.button(
            "🚀  Try the Chatbot",
            type="primary",
            use_container_width=True,
            on_click=go_to,
            args=("chat",),
            key="try_top",
        )

    st.write("")
    s1, s2, s3, s4 = st.columns(4)
    stats = [
        ("100%", "Answers grounded in your docs"),
        ("Page-level", "Source citations"),
        ("Multi-turn", "Conversation memory"),
        ("Secure", "Emails & card numbers redacted"),
    ]
    for col, (big, small) in zip((s1, s2, s3, s4), stats):
        col.markdown(
            f'<div class="stat"><div class="big">{big}</div><div class="small">{small}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">What is this project?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">A document-aware assistant that helps CRM and support teams find answers fast, '
        "without digging through long PDFs.</div>",
        unsafe_allow_html=True,
    )

    features = [
        ("📄", "Chat with your PDFs", "Upload a PDF and ask questions in plain language. Tables, headings and layout are parsed intelligently."),
        ("🎯", "Grounded answers", "Responses use only your uploaded documents. If the answer isn't there, the bot says so instead of guessing."),
        ("🔖", "Source citations", "Every answer points back to the file and page it came from, so you can verify it quickly."),
        ("🧠", "Context-aware memory", "Follow-up questions like “what about refunds?” are understood using the earlier conversation."),
        ("💡", "Suggested follow-ups", "After each answer you get 3 to 5 relevant questions to explore next, one click away."),
        ("🛡️", "Privacy first", "Sensitive data such as email addresses and credit card numbers is redacted before processing."),
    ]
    for row in (features[:3], features[3:]):
        cols = st.columns(3)
        for col, (icon, title, desc) in zip(cols, row):
            col.markdown(
                f'<div class="card"><div class="icon">{icon}</div><h4>{title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )
        st.write("")

    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">A smart pipeline decides when to search, when to reuse context, and when to retry.</div>',
        unsafe_allow_html=True,
    )
    steps = [
        ("Upload", "Your PDF is parsed, split into chunks and stored in a vector database."),
        ("Understand", "Your question is cleaned and rewritten into a standalone search query."),
        ("Retrieve & verify", "Relevant chunks are fetched and checked. If weak, the search is broadened and retried."),
        ("Answer", "A cited answer is generated, along with suggested next questions."),
    ]
    cols = st.columns(4)
    for i, (col, (title, desc)) in enumerate(zip(cols, steps), start=1):
        col.markdown(
            f'<div class="step"><div class="num">{i}</div><h4>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Where it helps in CRM</div>', unsafe_allow_html=True)
    u1, u2, u3 = st.columns(3)
    u1.markdown(
        '<div class="card"><div class="icon">🎧</div><h4>Customer support</h4>'
        "<p>Agents instantly find policy details, SLAs, and troubleshooting steps while serving a customer.</p></div>",
        unsafe_allow_html=True,
    )
    u2.markdown(
        '<div class="card"><div class="icon">🤝</div><h4>Sales &amp; onboarding</h4>'
        "<p>Answer product, pricing and contract questions from internal playbooks and proposals.</p></div>",
        unsafe_allow_html=True,
    )
    u3.markdown(
        '<div class="card"><div class="icon">📊</div><h4>Account management</h4>'
        "<p>Pull key facts out of reports and agreements without reading every page.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Built with</div>', unsafe_allow_html=True)
    stack = [
        "LangGraph", "LangChain", "FastAPI", "Streamlit", "Docling",
        "Qdrant", "PostgreSQL", "Groq LLM", "Pydantic", "Docker",
    ]
    chips = "".join(f'<span class="chip">{t}</span>' for t in stack)
    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="cta">
            <h2>Ready to see it in action?</h2>
            <p>Upload a PDF and start asking questions in under a minute.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    _, mid2, _ = st.columns([1.2, 1, 1.2])
    with mid2:
        st.button(
            "💬  Start Chatting",
            type="primary",
            use_container_width=True,
            on_click=go_to,
            args=("chat",),
            key="try_bottom",
        )

    st.markdown(
        '<div class="footer">Question Answering Chatbot for Customer Relationship Management</div>',
        unsafe_allow_html=True,
    )


def get_answer(prompt: str):
    payload = {
        "message": prompt,
        "user_id": st.session_state.user_id,
        "thread_id": st.session_state.thread_id,
    }
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload, timeout=120)
    except requests.RequestException as e:
        return f"Error: {e}", []

    if response.status_code == 200:
        data = response.json()
        suggestions = [] if data.get("is_fallback") else data.get("suggested_questions", [])
        return data.get("response", ""), suggestions
    return f"Error: {response.text}", []


def render_chat():
    st.title("💬 CRM Question Answering Chatbot")

    with st.sidebar:
        st.button("⬅️ Back to home", on_click=go_to, args=("landing",), use_container_width=True)
        st.divider()
        st.subheader("Upload a document")
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
        if uploaded_file and st.button("Upload"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {"user_id": st.session_state.user_id}
            with st.spinner("Uploading and processing..."):
                response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
            if response.status_code == 200:
                st.success(response.json().get("message"))
            else:
                st.error(f"Upload failed: {response.text}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.suggestions:
        with st.container(border=True):
            st.markdown("**💡 You might also ask**")
            for i, q in enumerate(st.session_state.suggestions):
                st.button(
                    q,
                    key=f"sugg_{len(st.session_state.messages)}_{i}",
                    on_click=ask_followup,
                    args=(q,),
                    use_container_width=True,
                )

    typed_prompt = st.chat_input("Ask something about your document...")
    prompt = st.session_state.pending_question or typed_prompt
    st.session_state.pending_question = None

    if prompt:
        st.session_state.suggestions = []
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, suggestions = get_answer(prompt)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.suggestions = suggestions
        st.rerun()


if on_landing:
    render_landing()
else:
    render_chat()
