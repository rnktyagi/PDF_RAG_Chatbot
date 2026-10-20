import streamlit as st
import requests
import uuid

BASE_URL = "https://lightrt-pdf-rag.hf.space"

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def ask_followup(question: str):
    st.session_state.pending_question = question


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


st.title("📄 PDF RAG Chatbot")

with st.sidebar:
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

# Suggested follow-ups, shown only under the latest answer
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
