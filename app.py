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

if prompt := st.chat_input("Ask something about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "message": prompt,
        "user_id": st.session_state.user_id,
        "thread_id": st.session_state.thread_id,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(f"{BASE_URL}/chat", json=payload)
        if response.status_code == 200:
            answer = response.json().get("response", "")
        else:
            answer = f"Error: {response.text}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})