import streamlit as st
import requests
import uuid

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("📚 Enterprise Document Assistant")
st.markdown(
    "Upload a PDF to the knowledge base and ask questions about it."
)

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Document Ingestion")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type="pdf"
    )

    if st.button("Process Document"):
        if uploaded_file:
            with st.spinner("Transmitting to backend..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                }

                payload_data = {
                    "user_id": st.session_state.user_id
                }

                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/upload",
                        files=files,
                        data=payload_data
                    )

                    if response.status_code == 200:
                        st.success(
                            "File uploaded! The AI is reading it in the background."
                        )
                    else:
                        st.error(
                            f"Upload failed: {response.text}"
                        )

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to backend. Is FastAPI running?"
                    )
        else:
            st.warning("Please select a file first.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(
    "Ask a question about your documents..."
):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        message_placeholder.markdown(
            "*(Thinking...)*"
        )

        payload = {
            "message": prompt,
            "user_id": st.session_state.user_id,
            "thread_id": st.session_state.thread_id
        }

        try:
            chat_response = requests.post(
                "http://127.0.0.1:8000/chat",
                json=payload
            )

            if chat_response.status_code == 200:
                answer = chat_response.json().get(
                    "response",
                    "No response found."
                )

                message_placeholder.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            else:
                message_placeholder.error(
                    f"Error: {chat_response.text}"
                )

        except requests.exceptions.ConnectionError:
            message_placeholder.error(
                "Cannot connect to backend. Is FastAPI running?"
            )