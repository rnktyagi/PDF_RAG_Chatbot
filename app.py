import streamlit as st
import requests
import uuid

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🤖", layout="centered")
st.title("📚 Enterprise Document Assistant")
st.markdown("Upload a PDF to the knowledge base and ask questions about it.")

# 2. SESSION STATE INITIALIZATION (The Memory Bank)
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. SIDEBAR: PDF UPLOAD (The Handoff to FastAPI)
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if st.button("Process Document"):
        if uploaded_file:
            with st.spinner("Transmitting to backend..."):
                # Package the file as multipart/form-data
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}

                payload_data = {"user_id": st.session_state.user_id}
                
                # Send the POST request to your local FastAPI server
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000", 
                        files=files,
                        data=payload_data
                    )
                    if response.status_code == 200:
                        st.success("File uploaded! The AI is reading it in the background.")
                    else:
                        st.error(f"Upload failed: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is FastAPI running?")
        else:
            st.warning("Please select a file first.")

# 4. CHAT HISTORY RENDERING
for msg in st.session_state.messages:
    # This creates a chat bubble. role is either 'user' or 'assistant'
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. CHAT INPUT & BACKEND COMMUNICATION
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # Immediately render the user's new message to the UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show a loading indicator while we wait for FastAPI and LangGraph
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*(Thinking...)*")
        
        # Prepare the JSON payload for FastAPI
        payload = {
            "message": prompt,
            "user_id": st.session_state.user_id,
            "thread_id": st.session_state.thread_id
        }
        
        try:
            # Send the question to your LangGraph backend
            chat_response = requests.post("http://127.0.0.1:8000", json=payload)
            
            if chat_response.status_code == 200:
                # Extract the answer from the JSON response
                answer = chat_response.json().get("response", "No response found.")
                
                # Update the UI placeholder with the actual answer
                message_placeholder.markdown(answer)
                
                # Save the AI's answer to the session state memory
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                message_placeholder.error(f"Error: {chat_response.text}")
                
        except requests.exceptions.ConnectionError:
            message_placeholder.error("Cannot connect to backend. Is FastAPI running?")