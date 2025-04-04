import os
import streamlit as st
from document_processor import DocumentProcessor
from llm_handler import LLMHandler
import tempfile


# Initialize the document processor and LLM handler
@st.cache_resource
def get_document_processor():
    return DocumentProcessor()


@st.cache_resource
def get_llm_handler():
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        try:
            return LLMHandler()
        except ValueError:
            return None
    try:
        return LLMHandler(api_key=api_key)
    except ValueError:
        return None


# App title and description
st.title("Sivolta Chat - Your Custom Documentation AI")
st.subheader("Upload documents, ask questions, and get context-aware answers")

# Sidebar for settings and document upload
with st.sidebar:
    st.header("Settings")

    # API Key input
    api_key = st.text_input("OpenAI API Key (Optional - uses configured key if blank)", type="password", key="api_key_input")
    if api_key:
        st.session_state["api_key"] = api_key

    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "md", "ppt", "pptx"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                processor = get_document_processor()
                processor.process_document(tmp_path)
                st.success(f"Document {uploaded_file.name} processed successfully!")

            # Clean up the temp file
            os.unlink(tmp_path)

    st.header("Saved Conversations")
    llm_handler = get_llm_handler()
    if llm_handler:
        conversations = llm_handler.list_conversations()
        for conv in conversations:
            if st.button(f"{conv['topic']} ({conv['timestamp']})"):
                st.session_state["current_conversation"] = llm_handler.load_conversation(conv['filename'])
                st.rerun()

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages from history or current conversation
if "current_conversation" in st.session_state:
    for message in st.session_state["current_conversation"]["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    # Clear the current conversation after displaying
    del st.session_state["current_conversation"]
else:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Input for new question
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get response from LLM
    llm_handler = get_llm_handler()
    processor = get_document_processor()

    if not llm_handler:
        with st.chat_message("assistant"):
            st.error("Please enter a valid OpenAI API key in the sidebar.")
    else:
        # Search for relevant context
        context_chunks = processor.search(prompt, top_k=5)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if not context_chunks:
                    response = "I don't have enough information from your documents to answer this question. Try uploading more relevant documents."
                else:
                    # Get sources for citation
                    sources = set([chunk["metadata"]["document_name"] for chunk in context_chunks])
                    source_str = ", ".join(sources)

                    # Generate response
                    response = llm_handler.generate_response(prompt, context_chunks)
                    response += f"\n\nSources: {source_str}"

                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Option to save the conversation
if st.session_state.messages and len(st.session_state.messages) > 1:
    topic = st.text_input("Conversation Topic (to save this conversation)")
    if st.button("Save Conversation") and topic:
        llm_handler = get_llm_handler()
        if llm_handler:
            filepath = llm_handler.save_conversation(topic, st.session_state.messages)
            st.success(f"Conversation saved as {os.path.basename(filepath)}")

