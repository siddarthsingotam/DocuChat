from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import os
import json
from typing import List, Optional
import shutil

from document_processor import DocumentProcessor
from llm_handler import LLMHandler

app = FastAPI(title="DocuChat API", description="API for document-based contextual AI chat")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
document_processor = DocumentProcessor()


def get_llm_handler(api_key: str = Form(...)):
    try:
        return LLMHandler(api_key=api_key)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid API key")


# Document management endpoints
@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    # Validate file
    allowed_extensions = [".pdf", ".docx", ".txt", ".md", ".ppt", ".pptx"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}")

    # Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        # Process the document
        document_processor.process_document(tmp_path)

        # Clean up the temp file
        os.unlink(tmp_path)

        return {"status": "success", "message": f"Document {file.filename} processed successfully"}
    except Exception as e:
        # Clean up on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.get("/documents/list")
async def list_documents():
    """List all processed documents"""
    try:
        docs_dir = document_processor.docs_dir
        files = os.listdir(docs_dir)
        documents = []

        for file in files:
            file_path = os.path.join(docs_dir, file)
            if os.path.isfile(file_path):
                stats = os.stat(file_path)
                documents.append({
                    "name": file,
                    "size": stats.st_size,
                    "created": stats.st_ctime
                })

        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


# Chat endpoints
@app.post("/chat/query")
async def query(
        query: str = Form(...),
        api_key: str = Form(...),
        top_k: Optional[int] = Form(5)
):
    """Query the system with context from documents"""
    try:
        # Create LLM handler
        llm_handler = LLMHandler(api_key=api_key)

        # Search for relevant context
        context_chunks = document_processor.search(query, top_k=top_k)

        if not context_chunks:
            return {
                "response": "I don't have enough information from your documents to answer this question.",
                "sources": []
            }

        # Generate response
        response = llm_handler.generate_response(query, context_chunks)

        # Extract sources for citation
        sources = [
            {
                "document": chunk["metadata"]["document_name"],
                "page": chunk["metadata"]["page"],
                "score": chunk.get("score", 0)
            }
            for chunk in context_chunks
        ]

        return {
            "response": response,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


# Conversation management
@app.post("/conversations/save")
async def save_conversation(
        topic: str = Form(...),
        messages: str = Form(...),  # JSON string of messages
        api_key: str = Form(...)
):
    """Save a conversation"""
    try:
        llm_handler = LLMHandler(api_key=api_key)

        # Parse messages
        messages_data = json.loads(messages)

        # Save conversation
        filepath = llm_handler.save_conversation(topic, messages_data)

        return {
            "status": "success",
            "message": f"Conversation saved",
            "filepath": os.path.basename(filepath)
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid message format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving conversation: {str(e)}")


@app.get("/conversations/list")
async def list_conversations(
        llm_handler: LLMHandler = Depends(get_llm_handler)
):
    """List all saved conversations"""
    try:
        conversations = llm_handler.list_conversations()
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")


@app.get("/conversations/{filename}")
async def get_conversation(
        filename: str,
        llm_handler: LLMHandler = Depends(get_llm_handler)
):
    """Get a specific conversation"""
    try:
        conversation = llm_handler.load_conversation(filename)
        if "error" in conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except Exception as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail="Conversation not found")
        raise HTTPException(status_code=500, detail=f"Error loading conversation: {str(e)}")


# Mount static files for the frontend
app.mount("/", StaticFiles(directory=".", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)