import os
import hashlib
from typing import List, Dict, Any
import faiss
import numpy as np
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    def __init__(self, embeddings_dir="embeddings", docs_dir="documents"):
        self.embeddings_dir = embeddings_dir
        self.docs_dir = docs_dir
        self.index = None
        self.document_chunks = []
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Create Directories if they don't exist
        os.makedirs(embeddings_dir, exist_ok=True)
        os.makedirs(docs_dir, exist_ok=True)

        # Initialize FAISS index
        self.initialize_index()

    def initialize_index(self):
        """Init or LOAD existing FAISS index"""
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dimension)

        # Load existing chunks and update index if needed
        if os.path.exists(os.path.join(self.embeddings_dir, "chunks.npy")):
            self.document_chunks = np.load(os.path.join(self.embeddings_dir), allow_pickle=True).tolist()
            embeddings = np.load(os.path.join(self.embeddings_dir, "embeddings.npy"))
            if len(embeddings) > 0:
                self.index.add(embeddings)


    def load_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Load the document and splitting into chunks of data"""
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        elif file_extension in [".txt", ".md"]:
            loader = TextLoader(file_path)
        elif file_extension in [".ppt", ".pptx"]:
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        documents = loader.load()

        # Splitting documents into chunks
        text_splitter = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents=documents)

        # Convert the chunks into a list of dictionaries

        result = []
        for i, chunk in enumerate(chunks):
            # Generate an ID for each chunk
            content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
            doc_id = f"{os.path.basename(file_path)}_{i}_{content_hash[:8]}"

            result.append({
                "id": doc_id,
                "content": chunk.page_content,
                "metadata": {
                    "source": file_path,
                    "page": chunk.metadata.get("page", i),
                    "document_name": os.path.basename(file_path)
                }
            })
        return result

    def process_document(self, file_path: str) -> None:
        pass