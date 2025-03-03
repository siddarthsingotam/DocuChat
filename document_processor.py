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
        """Initialize or load existing FAISS index"""
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dimension)

        # Load existing chunks and update index
        chunks_path = os.path.join(self.embeddings_dir, "chunks.npy")
        embeddings_path = os.path.join(self.embeddings_dir, "embeddings.npy")

        if os.path.exists(chunks_path) and os.path.exists(embeddings_path):
            # Load document chunks
            self.document_chunks = np.load(chunks_path, allow_pickle=True).tolist()

            # Load embeddings
            embeddings = np.load(embeddings_path)

            # FAISS requires embeddings as float32 numpy array
            if len(embeddings) > 0:
                # Make sure embeddings are in the right format (float32)
                embeddings = embeddings.astype(np.float32)

                # FAISS expects the number of vectors and the vectors separately in some versions
                # But in most recent versions, just passing the array works
                try:
                    # Modern FAISS approach
                    self.index.add(embeddings)
                except TypeError:
                    # Older FAISS versions may require this syntax
                    n = embeddings.shape[0]  # Number of vectors
                    self.index.add(n, faiss.swig_ptr(embeddings))

                print(f"Loaded {len(embeddings)} embeddings into the index")


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
        """Process a document and add it to the index"""

        doc_name = os.path.basename(file_path)
        doc_path = os.path.join(self.docs_dir, doc_name)  # Copy doc and add it to documents directory

        # Skip processing if same file exists
        if os.path.exists(doc_path):
            print(f"Document {doc_name} exists already. Skipping...")
            return

        # Copy file
        import shutil
        shutil.copy2(file_path, doc_path)

        # Load and chunk document
        chunks = self.load_document(doc_path)

        # Create embeddings
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)

        # Add to index
        if len(chunks) > 0:
            self.index.add(embeddings)
            self.document_chunks.extend(chunks)

            # Save updated index and chunks
            self.save_index()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks"""
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            idx = int(idx)
            if idx < len(self.document_chunks) and idx >= 0:
                result = self.document_chunks[idx].copy()
                result["score"] = float(scores[0][i])
                results.append(result)

        return results

    def save_index(self) -> None:
        """Save the index and document chunks to disk"""
        if len(self.document_chunks) > 0:
            # Extract all embeddings
            texts = [chunk["content"] for chunk in self.document_chunks]
            embeddings = self.embedding_model.encode(texts)

            # Save embeddings and chunks
            np.save(os.path.join(self.embeddings_dir, "embeddings.npy"), embeddings)
            np.save(os.path.join(self.embeddings_dir, "chunks.npy"), self.document_chunks)
            print(f"Saved {len(self.document_chunks)} document chunks")


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    # Process a document
    processor.process_document("example_data/Siddarth_Singotam_CV-17-12-2024.pdf")
    # Search
    results = processor.search("What is the architecture?")
    for result in results:
        print(f"Score: {result['score']}, Source: {result['metadata']['document_name']}")
        print(result['content'][:200] + "...\n")