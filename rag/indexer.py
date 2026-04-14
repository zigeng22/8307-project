"""
Build FAISS vector index from MedQuAD dataset.

Usage:
    python rag/indexer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAG_CONFIG, MEDQUAD_CSV


def build_index():
    """Load MedQuAD, chunk, embed, and save FAISS index."""
    import pandas as pd
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document

    print("Loading MedQuAD...")
    df = pd.read_csv(MEDQUAD_CSV)

    # combine question + answer as document content
    docs = []
    for _, row in df.iterrows():
        text = f"Question: {row['question']}\nAnswer: {row['answer']}"
        metadata = {
            "source": row.get("source", ""),
            "focus_area": row.get("focus_area", ""),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    print(f"Loaded {len(docs)} documents")

    # split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CONFIG["chunk_size"],
        chunk_overlap=RAG_CONFIG["chunk_overlap"],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # build embeddings and index
    print(f"Building embeddings with {RAG_CONFIG['embedding_model']}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=RAG_CONFIG["embedding_model"]
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # save to disk
    index_dir = RAG_CONFIG["index_dir"]
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"FAISS index saved to {index_dir}")


if __name__ == "__main__":
    build_index()
