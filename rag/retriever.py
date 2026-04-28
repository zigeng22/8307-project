"""
RAG retriever: load FAISS index and retrieve relevant context.
"""
import sys
import math
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAG_CONFIG


_vectorstore = None


def _safe_text(value) -> str:
    """Convert arbitrary input to safe string for embedding query."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def get_vectorstore():
    """Lazy-load the FAISS vectorstore (singleton)."""
    global _vectorstore
    if _vectorstore is None:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=RAG_CONFIG["embedding_model"]
        )
        _vectorstore = FAISS.load_local(
            RAG_CONFIG["index_dir"], embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def retrieve(query: str, top_k: int = None) -> str:
    """Retrieve top-k relevant passages and return as a single string."""
    k = top_k or RAG_CONFIG["top_k"]
    query = _safe_text(query)
    if not query:
        return ""
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    passages = [doc.page_content for doc in docs]
    return "\n\n".join(passages)
