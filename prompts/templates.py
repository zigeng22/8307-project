"""
Prompt templates for all three tasks.
All models use the same prompts to ensure fair comparison.
"""
from config import SENTIMENT_LABELS


LABEL_LIST_STR = " / ".join(SENTIMENT_LABELS)


# ============================================================
# Task 1: Mental health status classification
# ============================================================

TASK1_SYSTEM = (
    "You are a mental health classification assistant. "
    "Given a user's text, classify their mental health status into exactly one category."
)

TASK1_USER_TEMPLATE = (
    "Classify the following text into one of these categories:\n"
    f"{LABEL_LIST_STR}\n\n"
    "Text: \"{text}\"\n\n"
    "Respond with ONLY the category name, nothing else."
)


# ============================================================
# Task 2: Clinical dialogue generation
# ============================================================

TASK2_SYSTEM = (
    "You are a professional mental health counselor. "
    "Provide empathetic, professional, and helpful responses to patients."
)

TASK2_USER_TEMPLATE = (
    "A patient says: \"{patient_input}\"\n\n"
    "Please provide a professional and empathetic counseling response."
)


# ============================================================
# Task 3: Medical knowledge QA
# ============================================================

TASK3_SYSTEM = (
    "You are a medical knowledge assistant specializing in mental health. "
    "Answer the following medical question accurately and concisely."
)

TASK3_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Please provide a detailed and accurate answer."
)


# ============================================================
# RAG-augmented versions (prepend retrieved context)
# ============================================================

RAG_CONTEXT_PREFIX = (
    "Use the following reference information to help answer the question.\n\n"
    "Reference:\n{context}\n\n"
    "---\n\n"
)


def build_task1_messages(text: str, rag_context: str = ""):
    """Build chat messages for Task 1."""
    user_msg = TASK1_USER_TEMPLATE.format(text=text)
    if rag_context:
        user_msg = RAG_CONTEXT_PREFIX.format(context=rag_context) + user_msg
    return [
        {"role": "system", "content": TASK1_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def build_task2_messages(patient_input: str, rag_context: str = ""):
    """Build chat messages for Task 2."""
    user_msg = TASK2_USER_TEMPLATE.format(patient_input=patient_input)
    if rag_context:
        user_msg = RAG_CONTEXT_PREFIX.format(context=rag_context) + user_msg
    return [
        {"role": "system", "content": TASK2_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def build_task3_messages(question: str, rag_context: str = ""):
    """Build chat messages for Task 3."""
    user_msg = TASK3_USER_TEMPLATE.format(question=question)
    if rag_context:
        user_msg = RAG_CONTEXT_PREFIX.format(context=rag_context) + user_msg
    return [
        {"role": "system", "content": TASK3_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
