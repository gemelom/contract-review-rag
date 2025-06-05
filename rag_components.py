from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    LLM_MODEL_NAME,
    DASHSCOPE_API_KEY,
    VECTOR_STORE_SEARCH_K,
    PROMPT_TEMPLATE_STR,
)


def get_embedding_model():
    """Initializes and returns the HuggingFace embedding model."""
    print(
        f"Initializing embedding model: {EMBEDDING_MODEL_NAME} on device: {EMBEDDING_DEVICE}..."
    )
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("Embedding model initialized successfully.")
        return embeddings
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        print(
            "Please ensure 'sentence-transformers' and 'torch' are installed, and the model name is correct."
        )
        raise


def get_llm():
    """Initializes and returns the Tongyi Qianwen LLM."""
    print(f"Initializing LLM: {LLM_MODEL_NAME}...")
    if not DASHSCOPE_API_KEY:
        # This check is also in main.py, but good for defensive programming
        raise ValueError(
            "DASHSCOPE_API_KEY environment variable not set. Please set it to use the Tongyi LLM."
        )
    try:
        llm = ChatTongyi(model_name=LLM_MODEL_NAME)
        print("Tongyi LLM initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing Tongyi LLM: {e}")
        print(
            "Please ensure 'dashscope' library is installed and DASHSCOPE_API_KEY is correctly set."
        )
        raise


def create_vector_store_and_retriever(knowledge_base_documents, embeddings):
    """Creates a FAISS vector store and retriever from documents and embeddings."""
    print("Creating vector store and retriever...")
    if not knowledge_base_documents or not any(
        d.page_content for d in knowledge_base_documents
    ):
        print(
            "Warning: Knowledge base documents are empty or invalid. Cannot create vector store."
        )
        return None

    try:
        vectorstore = FAISS.from_documents(knowledge_base_documents, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": VECTOR_STORE_SEARCH_K})
        print(
            f"Vector store and retriever created. Will retrieve top {VECTOR_STORE_SEARCH_K} documents."
        )
        return retriever
    except Exception as e:
        print(f"Error creating vector store or retriever: {e}")
        return None


def format_docs_for_rag(docs):
    """Formats retrieved documents for the RAG prompt."""
    if not docs:
        return "No relevant background knowledge found."
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever, llm):
    """Creates the RAG chain."""
    print("Creating RAG chain...")
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE_STR)

    if retriever:
        rag_chain = (
            {
                "context": retriever | format_docs_for_rag,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG chain with retriever created successfully.")
    else:
        print(
            "Warning: Retriever is None. Creating RAG chain without document retrieval context."
        )
        rag_chain = (
            {
                "context": (
                    lambda x: "Warning: Retriever was not initialized. No context available."
                ),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG chain (without retriever context) created.")
    return rag_chain
