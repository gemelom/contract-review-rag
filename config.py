import os

# API Key for DashScope (Tongyi Qianwen)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# Model names
# Embedding model from HuggingFace
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = "cpu"  # or "cuda" if GPU is available

# LLM model from DashScope (Tongyi Qianwen)
LLM_MODEL_NAME = "qwen-plus"  # Or "qwen-turbo", "qwen-max"

# RAG configuration
VECTOR_STORE_SEARCH_K = 3  # Number of relevant documents to retrieve

# File paths (assuming CSVs are in the same directory as main.py)
CONTRACT_FILE = "data/contract.csv"
CHECKLIST_FILE = "data/checklist.csv"
CLAUSE_REVIEW_PAIR_FILE = "data/clause_review_pair.csv"
GROUND_TRUTH_FILE = "data/GT_contract_risk.csv"

PROMPT_TEMPLATE_STR = """
You are a professional contract risk assessment expert. Based on the provided background knowledge and the contract clause under review, determine if the clause contains any risks.
Background Knowledge (from risk checklists and expert cases):
---
{context}
---
Contract Clause Under Review:
---
{question}
---
Assessment: If there is a risk, answer 'A'; if there is no risk, answer 'B'. Please answer ONLY with 'A' or 'B'.
"""
