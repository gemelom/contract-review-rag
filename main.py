import sys
from config import DASHSCOPE_API_KEY

from config import (
    LANGCHAIN_TRACING_V2,
    LANGCHAIN_API_KEY_CONFIG,
    LANGCHAIN_PROJECT_CONFIG,
    LANGCHAIN_ENDPOINT_CONFIG,
)

from data_loader import load_all_data
from knowledge_base_builder import build_knowledge_base_documents
from rag_components import (
    get_embedding_model,
    get_llm,
    create_vector_store_and_retriever,
    create_rag_chain,
)
from risk_assessor import assess_risks
from evaluator import evaluate_predictions


def check_langsmith_status():
    """Checks and prints LangSmith tracing status based on environment variables."""
    print("\n--- LangSmith Tracing Status ---")
    if LANGCHAIN_TRACING_V2 == "true" and LANGCHAIN_API_KEY_CONFIG:
        project_name = (
            LANGCHAIN_PROJECT_CONFIG if LANGCHAIN_PROJECT_CONFIG else "Default Project"
        )
        endpoint_url = (
            LANGCHAIN_ENDPOINT_CONFIG
            if LANGCHAIN_ENDPOINT_CONFIG
            else "https://api.smith.langchain.com"
        )

        print("Status: ACTIVE")
        print(f"  Project: {project_name}")
        print(f"  Endpoint: {endpoint_url}")
        print("  (LANGCHAIN_API_KEY is set)")
        print("  LLM calls and LangChain component runs should be traced to LangSmith.")
    else:
        print("Status: INACTIVE")
        print(
            "  To enable LangSmith tracing, please set the following environment variables:"
        )
        print('    export LANGCHAIN_TRACING_V2="true"')
        print('    export LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY"')
        print('    export LANGCHAIN_PROJECT="Your Project Name"  (Optional)')
        print(
            '    export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com" (Optional, default)'
        )
    print("---")


def run_pipeline():
    """
    Main pipeline to run the contract risk assessment project.
    """
    print("--- Starting Contract Risk Assessment Pipeline ---")

    # Check LangSmith status early for user awareness
    check_langsmith_status()

    if not DASHSCOPE_API_KEY:
        print(
            "Error: DASHSCOPE_API_KEY environment variable is not set for Tongyi LLM."
        )
        print(
            "Please set this variable with your Alibaba Cloud DashScope API key to proceed."
        )
        sys.exit(1)

    # 1. Load Data
    try:
        data_frames = load_all_data()
        contract_df = data_frames.get("contract")
        checklist_df = data_frames.get("checklist")
        clause_review_pair_df = data_frames.get("clause_review_pair")
        gt_contract_risk_df = data_frames.get("ground_truth")

        if contract_df is None or contract_df.empty:
            print("Fatal error: contract.csv could not be loaded or is empty.")
            sys.exit(1)
        if gt_contract_risk_df is None or gt_contract_risk_df.empty:
            print("Fatal error: GT_contract_risk.csv could not be loaded or is empty.")
            sys.exit(1)
        if checklist_df is None or checklist_df.empty:
            print("Warning: checklist.csv could not be loaded or is empty.")
        if clause_review_pair_df is None or clause_review_pair_df.empty:
            print("Warning: clause_review_pair.csv could not be loaded or is empty.")

    except Exception as e:
        print(f"Fatal error during data loading or initial DataFrame access: {e}")
        sys.exit(1)

    # 2. Build Knowledge Base Documents
    try:
        knowledge_docs = build_knowledge_base_documents(
            checklist_df, clause_review_pair_df
        )
        if not knowledge_docs:
            print(
                "Warning: Knowledge base is empty. RAG will proceed without retrieved context."
            )
    except Exception as e:
        print(f"Fatal error during knowledge base building: {e}")
        sys.exit(1)

    # 3. Initialize RAG Components
    try:
        embedding_model = get_embedding_model()
        llm = get_llm()  # LangSmith will automatically trace calls to this LLM

        retriever = None
        if knowledge_docs:
            # LangSmith will trace calls made by/to this retriever
            retriever = create_vector_store_and_retriever(
                knowledge_docs, embedding_model
            )
        else:
            print("Skipping retriever creation as knowledge base is effectively empty.")

        # LangSmith will trace calls to this chain and its components
        rag_chain = create_rag_chain(retriever, llm)
    except Exception as e:
        print(f"Fatal error initializing RAG components: {e}")
        sys.exit(1)

    # 4. Perform Risk Assessment
    clause_column_name = "clause"
    if clause_column_name not in contract_df.columns:
        print(
            f"Error: Column '{clause_column_name}' not found in contract.csv. Available columns: {contract_df.columns.tolist()}"
        )
        sys.exit(1)
    contract_clauses_to_review = contract_df[clause_column_name].tolist()

    try:
        # Each call to rag_chain.invoke will be a separate run in LangSmith
        predictions = assess_risks(rag_chain, contract_clauses_to_review)
    except Exception as e:
        print(f"Fatal error during risk assessment: {e}")
        sys.exit(1)

    # 5. Evaluate Results
    ground_truth_column_name = "gt"
    if ground_truth_column_name not in gt_contract_risk_df.columns:
        print(
            f"Error: Column '{ground_truth_column_name}' not found in GT_contract_risk.csv. Available columns: {gt_contract_risk_df.columns.tolist()}"
        )
        sys.exit(1)
    true_labels = gt_contract_risk_df[ground_truth_column_name].tolist()

    try:
        evaluate_predictions(true_labels, predictions)
    except Exception as e:
        print(f"Error during evaluation: {e}")

    print("\n--- Contract Risk Assessment Pipeline Finished ---")


if __name__ == "__main__":
    run_pipeline()
