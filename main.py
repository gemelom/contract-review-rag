import sys
from config import DASHSCOPE_API_KEY
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


def run_pipeline():
    """
    Main pipeline to run the contract risk assessment project with corrected column name usage.
    """
    print(
        "--- Starting Contract Risk Assessment Pipeline (with corrected CSV column usage) ---"
    )

    if not DASHSCOPE_API_KEY:
        print("Error: DASHSCOPE_API_KEY environment variable is not set.")
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

        # Check if critical dataframes are loaded
        if contract_df is None or contract_df.empty:
            print("Fatal error: contract.csv could not be loaded or is empty.")
            sys.exit(1)
        if gt_contract_risk_df is None or gt_contract_risk_df.empty:
            print("Fatal error: GT_contract_risk.csv could not be loaded or is empty.")
            sys.exit(1)
        # Non-critical for RAG if empty, but good to note
        if checklist_df is None or checklist_df.empty:
            print(
                "Warning: checklist.csv could not be loaded or is empty. Knowledge base may be limited."
            )
        if clause_review_pair_df is None or clause_review_pair_df.empty:
            print(
                "Warning: clause_review_pair.csv could not be loaded or is empty. Knowledge base may be limited."
            )

    except Exception as e:  # Catch any other exception during data access
        print(f"Fatal error during data loading or initial DataFrame access: {e}")
        sys.exit(1)

    # 2. Build Knowledge Base Documents
    try:
        # Pass potentially empty DFs, builder handles it
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
        llm = get_llm()

        retriever = None
        if knowledge_docs:
            retriever = create_vector_store_and_retriever(
                knowledge_docs, embedding_model
            )
        else:
            print("Skipping retriever creation as knowledge base is effectively empty.")

        rag_chain = create_rag_chain(retriever, llm)
    except Exception as e:
        print(f"Fatal error initializing RAG components: {e}")
        sys.exit(1)

    # 4. Perform Risk Assessment
    # Actual header for clauses in contract.csv is 'clause'
    clause_column_name = "clause"
    if clause_column_name not in contract_df.columns:
        print(
            f"Error: Column '{clause_column_name}' not found in contract.csv. Available columns: {contract_df.columns.tolist()}"
        )
        sys.exit(1)
    contract_clauses_to_review = contract_df[clause_column_name].tolist()

    try:
        predictions = assess_risks(rag_chain, contract_clauses_to_review)
    except Exception as e:
        print(f"Fatal error during risk assessment: {e}")
        sys.exit(1)

    # 5. Evaluate Results
    # Actual header for risk labels in GT_contract_risk.csv is 'gt'
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
