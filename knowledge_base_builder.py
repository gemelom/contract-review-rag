from langchain.docstore.document import Document
import pandas as pd


def build_knowledge_base_documents(checklist_df, clause_review_pair_df):
    """
    Creates a list of LangChain Document objects from the checklist and clause review DataFrames
    using the corrected column names.
    """
    print("Building knowledge base documents with corrected headers...")
    knowledge_base_documents = []

    # Process checklist.csv
    # Actual headers: Risk_feature, Identified_items
    if not checklist_df.empty:
        print(f"Processing checklist_df with columns: {checklist_df.columns.tolist()}")
        for index, row in checklist_df.iterrows():
            risk_feature = row.get("Risk_feature", "N/A")  # Use .get for safety
            identified_items = row.get("Identified_items", "N/A")

            content = (
                f"Risk Feature: {risk_feature}\nIdentified Items: {identified_items}"
            )
            metadata = {"source": "checklist", "identifier": str(risk_feature)}
            knowledge_base_documents.append(
                Document(page_content=content, metadata=metadata)
            )
    else:
        print("Warning: checklist_df is empty. No documents will be created from it.")

    # Process clause_review_pair.csv
    # Actual headers: checkpoint, clause, review
    if not clause_review_pair_df.empty:
        print(
            f"Processing clause_review_pair_df with columns: {clause_review_pair_df.columns.tolist()}"
        )
        for index, row in clause_review_pair_df.iterrows():
            checkpoint_id = row.get("checkpoint", "N/A")
            case_clause = row.get("clause", "N/A")
            expert_review = row.get("review", "N/A")

            content = f"Related Checkpoint ID: {checkpoint_id}\nCase Clause Example: {case_clause}\nExpert Review/Comment: {expert_review}"
            metadata = {
                "source": "clause_review_pair",
                "checkpoint_id": str(checkpoint_id),
            }
            knowledge_base_documents.append(
                Document(page_content=content, metadata=metadata)
            )
    else:
        print(
            "Warning: clause_review_pair_df is empty. No documents will be created from it."
        )

    if not knowledge_base_documents:
        print(
            "Warning: No knowledge base documents were created in total. Check input CSV files and headers."
        )

    print(f"Total knowledge base documents created: {len(knowledge_base_documents)}")
    if knowledge_base_documents:
        print("Sample of first knowledge base document content:")
        print(knowledge_base_documents[0].page_content)
    return knowledge_base_documents


if __name__ == "__main__":
    dummy_checklist_data = {
        "Risk_feature": ["High Default Risk", "IP Leakage"],
        "Identified_items": [
            "No cure period for default",
            "Vague IP ownership clauses",
        ],
    }
    dummy_clause_review_data = {
        "checkpoint": ["CHK001", "CHK002"],
        "clause": [
            "Sample clause about termination...",
            "Another sample regarding IP rights...",
        ],
        "review": [
            "This clause is one-sided.",
            "Ambiguity in IP rights needs clarification.",
        ],
    }

    dummy_checklist_df = pd.DataFrame(dummy_checklist_data)
    dummy_clause_review_df = pd.DataFrame(dummy_clause_review_data)

    documents = build_knowledge_base_documents(
        dummy_checklist_df, dummy_clause_review_df
    )
    if documents:
        print("\n--- Sample Documents for Testing ---")
        print("\nSample document from checklist (Corrected):")
        print(f"Content: {documents[0].page_content}")
        print(f"Metadata: {documents[0].metadata}")
        if len(documents) > len(dummy_checklist_df):
            print("\nSample document from clause review pair (Corrected):")
            print(f"Content: {documents[len(dummy_checklist_df)].page_content}")
            print(f"Metadata: {documents[len(dummy_checklist_df)].metadata}")
