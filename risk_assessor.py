import pandas as pd


def assess_risks(rag_chain, contract_clauses):
    """
    Assesses risks for a list of contract clauses using the RAG chain.
    Returns a list of predictions ('A' or 'B').
    """
    if not contract_clauses:
        print("Warning: No contract clauses provided for assessment.")
        return []

    print(f"Starting risk assessment for {len(contract_clauses)} clauses...")
    predictions = []

    for i, clause_text in enumerate(contract_clauses):
        if pd.isna(clause_text) or not str(clause_text).strip():
            print(
                f"Info: Clause {i + 1} is empty or invalid, defaulting to 'B' (no risk)."
            )
            predictions.append("B")
            continue

        current_clause_display = str(clause_text)[:70].replace("\n", " ") + "..."
        print(
            f"Assessing clause {i + 1}/{len(contract_clauses)}: {current_clause_display}"
        )
        try:
            result = rag_chain.invoke(str(clause_text))  # Ensure input is string

            prediction = result.strip().upper()
            if prediction not in ["A", "B"]:
                print(
                    f"Warning: LLM returned an unexpected result '{prediction}' for clause {i + 1}. Defaulting to 'B'. Raw LLM output: {result}"
                )
                prediction = "B"
            predictions.append(prediction)
            # print(f"Clause {i+1} assessment complete. Prediction: {prediction}") # Can be verbose
        except Exception as e:
            print(
                f"Error assessing risk for clause {i + 1} ('{current_clause_display}'): {e}"
            )
            predictions.append("B")

    print(f"Risk assessment finished. Total predictions made: {len(predictions)}.")
    return predictions
