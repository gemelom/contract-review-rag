from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_predictions(true_labels, predictions):
    """
    Evaluates predictions against true labels and prints metrics.
    'A' is considered the positive label (risky).
    """
    print("\nEvaluating predictions...")
    if not true_labels:
        print("Error: No true labels provided for evaluation.")
        return
    if not predictions:
        print("Error: No predictions provided for evaluation.")
        return

    # Ensure true_labels is a list of strings, just like predictions are expected to be
    true_labels = [str(label) for label in true_labels]

    if len(predictions) != len(true_labels):
        print(
            f"Warning: Number of predictions ({len(predictions)}) does not match number of true labels ({len(true_labels)})."
        )
        print(
            "Aligning predictions to true labels for evaluation. This might affect metrics if lengths differ significantly."
        )

        aligned_predictions = predictions[: len(true_labels)]
        if len(aligned_predictions) < len(true_labels):
            # Pad with 'B' (no risk) if predictions are fewer.
            # This is a simplistic approach; ideally, an ID-based match would be better if data can be misaligned.
            aligned_predictions.extend(
                ["B"] * (len(true_labels) - len(aligned_predictions))
            )
    else:
        aligned_predictions = predictions

    # Calculate TP, FP, FN, TN
    TP = sum(
        1
        for true, pred in zip(true_labels, aligned_predictions)
        if true == "A" and pred == "A"
    )
    FP = sum(
        1
        for true, pred in zip(true_labels, aligned_predictions)
        if true == "B" and pred == "A"
    )
    FN = sum(
        1
        for true, pred in zip(true_labels, aligned_predictions)
        if true == "A" and pred == "B"
    )
    TN = sum(
        1
        for true, pred in zip(true_labels, aligned_predictions)
        if true == "B" and pred == "B"
    )

    print("\nConfusion Matrix Elements:")
    print(f"  TP (True Positives - risky clauses correctly identified): {TP}")
    print(f"  FP (False Positives - non-risky clauses misidentified as risky): {FP}")
    print(f"  FN (False Negatives - risky clauses missed): {FN}")
    print(f"  TN (True Negatives - non-risky clauses correctly identified): {TN}")

    # Calculate metrics using scikit-learn
    precision = precision_score(
        true_labels, aligned_predictions, pos_label="A", zero_division=0
    )
    recall = recall_score(
        true_labels, aligned_predictions, pos_label="A", zero_division=0
    )
    f1 = f1_score(true_labels, aligned_predictions, pos_label="A", zero_division=0)

    print("\nPerformance Metrics:")
    print(f"  Precision (P = TP / (TP + FP)): {precision:.4f}")
    print(f"  Recall    (R = TP / (TP + FN)): {recall:.4f}")
    print(f"  F1-Score  (F1 = 2 * P * R / (P + R)): {f1:.4f}")
