import pandas as pd
from config import (
    CONTRACT_FILE,
    CHECKLIST_FILE,
    CLAUSE_REVIEW_PAIR_FILE,
    GROUND_TRUTH_FILE,
)


def load_all_data():
    """
    Loads all necessary CSV files into pandas DataFrames.
    Returns a dictionary of DataFrames.
    """
    print("Loading data...")
    all_data = {}
    files_to_load = {
        "contract": CONTRACT_FILE,
        "checklist": CHECKLIST_FILE,
        "clause_review_pair": CLAUSE_REVIEW_PAIR_FILE,
        "ground_truth": GROUND_TRUTH_FILE,
    }

    success = True
    for key, file_name in files_to_load.items():
        try:
            df = pd.read_csv(file_name)
            all_data[key] = df
            print(
                f"Successfully loaded '{file_name}' ({len(df)} rows). Columns: {df.columns.tolist()}"
            )
        except FileNotFoundError:
            print(
                f"Error: File not found - '{file_name}'. Please ensure it's in the correct path."
            )
            success = False
            all_data[key] = pd.DataFrame()  # Return empty DataFrame on error
        except Exception as e:
            print(f"An error occurred while loading '{file_name}': {e}")
            success = False
            all_data[key] = pd.DataFrame()  # Return empty DataFrame on error

    if success:
        print("All CSV files processed for loading.")
    else:
        print("One or more CSV files could not be loaded. Please check errors above.")
        # Optionally raise an exception if loading any file is critical
        # raise FileNotFoundError("Critical data file(s) not found.")

    return all_data


if __name__ == "__main__":
    # For testing data loading
    data_frames = load_all_data()
    for name, df in data_frames.items():
        print(f"\nDataFrame: {name}")
        if not df.empty:
            print(df.head())
        else:
            print("DataFrame is empty (failed to load).")
