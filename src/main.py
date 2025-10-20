import pandas as pd
from dataset_loader import DatasetLoader
from feature_extractor import FeatureExtractor

import os

def main():
    # === Step 1: Load a few benchmark datasets from OpenML ===
    print("\n🔹 Loading datasets from OpenML-CC18 benchmark suite...\n")
    datasets = DatasetLoader().get_benchmark_suite(suite_id=99, max_datasets=10)
    print(f"Loaded {len(datasets)} datasets.\n")

    extractor = FeatureExtractor()
    meta_records = []

    # === Step 2: Extract dataset- and column-level meta-features ===
    for name, (df, target_col) in datasets.items():
        print(f"📊 Processing dataset: {name}")

        try:
            # Extract target series from dataframe
            y = df[target_col]
            X = df.drop(columns=[target_col])
            
            # Dataset-level meta-features
            dataset_meta = extractor.extract_dataset_features(X, y)
            dataset_meta["dataset_name"] = name

            # Column-level meta-features
            column_meta = extractor.extract_column_features(X, y)

            # Combine into one unified structure
            for col, feats in column_meta.items():
                record = {"dataset_name": name, "column_name": col}
                record.update(dataset_meta)
                record.update(feats)
                meta_records.append(record)

            print(f"✅ Extracted {len(column_meta)} column-level feature sets for {name}\n")

        except Exception as e:
            print(f"⚠️ Failed to process {name}: {e}\n")
            continue

    # === Step 3: Save meta-dataset ===
    if meta_records:
        # CREATE DIRECTORY IF IT DOESN'T EXIST ← ADD THIS LINE
        os.makedirs("meta_datasets", exist_ok=True)
        
        meta_df = pd.DataFrame(meta_records)
        meta_df.to_csv("meta_datasets/column_meta_features.csv", index=False)
        print(f"💾 Saved meta-dataset with {meta_df.shape[0]} rows and {meta_df.shape[1]} features.")
    else:
        print("⚠️ No meta-features extracted. Please check dataset or extractor logic.")

    # Optional: pretty-print a few examples
    print("\n🔍 Sample extracted meta-features:")
    print(meta_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()