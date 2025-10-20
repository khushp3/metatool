import pandas as pd
import numpy as np
import src.featuretools as ft
from scipy.stats import skew, kurtosis, entropy

def extract_meta_features(column: pd.Series) -> dict:
    meta = {}
    col = column.dropna()

    # --- Basic Meta Info ---
    meta["n_missing"] = column.isnull().sum()   
    meta["missing_ratio"] = column.isnull().mean()
    meta["n_unique"] = col.nunique()
    meta["dtype"] = str(column.dtype)

    # --- Featuretools semantic type ---
    if pd.api.types.is_numeric_dtype(column):
        meta["is_numeric"] = 1
        meta["is_categorical"] = 0
        meta["is_datetime"] = 0
        meta["is_boolean"] = 0
    elif pd.api.types.is_datetime64_any_dtype(column):
        meta["is_numeric"] = 0
        meta["is_categorical"] = 0
        meta["is_datetime"] = 1
        meta["is_boolean"] = 0
    elif pd.api.types.is_bool_dtype(column):
        meta["is_numeric"] = 0
        meta["is_categorical"] = 0
        meta["is_datetime"] = 0
        meta["is_boolean"] = 1
    else:
        meta["is_numeric"] = 0
        meta["is_categorical"] = 1
        meta["is_datetime"] = 0
        meta["is_boolean"] = 0

    # --- Numerical meta-features ---
    if pd.api.types.is_numeric_dtype(column):
        meta["mean"] = col.mean()
        meta["std"] = col.std()
        meta["min"] = col.min()
        meta["max"] = col.max()
        meta["skew"] = skew(col)
        meta["kurtosis"] = kurtosis(col)
        meta["iqr"] = np.percentile(col, 75) - np.percentile(col, 25)
        meta["entropy"] = entropy(np.histogram(col, bins="auto")[0] + 1e-9)

        # normalized values
        meta["mean_norm"] = (meta["mean"] - col.min()) / (col.max() - col.min() + 1e-9)
        meta["std_norm"] = meta["std"] / (col.max() - col.min() + 1e-9)

    # --- Categorical meta-features ---
    elif pd.api.types.is_object_dtype(column) or pd.api.types.is_categorical_dtype(column):
        value_counts = col.value_counts(normalize=True)
        meta["mode_freq"] = value_counts.iloc[0] if len(value_counts) > 0 else 0
        meta["category_entropy"] = entropy(value_counts + 1e-9)
        meta["n_categories"] = len(value_counts)
        meta["max_category_ratio"] = value_counts.max() if len(value_counts) > 0 else 0
        meta["min_category_ratio"] = value_counts.min() if len(value_counts) > 0 else 0

    # --- Boolean meta-features ---
    elif pd.api.types.is_bool_dtype(column):
        counts = col.value_counts(normalize=True)
        meta["true_ratio"] = counts.get(True, 0)
        meta["false_ratio"] = counts.get(False, 0)
        meta["entropy"] = entropy(counts + 1e-9)

    # --- Datetime meta-features ---
    elif pd.api.types.is_datetime64_any_dtype(column):
        meta["range_days"] = (col.max() - col.min()).days if len(col) > 0 else 0
        meta["year_variety"] = col.dt.year.nunique() if len(col) > 0 else 0
        meta["month_variety"] = col.dt.month.nunique() if len(col) > 0 else 0

    # --- General meta-features ---
    meta["n_total"] = len(column)
    meta["valid_ratio"] = 1 - meta["missing_ratio"]

    # --- Normalization ---
    for k, v in meta.items():
        if isinstance(v, (int, float)) and np.isfinite(v):
            meta[k] = round(float(v), 6)
        elif not np.isfinite(v):
            meta[k] = 0.0

    return meta