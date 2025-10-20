import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.feature_selection import f_classif, f_regression
from scipy.stats import chi2_contingency, entropy
from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder


class FeatureExtractor:
    def __init__(self):
        pass

    # ===============================================================
    # 1️⃣ DATASET-LEVEL META-FEATURES
    # ===============================================================
    def extract_dataset_features(self, df: pd.DataFrame, target: pd.Series = None) -> Dict[str, Any]:
        """Extract global dataset-level meta-features"""
        feats = {}

        # --- Basic structure
        feats["n_rows"] = df.shape[0]
        feats["n_columns"] = df.shape[1]
        feats["n_numeric"] = df.select_dtypes(include=[np.number]).shape[1]
        feats["n_categorical"] = df.select_dtypes(include=["object", "category"]).shape[1]
        feats["n_boolean"] = df.select_dtypes(include=["bool"]).shape[1]
        feats["n_datetime"] = df.select_dtypes(include=["datetime"]).shape[1]
        feats["feature_to_sample_ratio"] = df.shape[1] / max(df.shape[0], 1)

        # --- Missingness patterns
        feats["missing_total"] = df.isnull().sum().sum()
        feats["missing_mean"] = df.isnull().mean().mean()
        feats["missing_std"] = df.isnull().mean().std()
        feats["missing_max_col"] = df.isnull().mean().max()

        # --- Data redundancy & correlation
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            feats["mean_correlation"] = upper_tri.stack().mean()
            feats["max_correlation"] = upper_tri.stack().max()
            feats["n_high_corr_pairs"] = (upper_tri > 0.9).sum().sum()
        else:
            feats["mean_correlation"] = 0
            feats["max_correlation"] = 0
            feats["n_high_corr_pairs"] = 0

        # --- Target-level meta-features (if classification)
        if target is not None:
            feats["target_n_classes"] = target.nunique()
            feats["target_entropy"] = entropy(target.value_counts(normalize=True), base=2)
            feats["target_imbalance_ratio"] = (
                target.value_counts(normalize=True).max() / max(target.value_counts(normalize=True).min(), 1e-5)
                if target.nunique() > 1
                else 1
            )

        # --- Dataset complexity measures
        feats["avg_cardinality"] = df.nunique().mean()
        feats["median_cardinality"] = df.nunique().median()
        feats["std_cardinality"] = df.nunique().std()

        # --- Statistical spread across numeric features
        if numeric_df.shape[1] > 0:
            feats["mean_skewness"] = numeric_df.skew().mean()
            feats["mean_kurtosis"] = numeric_df.kurt().mean()
            feats["numeric_entropy_mean"] = np.mean(
                [entropy(np.histogram(col.dropna(), bins=10)[0] + 1) for col in numeric_df.T]
            )
        else:
            feats["mean_skewness"] = feats["mean_kurtosis"] = feats["numeric_entropy_mean"] = np.nan

        return feats

    # ===============================================================
    # 2️⃣ COLUMN-LEVEL META-FEATURES
    # ===============================================================
    def extract_column_features(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, Dict[str, Any]]:
        """Extract meta-features for each column individually"""
        all_features = {}

        for col in X.columns:
            col_data = X[col]
            f = {}
            f["dtype"] = str(col_data.dtype)
            f["is_numeric"] = pd.api.types.is_numeric_dtype(col_data)
            f["is_categorical"] = pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data)

            # --- Basic structure
            f["n_missing"] = col_data.isnull().sum()
            f["missing_percent"] = col_data.isnull().mean()
            f["n_unique"] = col_data.nunique(dropna=True)
            f["unique_percent"] = f["n_unique"] / len(col_data)
            f["is_constant"] = f["n_unique"] <= 1

            # =====================================================
            # NUMERIC COLUMN FEATURES
            # =====================================================
            if f["is_numeric"]:
                desc = col_data.describe(percentiles=[0.25, 0.5, 0.75])
                f.update({
                    "mean": desc["mean"],
                    "std": desc["std"],
                    "min": desc["min"],
                    "25%": desc["25%"],
                    "50%": desc["50%"],
                    "75%": desc["75%"],
                    "max": desc["max"],
                    "iqr": desc["75%"] - desc["25%"],
                    "skewness": col_data.skew(),
                    "kurtosis": col_data.kurt(),
                    "outlier_ratio": self._outlier_ratio(col_data),
                    "has_infinity": np.isinf(col_data).any(),
                    "normality_p": stats.normaltest(col_data.dropna()).pvalue if len(col_data.dropna()) > 20 else np.nan,
                    "entropy_hist": entropy(np.histogram(col_data.dropna(), bins=10)[0] + 1)
                })

            # =====================================================
            # CATEGORICAL COLUMN FEATURES
            # =====================================================
            elif f["is_categorical"]:
                vc = col_data.value_counts(dropna=True, normalize=True)
                f.update({
                    "cardinality": len(vc),
                    "top_freq": vc.iloc[0] if len(vc) > 0 else np.nan,
                    "freq_ratio": vc.iloc[0] / vc.iloc[-1] if len(vc) > 1 else 1,
                    "entropy": entropy(vc + 1e-9, base=2),
                    "most_freq_class": col_data.mode()[0] if not col_data.mode().empty else None,
                    "rare_class_ratio": (vc < 0.05).sum() / len(vc) if len(vc) > 0 else 0
                })

            # =====================================================
            # RELATIONSHIP WITH TARGET
            # =====================================================
            if y is not None:
                try:
                    f.update(self._calculate_target_relationship(col_data, y))
                except Exception:
                    pass

            all_features[col] = f

        return all_features

    # ===============================================================
    # 🔍 Helper Functions
    # ===============================================================
    def _calculate_target_relationship(self, feature: pd.Series, target: pd.Series) -> Dict[str, float]:
        """Compute feature-target relationship metrics"""
        rel = {}

        # Handle missing values
        valid_idx = feature.notna() & target.notna()
        feature_clean = feature[valid_idx]
        target_clean = target[valid_idx]

        # Encode categorical target if needed
        if not pd.api.types.is_numeric_dtype(target_clean):
            le = LabelEncoder()
            target_clean = le.fit_transform(target_clean.astype(str))

        # Numeric feature & numeric target
        if pd.api.types.is_numeric_dtype(feature_clean) and pd.api.types.is_numeric_dtype(target_clean):
            rel["pearson_corr"] = feature_clean.corr(pd.Series(target_clean))
            rel["spearman_corr"] = feature_clean.corr(pd.Series(target_clean), method="spearman")

        # Numeric feature & categorical target
        elif pd.api.types.is_numeric_dtype(feature_clean):
            f_stat, p_val = f_classif(feature_clean.values.reshape(-1, 1), target_clean)
            rel["f_statistic"] = f_stat[0]
            rel["f_pvalue"] = p_val[0]

        # Categorical feature & categorical target
        elif pd.api.types.is_object_dtype(feature_clean) or pd.api.types.is_categorical_dtype(feature_clean):
            contingency = pd.crosstab(feature_clean, target_clean)
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                chi2, p_val, _, _ = chi2_contingency(contingency)
                rel["chi2_statistic"] = chi2
                rel["chi2_pvalue"] = p_val

        return rel

    def _outlier_ratio(self, data: pd.Series) -> float:
        """Compute outlier ratio using IQR"""
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return ((data < lower) | (data > upper)).mean()


# ===============================================================
# 🧪 Quick Test
# ===============================================================
if __name__ == "__main__":
    df = pd.DataFrame({
        "age": [22, 35, np.nan, 40, 60, 45],
        "gender": ["M", "F", "F", "M", "F", np.nan],
        "income": [3000, 5000, 4500, 7000, 6000, 6500],
        "joined": pd.date_range("2020-01-01", periods=6, freq="Y")
    })
    y = pd.Series([1, 0, 0, 1, 1, 0])

    fe = FeatureExtractor()
    ds_feats = fe.extract_dataset_features(df, y)
    col_feats = fe.extract_column_features(df, y)

    print("DATASET FEATURES:\n", ds_feats)
    print("\nCOLUMN FEATURES (age):\n", col_feats["age"])
