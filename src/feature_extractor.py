import pandas as pd
import numpy as np
from typing import Dict, Any
import scipy.stats as stats
from sklearn.feature_selection import f_classif, f_regression
from scipy.stats import chi2_contingency, entropy
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import CategoricalDtype

# Class to extract dataset-level and column-level features
class FeatureExtractor:
    def __init__(self):
        pass

    # Function to extract dataset-level features from the loaded datasets
    def extract_dataset_features(self, df: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        # Dictionary to store the dataset-level features
        dataset_features = {}

        # Number of rows and columns in the dataset
        dataset_features["num_rows"] = df.shape[0]
        dataset_features["num_columns"] = df.shape[1]

        # Type of features in the dataset
        dataset_features["num_numeric"] = df.select_dtypes(include=[np.number]).shape[1]
        dataset_features["num_categorical"] = df.select_dtypes(include=['object', 'category']).shape[1]
        dataset_features["num_boolean"] = df.select_dtypes(include=[bool]).shape[1]
        dataset_features["num_datetime"] = df.select_dtypes(include=['datetime64', 'timedelta64']).shape[1]
        dataset_features["feature_to_row_ratio"] = df.shape[1] / max(df.shape[0], 1)

        # Missing values statistics 
        dataset_features["missing_total"] = df.isnull().sum().sum()
        dataset_features["missing_mean"] = df.isnull().mean().mean()
        dataset_features["missing_standard"] = df.isnull().mean().std()
        dataset_features["missing_max_column"] = df.isnull().mean().max()

        # Correlation for numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            # Computes the correlation matrix
            correlation_matrix = numeric_df.corr().abs()
            # Gets the upper triangle of the correlation matrix
            upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            # Calculates the mean, max, and number of high correlation pairs
            dataset_features["mean_correlation"] = upper_tri.stack().mean()
            dataset_features["max_correlation"] = upper_tri.stack().max()
            dataset_features["num_correlation_pairs"] = (upper_tri > 0.9).sum().sum()
        else:
            # If there are 1 or no numeric features, correlation features are set to 0
            dataset_features["mean_correlation"] = 0    
            dataset_features["max_correlation"] = 0
            dataset_features["num_correlation_pairs"] = 0

        # If there is a target variable, calculate the relationship between features and target
        if target is not None:
            # Calculates the number of unique classes in the target variable
            dataset_features["target_unique_classes"] = target.nunique()
            # Calculates the entropy and imbalance ratio of the target variable
            dataset_features["target_entropy"] = entropy(target.value_counts(normalize=True), base=2)
            dataset_features["target_imbalance_ratio"] = (
                target.value_counts(normalize=True).max() / max(target.value_counts(normalize=True).min(), 1e-5)
                if target.nunique() > 1
                else 1
            )
        # Calculates and adds cardinality statistics
        dataset_features["cardinality_average"] = df.nunique().mean()
        dataset_features["cardinality_median"] = df.nunique().median()
        dataset_features["cardinality_standard"] = df.nunique().std()

        # If there are numeric features, it calculates skewness, kurtosis, and entropy
        if numeric_df.shape[1] > 0:
            dataset_features["skewness_mean"] = numeric_df.skew().mean()
            dataset_features["kurtosis_mean"] = numeric_df.kurt().mean()
            entropies = []
            for col in numeric_df.columns:
                hist, _ = np.histogram(numeric_df[col].dropna(), bins=10)
                if hist.sum() > 0:
                    entropies.append(entropy(hist + 1)) 
            dataset_features["entropy_mean"] = np.mean(entropies) if entropies else 0
            dataset_features["sparsity_ratio"] = (numeric_df == 0).sum().sum() / (numeric_df.shape[0] * numeric_df.shape[1])
        else:
            # If there are no numeric features, these statistics are set to NaN
            dataset_features["skewness_mean"] = dataset_features["kurtosis_mean"] = dataset_features["entropy_mean"] = np.nan
            dataset_features["sparsity_ratio"] = 0

        return dataset_features

    # Function to extract column-level features from the loaded datasets
    def extract_column_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        # Dictionary to store the column-level features for each column
        column_features = {}

        # Iterate through each column in the DataFrame to store column data and its type
        for column_name in X.columns:
            column_data = X[column_name]
            column_type = str(column_data.dtype)

            # Initialise a dictionary to hold standard features applicable to all object types
            current_features = {
                # Stores the type of the column
                "data_type": column_type,
                "is_numeric": pd.api.types.is_numeric_dtype(column_data),
                "is_categorical": isinstance(column_data.dtype, CategoricalDtype) or pd.api.types.is_object_dtype(column_data) or pd.api.types.is_string_dtype(column_data),

                # Basic properties for all columns
                "number_unique": column_data.nunique(),
                "unique_percent": column_data.nunique() / len(column_data),
                "is_constant": column_data.nunique() <= 1,
                "number_missing": column_data.isnull().sum(),
                "missing_percent": column_data.isnull().mean(),
            }
            # If the column is numerical, the following features are added to the dictionary
            if current_features["is_numeric"]:
                current_features.update({
                    "mean": column_data.mean(),
                    "std": column_data.std(),
                    "min": column_data.min(),
                    '25%': column_data.quantile(0.25),
                    '50%': column_data.quantile(0.50),
                    '75%': column_data.quantile(0.75),
                    "max": column_data.max(),
                    "iqr": column_data.quantile(0.75) - column_data.quantile(0.25),
                    "skewness": column_data.skew(),
                    "kurtosis": column_data.kurt(),
                    'outlier_ratio': self.outlier(column_data),
                    "has_infinity": np.isinf(column_data).any(),
                    "entropy": entropy(np.histogram(column_data.dropna(), bins=10)[0] + 1),
                    'normality_pvalue': stats.normaltest(column_data).pvalue if len(column_data) > 20 else np.nan,
                })
            # If the column is categorical, the following features are added to the dictionary
            elif current_features["is_categorical"]:
                values = column_data.value_counts(dropna=True, normalize=True)
                current_features.update({
                    "cardinality": len(values),
                    "most_frequent_value": values.index[0] if len(values) > 0 else None,
                    "most_frequent_percent": values.iloc[0] if len(values) > 0 else None,
                    "most_frequent_count": values.iloc[0] if not values.empty else None,
                    "least_frequent_value": values.index[-1] if len(values) > 1 else None,
                    "least_frequent_percent": values.iloc[-1] if len(values) > 1 else None,
                    "least_frequent_count": values.iloc[-1] if not values.empty else None,
                    "frequency_ratio": values.iloc[0] / values.iloc[-1] if len(values) > 1 else None,
                    "frequency_entropy": stats.entropy(values, base=2) if not values.empty else None,
                    "most_frequent_class": column_data.mode().iloc[0] if not column_data.mode().empty else None,
                    "rare_class_ratio": (values < 0.05).sum() / len(values) if len(values) > 0 else None,
                    "majority_class_ratio": (values > 0.5).sum() / len(values) if len(values) > 0 else None,
                    "gini_index": 1 - (values ** 2).sum() if not values.empty else None,
                })
            # If a target variable is provided, its relationship with the column is calculated
            if y is not None:
                try:
                    current_features.update(self.calculate_target_relationship(column_data, y))
                except Exception as e:
                    print(f"Error calculating target relationship for {column_name}: {e}")
            # The features for the current column are added to the main dictionary
            column_features[column_name] = current_features
        # The complete dictionary of column-level features is returned
        return column_features
    
    # Helper function to calculate the outlier ratio in a numerical column
    def outlier(self, data: pd.Series) -> float:
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return 0.0
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((data < lower) | (data > upper)).sum()
        return outliers / len(data)

    # Helper function to calculate the relationship between feature and the target variable
    def calculate_target_relationship(self, feature: pd.Series, target: pd.Series) -> Dict[str, float]:  
        # Dictionary to store relationship metrics
        relationship = {}
        
        # Handle missing values
        valid_index = feature.notna()
        feature_clean = feature[valid_index]
        target_clean = target[valid_index]

        if len(feature_clean) == 0:
            return relationship
        
        # Numeric feature with categorical target
        if pd.api.types.is_numeric_dtype(feature_clean) and isinstance(target_clean.dtype, pd.CategoricalDtype):
            try:
                f_stat, p_value = f_classif(feature_clean.values.reshape(-1, 1), target_clean)
                relationship['f_statistic'] = f_stat[0]
                relationship['f_pvalue'] = p_value[0]
            except:
                pass
        
        # Categorical feature with categorical target
        elif isinstance(feature_clean.dtype, pd.CategoricalDtype) and isinstance(target_clean.dtype, pd.CategoricalDtype):
            try:
                contingency = pd.crosstab(feature_clean, target_clean)
                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    chi2, p_val, _, _ = chi2_contingency(contingency)
                    relationship["chi2_statistic"] = chi2
                    relationship["chi2_pvalue"] = p_val
            except:
                pass
        
        # Numeric feature with numeric target
        elif pd.api.types.is_numeric_dtype(target_clean) and pd.api.types.is_numeric_dtype(feature_clean):
            try:
                pearson_corr = feature_clean.corr(pd.Series(target_clean))
                spearman_corr = feature_clean.corr(pd.Series(target_clean), method="spearman")
                relationship['pearson_correlation'] = pearson_corr
                relationship['spearman_correlation'] = spearman_corr
            except:
                pass
        
        return relationship


# Testing the feature extraction
if __name__ == "__main__":
    df = pd.DataFrame({
        "age": [22, 35, np.nan, 40, 60, 45],
        "gender": ["M", "F", "F", "M", "F", np.nan],
        "income": [3000, 5000, 4500, 7000, 6000, 6500],
        "joined": pd.date_range("2020-01-01", periods=6, freq="YE")
    })
    y = pd.Series([1, 0, 0, 1, 1, 0])

    fe = FeatureExtractor()
    ds_feats = fe.extract_dataset_features(df, y)
    col_feats = fe.extract_column_features(df, y)

    print("DATASET FEATURES:\n", ds_feats)
    print("\nCOLUMN FEATURES (age):\n", col_feats["age"])