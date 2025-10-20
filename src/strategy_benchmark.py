import os
import warnings
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
import warnings
import math
import joblib

# Target Encoder implementation
class SimpleTargetEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.mapping_ = {}
        self.global_mean_ = None

    def fit(self, X, y):
        # X expected to be 1-D array-like or DataFrame/Series with single column
        if isinstance(X, pd.DataFrame):
            col = X.iloc[:, 0]
        else:
            col = pd.Series(X).astype(object)

        self.global_mean_ = float(pd.Series(y).mean())
        agg = pd.DataFrame({'cat': col, 'y': y}).groupby('cat')['y'].agg(['mean', 'count'])
        # smoothing to avoid overfitting to rare categories
        smooth = (agg['count'] * agg['mean'] + self.smoothing * self.global_mean_) / (agg['count'] + self.smoothing)
        self.mapping_ = smooth.to_dict()
        return self

    def transform(self, X):
        if self.global_mean_ is None:
            raise NotFittedError("SimpleTargetEncoder instance is not fitted yet")
        if isinstance(X, pd.DataFrame):
            col = X.iloc[:, 0]
        else:
            col = pd.Series(X).astype(object)
        return col.map(self.mapping_).fillna(self.global_mean_).to_numpy().reshape(-1, 1)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


def _make_numeric_pipeline(imputer_strategy: str, scaler: Optional[str], use_knn: bool = False, knn_neighbors: int = 5):
    steps = []
    if use_knn:
        steps.append(("imputer", KNNImputer(n_neighbors=knn_neighbors)))
    else:
        if imputer_strategy == "constant":
            steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0)))
        else:
            steps.append(("imputer", SimpleImputer(strategy=imputer_strategy)))

    if scaler is not None:
        if scaler == "standard":
            steps.append(("scaler", StandardScaler()))
        elif scaler == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        elif scaler == "robust":
            steps.append(("scaler", RobustScaler()))

    # classifier/regressor will be appended by caller
    return Pipeline(steps)


def _make_categorical_pipeline(imputer_strategy: str, encoder: str, use_target_encode: bool = False):
    """
    Create pipeline for categorical single-column input.
    encoder: 'onehot'|'ordinal'|'target'
    """
    steps = []
    # imputation
    if imputer_strategy == "most_frequent":
        steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    elif imputer_strategy == "constant":
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="missing")))
    else:
        steps.append(("imputer", SimpleImputer(strategy="most_frequent")))

    # encoder
    if use_target_encode or encoder == "target":
        # target encoder needs y during fit; we will wrap it in a custom pipeline application
        steps.append(("encoder", SimpleTargetEncoder()))
    else:
        if encoder == "onehot":
            steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)))
        elif encoder == "ordinal":
            steps.append(("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
        else:
            steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)))

    return Pipeline(steps)


def _evaluate_pipeline_for_column(X_col: pd.DataFrame, y: pd.Series, pipe: Pipeline, problem_type: str, cv: int = 3, scoring: str = "f1_macro", n_jobs: int = 1) -> float:
    """
    Given X_col (DataFrame with single column) and a pipeline that includes an estimator at the end,
    return mean CV score.
    """
    if problem_type == "classification":
        cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)

    # cross_val_score will call fit on pipeline; ensure pipeline is proper
    try:
        scores = cross_val_score(pipe, X_col, y, cv=cv_obj, scoring=scoring, n_jobs=n_jobs)
        return float(np.mean(scores))
    except Exception as e:
        warnings.warn(f"Pipeline evaluation failed: {e}")
        return -np.inf


# ----------------------
# Strategy space
# ----------------------
DEFAULT_NUMERIC_STRATEGIES = [
    {"name": "mean_impute_raw", "imputer": "mean", "scaler": None, "use_knn": False},
    {"name": "mean_impute_standard", "imputer": "mean", "scaler": "standard", "use_knn": False},
    {"name": "median_impute_raw", "imputer": "median", "scaler": None, "use_knn": False},
    {"name": "median_impute_standard", "imputer": "median", "scaler": "standard", "use_knn": False},
    {"name": "knn_impute_standard", "imputer": "mean", "scaler": "standard", "use_knn": True},
    {"name": "constant_impute_minmax", "imputer": "constant", "scaler": "minmax", "use_knn": False},
    {"name": "robust_impute_robust", "imputer": "median", "scaler": "robust", "use_knn": False},
]

DEFAULT_CATEGORICAL_STRATEGIES = [
    {"name": "most_frequent_impute_onehot", "imputer": "most_frequent", "encoder": "onehot", "use_target": False},
    {"name": "constant_impute_onehot", "imputer": "constant", "encoder": "onehot", "use_target": False},
    {"name": "most_frequent_impute_ordinal", "imputer": "most_frequent", "encoder": "ordinal", "use_target": False},
    {"name": "most_frequent_impute_target", "imputer": "most_frequent", "encoder": "target", "use_target": True},
    {"name": "constant_impute_target", "imputer": "constant", "encoder": "target", "use_target": True},
]

# ----------------------
# Public API: benchmark_column
# ----------------------
def benchmark_column(
    df: pd.DataFrame,
    column: str,
    target: str,
    problem_type: str = "classification",
    strategies: Optional[List[Dict[str, Any]]] = None,
    cv: int = 3,
    scoring: Optional[str] = None,
    n_jobs: int = 1,
    bo: bool = False,
    bo_iterations: int = 15
) -> Dict[str, Any]:

    if scoring is None:
        scoring = "f1_macro" if problem_type == "classification" else "r2"

    if strategies is None:
        # choose numeric or categorical defaults based on dtype
        if pd.api.types.is_numeric_dtype(df[column]):
            strategies = DEFAULT_NUMERIC_STRATEGIES
        else:
            strategies = DEFAULT_CATEGORICAL_STRATEGIES

    X_col = df[[column]].copy()
    y = df[target].copy()

    results = {}
    estimator = None

    # choose fast but reasonable baseline estimator
    if problem_type == "classification":
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    else:
        estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)

    # iterate strategies
    for strat in strategies:
        name = strat.get("name", "unnamed")
        try:
            if pd.api.types.is_numeric_dtype(df[column]):
                # numeric strategy
                pipe_pre = _make_numeric_pipeline(
                    imputer_strategy=strat.get("imputer", "mean"),
                    scaler=strat.get("scaler", None),
                    use_knn=strat.get("use_knn", False),
                    knn_neighbors=strat.get("knn_neighbors", 5)
                )
                # append estimator
                pipe = Pipeline(pipe_pre.steps + [("est", estimator)])
                score = _evaluate_pipeline_for_column(X_col, y, pipe, problem_type, cv=cv, scoring=scoring, n_jobs=n_jobs)
            else:
                # categorical strategy
                use_target = strat.get("use_target", False)
                enc = strat.get("encoder", "onehot")
                pipe_pre = _make_categorical_pipeline(imputer_strategy=strat.get("imputer", "most_frequent"), encoder=enc, use_target_encode=use_target)

                if use_target:
                    # need to fit encoder with y: we'll build a custom pipeline that applies the imputer
                    # then fit the target encoder separately inside cross_val (we can wrap in pipeline - scikit-learn will call fit with y)
                    pipe = Pipeline(pipe_pre.steps + [("est", estimator)])
                else:
                    pipe = Pipeline(pipe_pre.steps + [("est", estimator)])

                score = _evaluate_pipeline_for_column(X_col, y, pipe, problem_type, cv=cv, scoring=scoring, n_jobs=n_jobs)

            results[name] = score
        except Exception as e:
            warnings.warn(f"Strategy {name} failed for column {column}: {e}")
            results[name] = -np.inf

    # choose best initial strategy
    best_name = max(results, key=results.get)
    best_score = results[best_name]

    bo_trace = None
    # Optional: Bayesian optimisation over strategy choices
    if bo:
        if not SKOPT_AVAILABLE:
            raise RuntimeError("scikit-optimize (skopt) is required for Bayesian optimisation. Install via `pip install scikit-optimize`")
        # define search space -> list of strategy indices
        space = [Categorical([s["name"] for s in strategies], name="strategy")]
        # objective: negative score (minimise)
        def objective(x):
            strat_name = x[0]
            strat = next(s for s in strategies if s["name"] == strat_name)
            # reuse logic above to compute score for single strategy
            if pd.api.types.is_numeric_dtype(df[column]):
                pipe_pre = _make_numeric_pipeline(
                    imputer_strategy=strat.get("imputer", "mean"),
                    scaler=strat.get("scaler", None),
                    use_knn=strat.get("use_knn", False),
                    knn_neighbors=strat.get("knn_neighbors", 5)
                )
                pipe = Pipeline(pipe_pre.steps + [("est", estimator)])
            else:
                enc = strat.get("encoder", "onehot")
                use_target = strat.get("use_target", False)
                pipe_pre = _make_categorical_pipeline(imputer_strategy=strat.get("imputer", "most_frequent"), encoder=enc, use_target_encode=use_target)
                pipe = Pipeline(pipe_pre.steps + [("est", estimator)])
            score = _evaluate_pipeline_for_column(X_col, y, pipe, problem_type, cv=cv, scoring=scoring, n_jobs=1)
            # we minimize: negative of score
            return -score if np.isfinite(score) else 1e6

        # run gp_minimize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = gp_minimize(objective, space, n_calls=bo_iterations, random_state=42)
        bo_trace = res
        # update best based on BO
        chosen = res.x[0]
        best_name = chosen
        best_score = -res.fun if hasattr(res, "fun") else results.get(chosen, best_score)

    return {
        "best_strategy_name": best_name,
        "best_score": best_score,
        "all_results": results,
        "bo_trace": bo_trace
    }

if __name__ == "__main__":
    # A small example using sklearn's iris dataset
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    df = data.frame
    # artificially add a missingness and change a column to categorical
    df.loc[::10, "sepal length (cm)"] = np.nan
    df["species_cat"] = df["target"].astype(str)  # create a categorical column
    df["species_cat"].iloc[0] = np.nan

    # Use numeric column 'sepal width (cm)' as example
    result_numeric = benchmark_column(df, column="sepal width (cm)", target="target", problem_type="classification", cv=3, n_jobs=1, bo=False)
    print("Numeric column result:", result_numeric["best_strategy_name"], result_numeric["best_score"])

    # Use categorical column 'species_cat'
    result_cat = benchmark_column(df, column="species_cat", target="target", problem_type="classification", cv=3, n_jobs=1, bo=False)
    print("Categorical column result:", result_cat["best_strategy_name"], result_cat["best_score"])
