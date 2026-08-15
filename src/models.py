"""
Machine Learning Pipeline definitions and factories for Room Occupancy Estimation.
"""
from typing import Dict, Any, Union
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from src import config


def get_logistic_regression_pipeline(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = config.RANDOM_STATE
) -> Pipeline:
    """
    Construct Logistic Regression pipeline with StandardScaler.
    Multiclass handling is automatically managed via multinomial/ovr in scikit-learn.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state
        ))
    ])


def get_knn_pipeline(
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "minkowski"
) -> Pipeline:
    """
    Construct K-Nearest Neighbors pipeline with StandardScaler.
    Distance-based model requiring standardized feature scaling.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        ))
    ])


def get_decision_tree_pipeline(
    max_depth: Union[int, None] = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    criterion: str = "gini",
    random_state: int = config.RANDOM_STATE
) -> Pipeline:
    """
    Construct Decision Tree pipeline (scaling is optional/omitted for tree algorithms).
    max_depth controls tree depth to mitigate overfitting.
    """
    return Pipeline([
        ("classifier", DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state
        ))
    ])


def get_model_zoo() -> Dict[str, Pipeline]:
    """
    Return dictionary of standard baseline candidate model pipelines.
    """
    return {
        "Logistic Regression": get_logistic_regression_pipeline(),
        "KNN (k=3)": get_knn_pipeline(n_neighbors=3),
        "KNN (k=5)": get_knn_pipeline(n_neighbors=5),
        "KNN (k=7)": get_knn_pipeline(n_neighbors=7),
        "Decision Tree (depth=3)": get_decision_tree_pipeline(max_depth=3),
        "Decision Tree (depth=5)": get_decision_tree_pipeline(max_depth=5),
        "Decision Tree (depth=7)": get_decision_tree_pipeline(max_depth=7),
        "Decision Tree (depth=10)": get_decision_tree_pipeline(max_depth=10)
    }


if __name__ == "__main__":
    zoo = get_model_zoo()
    print("Configured Model Pipelines:")
    for name, pipe in zoo.items():
        print(f"  - {name}: {pipe.named_steps}")
