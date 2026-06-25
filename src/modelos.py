from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def criar_knn():
    return KNeighborsClassifier(
        n_neighbors=5
    )


def criar_regressao_logistica():
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

def criar_arvore_decisao():
    return DecisionTreeClassifier(
        max_depth=20,
        random_state=42,
        class_weight="balanced",
    )

def criar_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=4,
        class_weight="balanced_subsample",
    )


def criar_xgboost():
    return XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=4
    )
