"""Orquestração do treinamento dos modelos de proficiência."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


# Alias de tipo para uma função sem argumentos que cria um estimador novo.
FabricaModelo = Callable[[], BaseEstimator]

# Definindo parâmetros de avaliação e teste
def obter_parametros(modelo):
    if isinstance(modelo, KNeighborsClassifier):
        return {
            "modelo__n_neighbors": [3, 5, 7]
        }
    if isinstance(modelo, LogisticRegression):
        return {
            "modelo__C": [0.1, 1.0, 10.0]
        }
    if isinstance(modelo, DecisionTreeClassifier):
        return {
            "modelo__max_depth": [5, 10, 20]
        }
    if isinstance(modelo, RandomForestClassifier):
        return {
            "modelo__n_estimators": [100, 300]
        }
    if isinstance(modelo, XGBClassifier):
        return {
            "modelo__max_depth": [4, 8],
            "modelo__learning_rate": [0.05, 0.1],
        }
    return {}



@dataclass(frozen=True)
class MetricasValidacao:
    """Métricas produzidas pela validação cruzada."""

    metricas_avaliacao: pd.DataFrame
    metricas_resumo: dict[str, float]
    matriz_confusao: np.ndarray


@dataclass(frozen=True)
class ResultadoTreinamento:
    """Pipeline ajustado e resultados de sua validação."""

    pipeline: Pipeline
    modelo: str
    alvo: str
    metricas_avaliacao: pd.DataFrame
    metricas_resumo: dict[str, float]
    matriz_confusao: np.ndarray


def treinar_modelo(
    tabela: pd.DataFrame,
    alvo: str,
    fabrica_modelo: FabricaModelo,
) -> ResultadoTreinamento:
    """Valida, avalia e ajusta um classificador sobre toda a tabela."""
    validar_entradas(tabela, alvo, fabrica_modelo)
    features, classes, escolas = separar_dados(tabela, alvo)
    metricas = avaliar_modelo(features, classes, escolas, fabrica_modelo)

    modelo = fabrica_modelo()
    pipeline = criar_pipeline(features, modelo)
    pipeline.fit(features, classes)

    return ResultadoTreinamento(
        pipeline=pipeline,
        modelo=type(modelo).__name__,
        alvo=alvo,
        metricas_avaliacao=metricas.metricas_avaliacao,
        metricas_resumo=metricas.metricas_resumo,
        matriz_confusao=metricas.matriz_confusao,
    )


def validar_entradas(
    tabela: pd.DataFrame,
    alvo: str,
    fabrica_modelo: FabricaModelo,
) -> None:
    """Valida a tabela, o alvo e a fábrica de estimadores."""
    raise NotImplementedError


def separar_dados(
    tabela: pd.DataFrame,
    alvo: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separa features, classes e grupos de escola sem vazamento."""
    raise NotImplementedError


def criar_pipeline(
    features: pd.DataFrame,
    modelo: BaseEstimator,
) -> Pipeline:
    # Criando uma implementação mínima:
    return Pipeline(
        [
            ("modelo", modelo),
        ]
    )


def avaliar_modelo(
    features: pd.DataFrame,
    classes: pd.Series,
    escolas: pd.Series,
    fabrica_modelo: FabricaModelo,
) -> MetricasValidacao:

    split_teste = StratifiedGroupKFold(
        n_splits=7,
        shuffle=True,
        random_state=42,
    )

    train_val_idx, test_idx = next(
        split_teste.split(
            features,
            classes,
            groups=escolas,
        )
    )

    X_train_val = features.iloc[train_val_idx]
    y_train_val = classes.iloc[train_val_idx]
    grupos_train_val = escolas.iloc[train_val_idx]

    X_test = features.iloc[test_idx]
    y_test = classes.iloc[test_idx]

    split_validacao = StratifiedGroupKFold(
        n_splits=6,
        shuffle=True,
        random_state=42,
    )

    train_idx_rel, val_idx_rel = next(
        split_validacao.split(
            X_train_val,
            y_train_val,
            groups=grupos_train_val,
        )
    )

    X_train = X_train_val.iloc[train_idx_rel]
    y_train = y_train_val.iloc[train_idx_rel]

    X_val = X_train_val.iloc[val_idx_rel]
    y_val = y_train_val.iloc[val_idx_rel]

    modelo = fabrica_modelo()

    pipeline = criar_pipeline(
        features,
        modelo,
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=obter_parametros(modelo),
        scoring="f1_macro",
        cv=3,
        n_jobs=4,
    )

    grid.fit(
        X_train,
        y_train,
    )

    pipeline = grid.best_estimator_

    y_pred_val = pipeline.predict(X_val)

    accuracy_val = accuracy_score(
        y_val,
        y_pred_val,
    )

    f1_val = f1_score(
        y_val,
        y_pred_val,
        average="macro",
    )

    X_train_final = pd.concat(
        [X_train, X_val]
    )

    y_train_final = pd.concat(
        [y_train, y_val]
    )

    pipeline.fit(
        X_train_final,
        y_train_final,
    )

    y_pred_test = pipeline.predict(X_test)

    accuracy_test = accuracy_score(
        y_test,
        y_pred_test,
    )

    f1_test = f1_score(
        y_test,
        y_pred_test,
        average="macro",
    )

    matriz = confusion_matrix(
        y_test,
        y_pred_test,
    )

    metricas_avaliacao = pd.DataFrame(
        [
            {
                "accuracy_validacao": accuracy_val,
                "f1_macro_validacao": f1_val,
                "accuracy_teste": accuracy_test,
                "f1_macro_teste": f1_test,
            }
        ]
    )

    metricas_resumo = {
        "accuracy_validacao": accuracy_val,
        "f1_macro_validacao": f1_val,
        "accuracy_teste": accuracy_test,
        "f1_macro_teste": f1_test,
    }

    return MetricasValidacao(
        metricas_avaliacao=metricas_avaliacao,
        metricas_resumo=metricas_resumo,
        matriz_confusao=matriz,
    )