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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

COLUNAS_ALVO = {
    "NIVEL_LP",
    "NIVEL_MT",
}

COLUNAS_IDENTIFICACAO = {
    "ID_ALUNO",
    "ID_ESCOLA",
    "ID_MUNICIPIO",
}

COLUNAS_VAZAMENTO = {
    "PROFICIENCIA_LP_SAEB",
    "PROFICIENCIA_MT_SAEB",
    "MEDIA_EM_LP",
    "MEDIA_EM_MT",
    "MEDIA_EM_NIVEL_LP",
    "MEDIA_EM_NIVEL_MT",
}

COLUNAS_CATEGORICAS_NUMERICAS = {
    "ID_UF",
    "ID_AREA",
    "ID_SERIE",
    "IN_PUBLICA",
}


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


def _quantidade_splits_possivel(
    classes: pd.Series,
    grupos: pd.Series,
    max_splits: int,
) -> int:
    """Define quantidade segura de folds para classes e grupos disponíveis."""
    menor_classe = int(classes.value_counts().min())
    total_grupos = int(grupos.nunique())
    return min(max_splits, menor_classe, total_grupos)



@dataclass(frozen=True)
class MetricasValidacao:
    """Métricas produzidas pela validação cruzada."""

    metricas_avaliacao: pd.DataFrame
    metricas_resumo: dict[str, float]
    matriz_confusao: np.ndarray
    melhores_parametros: dict[str, object]


@dataclass(frozen=True)
class ResultadoTreinamento:
    """Pipeline ajustado e resultados de sua validação."""

    pipeline: Pipeline
    modelo: str
    alvo: str
    metricas_avaliacao: pd.DataFrame
    metricas_resumo: dict[str, float]
    matriz_confusao: np.ndarray
    melhores_parametros: dict[str, object]


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
    pipeline.set_params(**metricas.melhores_parametros)
    pipeline.fit(features, classes)

    return ResultadoTreinamento(
        pipeline=pipeline,
        modelo=type(modelo).__name__,
        alvo=alvo,
        metricas_avaliacao=metricas.metricas_avaliacao,
        metricas_resumo=metricas.metricas_resumo,
        matriz_confusao=metricas.matriz_confusao,
        melhores_parametros=metricas.melhores_parametros,
    )


def validar_entradas(
    tabela: pd.DataFrame,
    alvo: str,
    fabrica_modelo: FabricaModelo,
) -> None:
    """Valida a tabela, o alvo e a fábrica de estimadores."""
    if not isinstance(tabela, pd.DataFrame):
        raise TypeError("tabela deve ser um pandas.DataFrame.")

    if tabela.empty:
        raise ValueError("tabela não pode estar vazia.")

    if not isinstance(alvo, str) or not alvo:
        raise TypeError("alvo deve ser uma string não vazia.")

    if alvo not in tabela.columns:
        raise ValueError(f"alvo '{alvo}' não existe na tabela.")

    if "ID_ESCOLA" not in tabela.columns:
        raise ValueError("tabela deve conter a coluna 'ID_ESCOLA' para agrupamento.")

    if tabela[alvo].isna().any():
        raise ValueError(f"alvo '{alvo}' contém valores ausentes.")

    if tabela["ID_ESCOLA"].isna().any():
        raise ValueError("coluna 'ID_ESCOLA' contém valores ausentes.")

    if tabela[alvo].nunique() < 2:
        raise ValueError(f"alvo '{alvo}' deve possuir pelo menos duas classes.")

    if tabela["ID_ESCOLA"].nunique() < 2:
        raise ValueError("tabela deve possuir pelo menos duas escolas distintas.")

    if _quantidade_splits_possivel(tabela[alvo], tabela["ID_ESCOLA"], 2) < 2:
        raise ValueError(
            "não há amostras/grupos suficientes para uma divisão estratificada."
        )

    if not callable(fabrica_modelo):
        raise TypeError("fabrica_modelo deve ser chamável.")

    modelo = fabrica_modelo()
    if not isinstance(modelo, BaseEstimator):
        raise TypeError("fabrica_modelo deve retornar um estimador do scikit-learn.")

    if not hasattr(modelo, "fit") or not hasattr(modelo, "predict"):
        raise TypeError("estimador retornado deve implementar fit e predict.")


def separar_dados(
    tabela: pd.DataFrame,
    alvo: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separa features, classes e grupos de escola sem vazamento."""
    classes = tabela[alvo].copy()
    escolas = tabela["ID_ESCOLA"].copy()

    colunas_remover = (
        COLUNAS_ALVO
        | COLUNAS_IDENTIFICACAO
        | COLUNAS_VAZAMENTO
    )
    colunas_remover = [
        coluna
        for coluna in colunas_remover
        if coluna in tabela.columns
    ]

    features = tabela.drop(columns=colunas_remover)

    if features.empty:
        raise ValueError(
            "nenhuma feature disponível após remover alvos e identificadores."
        )

    return features, classes, escolas


def criar_pipeline(
    features: pd.DataFrame,
    modelo: BaseEstimator,
) -> Pipeline:
    """Cria pipeline com pré-processamento e estimador."""
    colunas_categoricas = [
        coluna
        for coluna in features.columns
        if (
            coluna in COLUNAS_CATEGORICAS_NUMERICAS
            or pd.api.types.is_object_dtype(features[coluna])
            or pd.api.types.is_string_dtype(features[coluna])
            or isinstance(features[coluna].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(features[coluna])
        )
    ]

    colunas_numericas = [
        coluna
        for coluna in features.select_dtypes(include=[np.number]).columns
        if coluna not in colunas_categoricas
    ]

    transformadores = []

    if colunas_numericas:
        transformadores.append(
            (
                "numericas",
                Pipeline(
                    [
                        ("imputador", SimpleImputer(strategy="median")),
                        ("padronizador", StandardScaler()),
                    ]
                ),
                colunas_numericas,
            )
        )

    if colunas_categoricas:
        transformadores.append(
            (
                "categoricas",
                Pipeline(
                    [
                        ("imputador", SimpleImputer(strategy="most_frequent")),
                        (
                            "codificador",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                colunas_categoricas,
            )
        )

    if not transformadores:
        raise ValueError(
            "features deve conter pelo menos uma coluna numérica ou categórica."
        )

    preprocessador = ColumnTransformer(
        transformadores,
        remainder="drop",
    )

    return Pipeline(
        [
            ("preprocessamento", preprocessador),
            ("modelo", modelo),
        ]
    )


def avaliar_modelo(
    features: pd.DataFrame,
    classes: pd.Series,
    escolas: pd.Series,
    fabrica_modelo: FabricaModelo,
) -> MetricasValidacao:

    n_splits_teste = _quantidade_splits_possivel(
        classes,
        escolas,
        max_splits=7,
    )
    if n_splits_teste < 2:
        raise ValueError("dados insuficientes para separar conjunto de teste.")

    split_teste = StratifiedGroupKFold(
        n_splits=n_splits_teste,
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

    n_splits_validacao = _quantidade_splits_possivel(
        y_train_val,
        grupos_train_val,
        max_splits=6,
    )
    if n_splits_validacao < 2:
        raise ValueError("dados insuficientes para separar conjunto de validação.")

    split_validacao = StratifiedGroupKFold(
        n_splits=n_splits_validacao,
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
    grupos_train = grupos_train_val.iloc[train_idx_rel]

    X_val = X_train_val.iloc[val_idx_rel]
    y_val = y_train_val.iloc[val_idx_rel]

    modelo = fabrica_modelo()

    pipeline = criar_pipeline(
        features,
        modelo,
    )

    n_splits_grid = _quantidade_splits_possivel(
        y_train,
        grupos_train,
        max_splits=3,
    )
    if n_splits_grid < 2:
        raise ValueError("dados insuficientes para busca de hiperparâmetros.")

    split_grid = StratifiedGroupKFold(
        n_splits=n_splits_grid,
        shuffle=True,
        random_state=42,
    )

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=obter_parametros(modelo),
        scoring="f1_macro",
        cv=split_grid,
        n_jobs=4,
    )

    grid.fit(
        X_train,
        y_train,
        groups=grupos_train,
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
        zero_division=0,
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
        zero_division=0,
    )

    matriz = confusion_matrix(
        y_test,
        y_pred_test,
        labels=np.sort(classes.unique()),
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
        melhores_parametros=grid.best_params_,
    )
