"""Orquestração do treinamento dos modelos de proficiência."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


# Alias de tipo para uma função sem argumentos que cria um estimador novo.
FabricaModelo = Callable[[], BaseEstimator]


@dataclass(frozen=True)
class MetricasValidacao:
    """Métricas produzidas pela validação cruzada."""

    metricas_folds: pd.DataFrame
    metricas_resumo: dict[str, float]
    matriz_confusao: np.ndarray


@dataclass(frozen=True)
class ResultadoTreinamento:
    """Pipeline ajustado e resultados de sua validação."""

    pipeline: Pipeline
    modelo: str
    alvo: str
    metricas_folds: pd.DataFrame
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
        metricas_folds=metricas.metricas_folds,
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
    """Cria o pipeline de pré-processamento e classificação."""
    raise NotImplementedError


def avaliar_modelo(
    features: pd.DataFrame,
    classes: pd.Series,
    escolas: pd.Series,
    fabrica_modelo: FabricaModelo,
) -> MetricasValidacao:
    """Executa a validação cruzada estratificada e agrupada."""
    raise NotImplementedError
