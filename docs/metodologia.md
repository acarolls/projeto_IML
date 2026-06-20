# Metodologia de Processamento e Modelagem

## Visão geral

O projeto transforma os microdados do Saeb em uma tabela de estudantes elegíveis e treina classificadores para prever níveis de proficiência em Língua Portuguesa (LP) ou Matemática (MT). O fluxo terá duas interfaces principais: uma função de preparação que retorna um `pandas.DataFrame` e uma função de treinamento que recebe essa tabela, a coluna-alvo e uma fábrica de estimador.

## Preparação dos dados

1. Ler `TS_ALUNO_34EM.csv` e `TS_DIRETOR.csv` com separador `;` e codificação Latin-1.
2. Substituir `.` e `*` por valores ausentes e selecionar as colunas registradas em `Informações importantes.txt`.
3. Manter estudantes das séries de ensino médio analisadas que tenham presença e proficiência válida na disciplina correspondente ao alvo. Questionários incompletos permanecem na tabela; seus valores ausentes serão tratados pelo pipeline.
4. Filtrar os registros de direção aplicáveis e fazer uma junção à esquerda por `ID_ESCOLA`. A junção deve preservar todos os estudantes elegíveis e ser validada como muitos-para-um, impedindo a multiplicação acidental de linhas.
5. Retornar a tabela em memória. Arquivos brutos e tabelas geradas não devem ser versionados.

### Alvos de classificação

Serão criadas as colunas `NIVEL_PROFICIENCIA_LP` e `NIVEL_PROFICIENCIA_MT` a partir dos Quadros 5 e 6 de `Escalas de Proficiência do Saeb.pdf`, referentes à 3ª série do Ensino Médio:

| Disciplina | Nível 0 | Níveis intermediários | Último nível |
| --- | --- | --- | --- |
| LP | pontuação `< 225` | níveis 1 a 7 em intervalos `[225, 250)`, ..., `[375, 400)` | nível 8 para `>= 400` |
| MT | pontuação `< 225` | níveis 1 a 9 em intervalos `[225, 250)`, ..., `[425, 450)` | nível 10 para `>= 450` |

Os limites inferiores são inclusivos e os superiores, exclusivos. Ao treinar qualquer alvo, ambas as proficiências brutas, ambos os níveis derivados, identificadores de aluno e demais campos sem significado preditivo serão excluídos das entradas. Isso evita que o resultado da prova, ou uma transformação direta dele, revele a resposta ao modelo.

## Contrato de treinamento

```python
treinar_modelo(tabela, alvo, fabrica_modelo) -> ResultadoTreinamento
```

`fabrica_modelo` é uma função sem argumentos que devolve um estimador novo e compatível com a API do scikit-learn (`fit` e `predict`). Esse formato evita reutilizar estado ajustado entre folds. O projeto comparará fábricas para:

- `LogisticRegression`;
- `KNeighborsClassifier`;
- `DecisionTreeClassifier`;
- `RandomForestClassifier`;
- `XGBClassifier`.

Exemplos de fábricas válidas são `lambda: LogisticRegression(max_iter=1000, random_state=42)`, `lambda: KNeighborsClassifier()`, `lambda: DecisionTreeClassifier(random_state=42)`, `lambda: RandomForestClassifier(random_state=42)` e `lambda: XGBClassifier(random_state=42, eval_metric="mlogloss")`.

Os hiperparâmetros pertencem à fábrica; a função de treinamento não fará busca automática. O retorno `ResultadoTreinamento` conterá o pipeline final, nome do modelo, alvo, métricas de cada fold e média e desvio-padrão das métricas.

O pipeline fará todo o pré-processamento necessário. Variáveis numéricas terão imputação pela mediana e padronização; variáveis categóricas terão imputação pelo valor mais frequente e codificação one-hot. Como essas etapas ficam no mesmo objeto que o estimador, o pipeline treinado poderá receber novas linhas no mesmo formato da tabela processada e executar `predict` sem preparação manual paralela.

## Validação cruzada

Cada configuração será avaliada por validação cruzada estratificada e agrupada em cinco folds, usando `ID_ESCOLA` como grupo e `random_state=42`. A estratificação tenta manter proporções semelhantes dos níveis de proficiência em cada fold. O agrupamento mantém todos os estudantes de uma escola no mesmo fold; assim, atributos repetidos da escola ou da direção não aparecem simultaneamente no treino e na validação.

Em cada rodada, quatro folds treinam o pipeline completo e o fold restante mede o desempenho. Imputadores, codificadores e escaladores são ajustados somente nos quatro folds de treino, evitando vazamento de informação. Após as cinco rodadas, a mesma configuração é reajustada com todas as linhas e armazenada em `ResultadoTreinamento` para uso em previsões.

A métrica principal será **Macro F1**: calcula-se o F1 de cada nível separadamente e depois a média simples, dando a níveis raros a mesma importância dos frequentes. Serão registrados também balanced accuracy, que calcula a média da taxa de acerto por classe, e a matriz de confusão, que mostra quais níveis são confundidos entre si. A comparação dos cinco algoritmos usará primeiro o Macro F1 médio e considerará seu desvio entre folds.

## Reprodutibilidade e verificações

- Confirmar que a quantidade de estudantes não aumenta após a junção.
- Verificar ausência de sobreposição de `ID_ESCOLA` entre treino e validação.
- Confirmar que todas as classes esperadas estão representadas antes do treinamento; registrar quando um nível não existir nos dados filtrados.
- Executar o notebook do início ao fim após implementar o fluxo e registrar versões das dependências, inclusive `scikit-learn` e `xgboost`.
