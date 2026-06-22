# Metodologia de Processamento e Modelagem

## Visão geral

O projeto tem por objetivo prever os níveis Saeb de proficiência em Lingua Portuguesa (LP) ou Matemática (MT) de alunos do 3º e 4º Anos do Ensino Médio. Para isso, utilizaremos classificadores de Aprendizado de Máquina em cima dos microdados do Saeb de 2023 e a biblioteca Pandas do Python.

O objetivo não é reproduzir a nota do estudante, mas investigar quanto características pessoais, familiares e escolares conseguem distinguir faixas oficiais de desempenho.

```text
Tabelas brutas "Aluno", "Diretor" e "Escola" -> Limpeza e adequação de features -> Junção das tabelas -> Adicionamos o nível Saeb -> Análise Exploratória -> Pipeline de atributos -> validação cruzada -> Modelos -> Métricas
```

## Preparação dos dados

1. Ler `TS_ALUNO_34EM.csv`, `TS_DIRETOR.csv` e `TS_ESCOLA.csv`.
2. Removemos dados incompletos das tabelas, utilizando atributos de controle, como `IN_PRESENCA_LP` e `IN_PREENCHIMENTO_QUESTIONARIO`.
3. Removemos dados que não são relevantes para nossa análise, como diretores de outras séries que não sejam do Ensino Médio, ou escolas que não possuem Ensino Médio.
4. Verificamos se temos alunos, diretores ou escolas duplicadas.
5. Realizamos adaptações nas features de diretor de categórica para numérica.
6. Realizamos a junção das 3 tabelas de dados.
7. Adicionamos marcador de ausência de diretor e trocamos a nota de proficiência pelo nível Saeb.

### Justificativa dos filtros e da junção

A justificativa para esse processo é enriquecer a tabela de alunos com dados que podem ser impactantes no desempenho de estudantes como informações socioeconomicas, estruturais da escola, infraestrutura entre outros.

Escolhemos um conjunto de features do dataset tendo em vista sua descrição no dicionário de dados. Nesse momento de escolha das features, ainda não teremos certeza sobre sua relevância global, no entanto inferimos que poderiam fazer parte do nosso escopo.

A maioria das features apresenta o formato de multipla escolha, apontando grau de concordância ou adequação. Dessa forma é possível inferir que esses dados possuem intenção de ordenamento. Por isso, podemos tratálos de forma numérica.

### Alvos de classificação

Serão removidas as colunas numéricas de `PROFICIENCIA_LP_SAEB`, `PROFICIENCIA_MT_SAEB`, que darão lugar a criação das colunas `NIVEL_LP` e `NIVEL_MT` a partir dos Quadros 5 e 6 da publicação oficial *Escalas de proficiência do Saeb*, referentes à 3ª série do Ensino Médio [1]:

| Disciplina | Nível 0 | Níveis intermediários | Último nível |
| --- | --- | --- | --- |
| LP | pontuação `< 225` | níveis 1 a 7 em intervalos `[225, 250)`, ..., `[375, 400)` | nível 8 para `>= 400` |
| MT | pontuação `< 225` | níveis 1 a 9 em intervalos `[225, 250)`, ..., `[425, 450)` | nível 10 para `>= 450` |

Os limites inferiores são inclusivos e os superiores, exclusivos. Ao treinar qualquer alvo, ambas as proficiências brutas, ambos os níveis derivados, identificadores de aluno e demais campos sem significado preditivo serão excluídos das entradas. Isso evita que o resultado da prova, ou uma transformação direta dele, revele a resposta ao modelo.

Usar as faixas oficiais mantém o significado pedagógico das classes e permite relacionar os resultados à escala publicada pelo Inep. Faixas criadas por quantis produziriam classes mais equilibradas, mas seus limites mudariam com a amostra e perderiam essa interpretação. A remoção das notas de LP e MT é deliberadamente conservadora: embora a nota da outra disciplina possa elevar a precisão, ela funciona como uma medida paralela do mesmo desempenho escolar e desviaria o estudo para uma previsão baseada em outra prova.

Os níveis possuem ordem natural, mas nesta primeira abordagem serão tratados como classes distintas pelos cinco algoritmos. Macro F1 não diferencia um erro entre níveis vizinhos de um erro entre níveis distantes; por isso, a matriz de confusão deve ser lida junto com a métrica.

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

Embora os exemplos acima apresentem apenas algumas configurações possíveis, os hiperparâmetros utilizados foram definidos manualmente com o objetivo de manter o custo computacional dos experimentos em níveis compatíveis com máquinas de uso comum, sem a necessidade de hardware especializado. Dessa forma, buscamos um equilíbrio entre tempo de execução e capacidade de modelagem, priorizando configurações que permitissem a execução dos testes de forma estável e acessível. 

Para o `KNeighborsClassifier` foi utilizado `n_neighbors=5`, mantendo uma configuração simples e amplamente utilizada como referência. 

O `LogisticRegression` utilizou `max_iter=1000` para garantir a convergência do treinamento. 

O `DecisionTreeClassifier` teve sua profundidade limitada por `max_depth=20`, para que o crescimento da estrutura e o consumo de recursos computacionais não se tornasse excessivo. 

O `RandomForestClassifier` foi configurado com 300 árvores (`n_estimators=300`), buscando limitar o tempo de treinamento sem comprometer significativamente a estabilidade das previsões. 

Já o `XGBClassifier` utilizou 500 estimadores (`n_estimators=500`), profundidade máxima de 8 níveis (`max_depth=8`) e taxa de aprendizado de 0,05 (`learning_rate=0.05`), adotando valores intermediários que permitissem a execução dos experimentos em tempo viável. 

Os hiperparâmetros iniciais foram definidos nas fábricas, onde servem como ponto de partida para cada modelo. Durante o treinamento, é aplicada uma busca em grade (Grid Search) sobre um conjunto reduzido de combinações predefinidas para cada abordagem. Desta forma, podemos ajustar os principais hiperparâmetros paranão ultrapassar o custo computacional dos experimentos. 

O retorno `ResultadoTreinamento` conterá o pipeline final, nome do modelo, alvo, métricas de cada fold e média e desvio-padrão das métricas.

Uma fábrica, em vez de um estimador já criado, garante uma instância sem estado para cada fold e para o ajuste final. Isso impede que parâmetros aprendidos em uma rodada sejam reutilizados na seguinte. O contrato previsto para o resultado é:

| Campo | Conteúdo |
| --- | --- |
| `pipeline` | pipeline treinado com o melhor conjunto de hiperparâmetros |
| `modelo` | nome do algoritmo avaliado |
| `alvo` | coluna de proficiência prevista |
| `metricas_avaliacao` | métricas registradas durante as etapas de validação e teste |
| `metricas_resumo` | resumo das métricas utilizadas na comparação dos modelos |
| `matriz_confusao` | matriz de confusão produzida sobre o conjunto de teste |

O pipeline fará todo o pré-processamento necessário. Variáveis numéricas terão imputação pela mediana e padronização; variáveis categóricas terão imputação pelo valor mais frequente e codificação one-hot. Como essas etapas ficam no mesmo objeto que o estimador, o pipeline treinado poderá receber novas linhas no mesmo formato da tabela processada e executar `predict` sem preparação manual paralela.

### Escolha dos modelos

Esses modelos supervisionados foram escolhidos para capturar diferentes formas de relação entre variáveis tabulares estruturadas do Saeb, explorando tanto as relações lineares quanto os padrões não lineares, além de permitir compreender as interações entre atributos.

A regressão logística foi utilizada como base por possuir uma relação aproximadamente linear entre as features e as classes de proficiência, servindo como referência interpretável para comparação com os modelos mais complexos.

O K-Nearest Neighbors foi incluído pois depende diretamente de medidas de distância no espaço de atributos, o que permite avaliar se estudantes com características semelhantes tendem a compartilhar níveis de desempenho próximos dentro do conjunto de dados.

A árvore de decisão foi escolhida por particionar as features em regras hierárquicas, construindo condições específicas a partir das combinações de resultados dos questionários e variáveis escolares.

O Random Forest é utilizado por ser uma extensão baseada em ensembles de árvores independentes, reduzindo a variância e melhorando a estabilidade das previsões de diversos subconjuntos de dados. 

O XGBoost foi incluído por ser um método amplamente utilizado em artigos científicos cujo objetivo é resolver problemas de dados tabulares estruturados, principalmente em contextos educacionais e de classificação multiclasse, pois ele é capaz de construir modelos sequenciais que corrigem erros das iterações anteriores.


Todos os modelos seguem o mesmo pipeline de pré-processamento e exatamente as mesmas divisões de validação, o que possibilita e facilita a comparação entre as abordagens selecionadas. A diferença de desempenho observada entre eles é definida exclusivamente pela capacidade de modelagem de cada algoritmo.


É importante ressaltar que a escolha desses modelos tem como propósito comparar diferentes abordagens de aprendizado sob a mesma estrutura de dados e avaliação. Isso permite analisar até que ponto as relações mais simples são suficientes ou se ganhos significativos são obtidos ao introduzir não linearidade e métodos baseados em ensembles.

### Justificativa do pré-processamento

A mediana é menos sensível a valores extremos que a média. A categoria mais frequente fornece um valor válido para respostas ausentes sem criar códigos numéricos artificiais. One-hot encoding evita impor uma ordem inexistente às alternativas dos questionários e deve ignorar categorias novas durante a predição, mantendo o esquema aprendido no treino.

A padronização é essencial para KNN, que usa distâncias, e melhora a otimização da regressão logística quando as escalas numéricas diferem. Árvores e florestas não dependem da escala, mas podem compartilhar a mesma transformação sem mudar a ordem dos valores. Todo pré-processamento deve ser ajustado dentro de cada fold: calcular medianas, categorias ou escalas antes da divisão permitiria que a validação influenciasse o treino.

## Validação e seleção de modelos

Os dados foram divididos em três subconjuntos independentes:

- Treinamento (~70%)
- Validação (~15%)
- Teste (~15%)

A divisão foi realizada de forma estratificada, buscando preservar a distribuição dos níveis de proficiência em cada subconjunto. Dessa forma, evitamos que uma classe fique super ou sub-representada em alguma etapa do processo de treinamento e avaliação.

O conjunto de treinamento é utilizado para ajuste dos modelos e busca de hiperparâmetros. O conjunto de validação é utilizado para comparar diferentes configurações e selecionar aquelas que apresentam melhor capacidade de generalização. Já o conjunto de teste permanece isolado durante todo o processo e é utilizado apenas na avaliação final dos modelos.

Essa separação é importante porque um modelo pode apresentar excelente desempenho nos dados utilizados durante o treinamento e ainda assim falhar ao analisar novos estudantes. A utilização de um conjunto de validação permite verificar se os padrões aprendidos são realmente generalizáveis e não apenas resultado de sobreajuste aos dados de treinamento.

Após a escolha da melhor configuração, o modelo é reajustado utilizando os dados de treinamento e validação. A avaliação final é então realizada sobre o conjunto de teste, fornecendo uma estimativa mais confiável do desempenho esperado em novos dados.

### Busca de hiperparâmetros

Embora cada algoritmo possua uma configuração inicial definida manualmente, o desempenho dos modelos pode variar significativamente de acordo com a escolha de seus hiperparâmetros. Por esse motivo, foi utilizada uma etapa de busca automática baseada em Grid Search (`GridSearchCV`).

O Grid Search avalia diferentes combinações de hiperparâmetros previamente definidas para cada algoritmo e identifica aquela que produz o melhor resultado segundo uma métrica de desempenho escolhida. Neste trabalho, a métrica utilizada para seleção foi o F1-Score Macro.

A escolha do F1-Score Macro ocorreu porque os níveis de proficiência não apresentam distribuição uniforme. Em cenários desbalanceados, a acurácia pode produzir interpretações excessivamente otimistas ao favorecer classes mais frequentes. O F1-Score Macro atribui o mesmo peso a todas as classes, permitindo uma avaliação mais equilibrada do desempenho dos modelos.

A busca foi realizada utilizando validação cruzada interna com três partições (`cv=3`). Para cada combinação de hiperparâmetros, o conjunto de treinamento é dividido em três subconjuntos, permitindo avaliar a estabilidade da configuração antes de aplicá-la aos dados de validação.

Os principais hiperparâmetros avaliados foram:

- KNN: quantidade de vizinhos (`n_neighbors`);
- Regressão Logística: fator de regularização (`C`);
- Árvore de Decisão: profundidade máxima (`max_depth`);
- Random Forest: número de árvores (`n_estimators`);
- XGBoost: profundidade máxima (`max_depth`) e taxa de - aprendizado (`learning_rate`).

Ao final desse processo, a configuração com melhor F1-Score Macro é selecionada e utilizada nas etapas posteriores de avaliação.

### Experimentos exploratórios

Além da validação utilizada como método principal de avaliação dos modelos, foi criado um notebook de testes (`teste_modelos.ipynb`) com o objetivo de verificar o funcionamento do pipeline e observar o comportamento dos modelos em uma execução direta.

Nesse experimento, cada modelo foi treinado e avaliado de forma isolada, permitindo a análise de métricas como precision, recall e F1-score, além do tempo de execução do treinamento e da predição. Também foram observados os parâmetros utilizados na configuração do modelo e do pipeline.

Esses testes foram feitos com dados sintéticos, composto por 50.000 observações, 50 atributos, dos quais 20 foram definidos como atributos informativos para o processo de classificação, distribuídas em três classes. A divisão entre treino e teste foi realizada na proporção de 80% e 20%, utilizando estratificação das classes e `random_state=42` para garantir reprodutibilidade. 

Os resultados obtidos revelaram diferenças significativas entre os modelos tanto em desempenho quanto em custo computacional. Métodos baseados em ensembles, como Random Forest e XGBoost, apresentaram melhor desempenho no conjunto sintético, porém com maior tempo de treinamento quando comparados a modelos mais simples.

Vale ressaltar que o propósito desta avaliação não é medir o desempenho real dos modelos no problema estudado, mas sim analisar o comportamento e o custo computacional das abordagens em um ambiente controlado. Portanto, esses testes não substituem a validação principal, porém servem como uma verificação complementar.

## Interpretação e limitações

- Desempenho preditivo não demonstra que uma variável causa maior ou menor proficiência.
- Ausência de respostas pode carregar informação social; a imputação melhora a compatibilidade dos modelos, mas pode ocultar esse padrão e deve ser quantificada.
- Níveis extremos podem ter poucos estudantes ou poucas escolas, elevando a incerteza do Macro F1. Média e desvio-padrão devem sempre ser apresentados juntos.
- As respostas são autorrelatadas e estão sujeitas a erro de preenchimento.
- O modelo é específico ao ano, etapa e população filtrada. Aplicação em outro Saeb exige verificar alterações no questionário, nas escalas e na distribuição dos dados.
- Atributos sensíveis podem reproduzir desigualdades existentes. As análises devem ser agregadas e não usadas para decisões individuais de alto impacto.

## Reprodutibilidade e verificações

- Confirmar que a quantidade de estudantes não aumenta após a junção.
- Registrar quantidade e proporção de linhas removidas em cada filtro.
- Comparar a distribuição dos níveis antes e depois da junção com os dados de direção.
- Verificar ausência de sobreposição de `ID_ESCOLA` entre treino e validação.
- Confirmar que todas as classes esperadas estão representadas antes do treinamento; registrar quando um nível não existir nos dados filtrados.
- Reutilizar exatamente os mesmos folds na comparação dos cinco algoritmos.
- Testar a predição com categorias não observadas no treino e valores ausentes.
- Executar o notebook do início ao fim após implementar o fluxo e registrar versões das dependências, inclusive `scikit-learn` e `xgboost`.

## Referências

1. BRASIL. Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (Inep). [*Escalas de proficiência do Saeb*](http://download.inep.gov.br/publicacoes/institucionais/avaliacoes_e_exames_da_educacao_basica/escalas_de_proficiencia_do_saeb.pdf). Brasília, DF: Inep/MEC, 2020. Quadros 5 e 6, p. 29–38. Escalas específicas publicadas pelo Inep: [Língua Portuguesa — 3ª série do Ensino Médio](http://download.inep.gov.br/educacao_basica/prova_brasil_saeb/escala/escala_proficiencia/2018/LP_3EM.pdf) e [Matemática — 3ª série do Ensino Médio](http://download.inep.gov.br/educacao_basica/prova_brasil_saeb/escala/escala_proficiencia/2018/MT_3EM.pdf). Acesso em: 20 jun. 2026.
