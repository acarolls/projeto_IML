# Metodologia de Processamento e Modelagem

## Visão geral

O projeto busca prever os níveis Saeb de proficiência em Língua Portuguesa (`NIVEL_LP`) e Matemática (`NIVEL_MT`) de estudantes do 3º/4º ano do Ensino Médio, usando microdados do Saeb 2023. O fluxo final está consolidado em `NotebookFinal.ipynb` e a parte reutilizável da modelagem está em `src/modelagem.py` e `src/modelos.py`.

O objetivo não é reproduzir diretamente a nota do estudante, mas avaliar quanto características pessoais, socioeconômicas e escolares conseguem distinguir faixas oficiais de desempenho sem vazamento de informação.

```text
Tabelas brutas -> filtros e seleção de features -> junção por escola/série
-> criação dos níveis Saeb -> análise exploratória -> imputação de ausentes
-> amostragem de modelagem -> pipeline de treinamento -> métricas e conclusão
```

## Preparação dos dados

O notebook lê as tabelas `TS_ALUNO_34EM.csv`, `TS_DIRETOR.csv` e `TS_ESCOLA.csv` a partir de `dados/`, usando separador `;` e codificação `latin-1`.

As principais etapas de preparação são:

- Remover alunos sem presença, sem proficiência válida ou sem preenchimento do questionário.
- Remover diretores sem questionário preenchido e registros de séries fora do escopo do Ensino Médio.
- Remover escolas sem alunos presentes no Ensino Médio.
- Selecionar apenas colunas ligadas ao contexto do aluno, direção e escola.
- Tratar respostas inválidas (`*`, `.`, `F` e vazios) como ausentes.
- Converter respostas booleanas e ordinais quando há interpretação natural de ordem.
- Unificar as tabelas usando `ID_ESCOLA` e a série quando aplicável.
- Criar indicador de ausência de diretor para preservar informação sobre escolas sem questionário de direção.
- Criar `NIVEL_LP` e `NIVEL_MT` a partir das proficiências brutas.

Após a análise exploratória, o notebook imputa ausências remanescentes na base final antes da modelagem. Médias escolares de proficiência, quando ausentes, são preenchidas por medianas agrupadas por UF ou por mediana global, conforme disponibilidade. Respostas de questionários são preenchidas por mediana ou moda de acordo com o tipo de variável.

## Alvos de classificação

As colunas `NIVEL_LP` e `NIVEL_MT` são derivadas das proficiências oficiais do Saeb, usando as faixas publicadas para a 3ª série do Ensino Médio:

| Disciplina | Nível 0 | Níveis intermediários | Último nível |
| --- | --- | --- | --- |
| Língua Portuguesa | pontuação `< 225` | níveis 1 a 7 em intervalos `[225, 250)`, ..., `[375, 400)` | nível 8 para `>= 400` |
| Matemática | pontuação `< 225` | níveis 1 a 9 em intervalos `[225, 250)`, ..., `[425, 450)` | nível 10 para `>= 450` |

Para a modelagem final, os registros com nível `0` em qualquer um dos dois alvos são removidos. Assim, os modelos ativos predizem níveis `1` a `8` para Língua Portuguesa e `1` a `10` para Matemática. Essa escolha evita que a classe de menor proficiência domine parte relevante da análise e mantém o foco nas faixas oficiais positivas de desempenho.

Os níveis possuem ordem natural, mas nesta versão são tratados como classes multiclasse nominais. Por isso, o F1-macro e as matrizes de confusão devem ser lidos com atenção: a métrica penaliza igualmente erros entre níveis vizinhos e erros entre níveis distantes.

## Amostragem de modelagem

Depois de remover os níveis `0`, o notebook usa uma amostra estratificada de 20% da base filtrada para treinar e avaliar os modelos. A estratificação é feita por `NIVEL_LP` com `random_state=42`.

Essa redução foi adotada por limitação prática de tempo e memória. O conjunto original é grande, e o treinamento de cinco algoritmos para dois alvos, com validação cruzada e busca de hiperparâmetros, torna a execução completa custosa em máquinas comuns.

## Prevenção de vazamento

Durante o treinamento, `src/modelagem.py` remove automaticamente colunas que não devem entrar como preditores:

- Alvos: `NIVEL_LP`, `NIVEL_MT`.
- Identificadores: `ID_ALUNO`, `ID_ESCOLA`, `ID_MUNICIPIO`.
- Proficiências e agregados diretamente ligados ao alvo: `PROFICIENCIA_LP_SAEB`, `PROFICIENCIA_MT_SAEB`, `MEDIA_EM_LP`, `MEDIA_EM_MT`, `MEDIA_EM_NIVEL_LP`, `MEDIA_EM_NIVEL_MT`.

`ID_ESCOLA` não entra como feature, mas é preservado temporariamente para formar os grupos da validação cruzada. Essa decisão evita que o modelo memorize escolas específicas e reduz vazamento entre treino e teste.

## Pipeline de pré-processamento

Cada modelo é treinado dentro de um pipeline único do scikit-learn. O pipeline separa variáveis numéricas e categóricas, aplicando:

- Imputação pela mediana e padronização (`StandardScaler`) para variáveis numéricas.
- Imputação pelo valor mais frequente e `OneHotEncoder(handle_unknown="ignore")` para variáveis categóricas.

Algumas colunas numéricas de identificação/categoria, como `ID_UF`, `ID_AREA`, `ID_SERIE` e `IN_PUBLICA`, são tratadas como categóricas para evitar impor uma relação ordinal artificial.

O pipeline também codifica internamente os rótulos do alvo com `LabelEncoder`, preservando os rótulos originais nas predições e em `classes_`.

## Modelos avaliados

Os cinco algoritmos avaliados são:

- Regressão Logística.
- KNN.
- Árvore de Decisão.
- Random Forest.
- XGBoost.

As fábricas de modelo estão em `src/modelos.py`. As configurações-base atuais são:

| Modelo | Configuração-base |
| --- | --- |
| KNN | `n_neighbors=5`, `weights="distance"` |
| Regressão Logística | `max_iter=1000`, `class_weight="balanced"` |
| Árvore de Decisão | `max_depth=20`, `random_state=42`, `class_weight="balanced"` |
| Random Forest | `n_estimators=300`, `random_state=42`, `n_jobs=1`, `class_weight="balanced_subsample"` |
| XGBoost | `n_estimators=200`, `max_depth=8`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `objective="multi:softprob"`, `tree_method="hist"`, `random_state=42`, `n_jobs=1` |

O wrapper de classificação também aplica `sample_weight` balanceado quando o estimador aceita pesos e não possui `class_weight` configurado, o que beneficia modelos como KNN/XGBoost no cenário de classes desbalanceadas.

## Validação e seleção de hiperparâmetros

A validação final usa `StratifiedGroupKFold`, com estratificação por classe e agrupamento por `ID_ESCOLA`.

O procedimento é:

1. Separar um fold externo para teste usando até `MAX_SPLITS_TESTE=7`.
2. Usar os folds restantes como treino/validação.
3. Rodar `GridSearchCV` com até `MAX_SPLITS_VALIDACAO=3` folds internos.
4. Selecionar hiperparâmetros por `f1_macro`.
5. Avaliar o melhor pipeline no fold externo de teste.
6. Ajustar um pipeline final com os melhores hiperparâmetros sobre toda a base de modelagem.

Os grids atuais são:

| Modelo | Hiperparâmetros avaliados |
| --- | --- |
| KNN | `modelo__n_neighbors`: 3, 5, 7 |
| Regressão Logística | `modelo__C`: 0.1, 1.0, 10.0 |
| Árvore de Decisão | `modelo__max_depth`: 5, 10, 20 |
| Random Forest | `modelo__n_estimators`: 100, 300 |
| XGBoost | `modelo__max_depth`: 4, 8; `modelo__learning_rate`: 0.05, 0.1 |

A execução é sequencial (`GRID_N_JOBS=1`) para reduzir consumo de memória.

## Checkpoint e artefatos

O notebook salva os resultados em `artefatos/resultados_modelagem.pkl` e o resumo tabular em `artefatos/resumo_modelos.csv`. Esses arquivos são locais e não devem ser versionados, pois podem ser grandes e dependem da execução local.

Se a execução for interrompida, `RESUMIR_EXECUCAO=True` permite retomar apenas combinações pendentes de alvo/modelo. Checkpoints corrompidos são movidos para um arquivo com sufixo `corrompido_<timestamp>`.

## Métricas e interpretação

Para cada combinação de modelo e alvo, são registradas:

- Acurácia e F1-macro nos folds de validação.
- Média e desvio-padrão das métricas de validação.
- Acurácia e F1-macro no conjunto de teste.
- Matriz de confusão no conjunto de teste.
- Métricas derivadas da matriz de confusão: precision, recall e F1 por classe, além de versões macro e ponderadas.

As matrizes de confusão são exibidas em duas formas:

- Contagens absolutas.
- Percentuais normalizados por classe real, facilitando a leitura do recall por classe.

Como as classes são desbalanceadas, a acurácia isolada pode favorecer modelos que predizem classes frequentes. O F1-macro, o recall por classe e as matrizes de confusão são mais adequados para avaliar se o modelo reconhece também níveis raros.

## Limitações

- As variáveis disponíveis descrevem contexto socioeconômico, familiar e escolar, mas não medem diretamente conhecimento do estudante.
- Os níveis Saeb são faixas discretas de uma proficiência contínua; erros entre níveis vizinhos são penalizados como qualquer outro erro de classe.
- O desbalanceamento é forte, especialmente em níveis altos de Matemática.
- A amostragem de 20% reduz custo computacional, mas pode limitar estabilidade para classes raras.
- A validação agrupada por escola é mais exigente e reduz a chance de superestimar desempenho por memorização de escolas.
- O modelo deve ser interpretado como análise agregada de padrões, não como ferramenta individual de decisão de alto impacto.

## Reprodutibilidade

Para reproduzir o fluxo:

1. Garantir que `dados/TS_ALUNO_34EM.csv`, `dados/TS_DIRETOR.csv` e `dados/TS_ESCOLA.csv` estejam disponíveis.
2. Instalar as dependências listadas em `Requirements.txt`.
3. Executar `NotebookFinal.ipynb` do início ao fim.
4. Verificar se os artefatos foram gerados em `artefatos/`.
5. Confirmar que nenhuma base bruta, checkpoint grande ou saída local foi adicionada ao Git.

## Referências

1. BRASIL. Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (Inep). [*Escalas de proficiência do Saeb*](http://download.inep.gov.br/publicacoes/institucionais/avaliacoes_e_exames_da_educacao_basica/escalas_de_proficiencia_do_saeb.pdf). Brasília, DF: Inep/MEC, 2020. Quadros 5 e 6, p. 29-38. Escalas específicas publicadas pelo Inep: [Língua Portuguesa - 3ª série do Ensino Médio](http://download.inep.gov.br/educacao_basica/prova_brasil_saeb/escala/escala_proficiencia/2018/LP_3EM.pdf) e [Matemática - 3ª série do Ensino Médio](http://download.inep.gov.br/educacao_basica/prova_brasil_saeb/escala/escala_proficiencia/2018/MT_3EM.pdf). Acesso em: 20 jun. 2026.
