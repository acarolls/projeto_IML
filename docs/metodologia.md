# Metodologia de Processamento e Modelagem

## Visão geral

O projeto transforma os microdados do Saeb em uma tabela de estudantes elegíveis e treina classificadores para prever níveis de proficiência em Língua Portuguesa (LP) ou Matemática (MT). O fluxo terá duas interfaces principais: uma função de preparação que retorna um `pandas.DataFrame` e uma função de treinamento que recebe essa tabela, a coluna-alvo e uma fábrica de estimador.

O objetivo não é reproduzir a nota exata do estudante, mas investigar quanto características pessoais, familiares e escolares conseguem distinguir faixas oficiais de desempenho. Por isso, o problema é tratado como classificação multiclasse. Os resultados devem ser interpretados como associações preditivas na população analisada, não como relações causais.

```text
microdados brutos -> seleção e limpeza -> junção por escola -> níveis Saeb
                  -> pipeline de atributos -> validação cruzada -> modelo final
```

## Preparação dos dados

1. Ler `TS_ALUNO_34EM.csv` e `TS_DIRETOR.csv` com separador `;` e codificação Latin-1.
2. Substituir `.` e `*` por valores ausentes e selecionar as colunas registradas em `Informações importantes.txt`.
3. Manter estudantes das séries de ensino médio analisadas que tenham presença e proficiência válida na disciplina correspondente ao alvo. Questionários incompletos permanecem na tabela; seus valores ausentes serão tratados pelo pipeline.
4. Filtrar os registros de direção aplicáveis e fazer uma junção à esquerda por `ID_ESCOLA`. A junção deve preservar todos os estudantes elegíveis e ser validada como muitos-para-um, impedindo a multiplicação acidental de linhas.
5. Retornar a tabela em memória. Arquivos brutos e tabelas geradas não devem ser versionados.

### Justificativa dos filtros e da junção

Presença e proficiência válida são requisitos porque não existe alvo observável para quem não realizou a prova. Não exigir questionário completo evita descartar sistematicamente estudantes com respostas ausentes e reduzir ainda mais a amostra. Essa decisão limita a população de referência: o modelo descreve estudantes participantes do Saeb e não deve ser generalizado automaticamente para ausentes.

A tabela de estudantes define a unidade de análise: uma linha representa um estudante. A junção à esquerda acrescenta contexto de direção sem eliminar estudantes cuja escola não tenha um registro de diretor utilizável. Antes da junção, registros de direção duplicados por `ID_ESCOLA` devem ser investigados e resolvidos por uma regra documentada; não se deve escolher uma linha arbitrariamente. Depois da junção, a quantidade de linhas e de `ID_ALUNO` distintos deve permanecer inalterada.

Os tipos serão definidos pelo significado das variáveis, não apenas pelo tipo lido pelo pandas. Códigos de respostas como `TX_RESP_*` são categóricos mesmo quando contêm números. Identificadores não são atributos preditivos. Apenas medidas quantitativas genuínas devem seguir o fluxo numérico.

### Alvos de classificação

Serão criadas as colunas `NIVEL_PROFICIENCIA_LP` e `NIVEL_PROFICIENCIA_MT` a partir dos Quadros 5 e 6 da publicação oficial *Escalas de proficiência do Saeb*, referentes à 3ª série do Ensino Médio [1]:

| Disciplina | Nível 0 | Níveis intermediários | Último nível |
| --- | --- | --- | --- |
| LP | pontuação `< 225` | níveis 1 a 7 em intervalos `[225, 250)`, ..., `[375, 400)` | nível 8 para `>= 400` |
| MT | pontuação `< 225` | níveis 1 a 9 em intervalos `[225, 250)`, ..., `[425, 450)` | nível 10 para `>= 450` |

Os limites inferiores são inclusivos e os superiores, exclusivos. Ao treinar qualquer alvo, ambas as proficiências brutas, ambos os níveis derivados, identificadores de aluno e demais campos sem significado preditivo serão excluídos das entradas. Isso evita que o resultado da prova, ou uma transformação direta dele, revele a resposta ao modelo.

Usar as faixas oficiais mantém o significado pedagógico das classes e permite relacionar os resultados à escala publicada pelo Inep. Faixas criadas por quantis produziriam classes mais equilibradas, mas seus limites mudariam com a amostra e perderiam essa interpretação. A remoção das notas de LP e MT é deliberadamente conservadora: embora a nota da outra disciplina possa elevar a precisão, ela funciona como uma medida paralela do mesmo desempenho escolar e desviaria o estudo para uma previsão baseada em outra prova.

Os níveis possuem ordem natural, mas nesta primeira abordagem serão tratados como classes distintas pelos cinco algoritmos. Macro F1 não diferencia um erro entre níveis vizinhos de um erro entre níveis distantes; por isso, a matriz de confusão deve ser lida junto com a métrica. Uma extensão futura poderá acrescentar métricas ordinais, como kappa ponderado, sem alterar os alvos oficiais.

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

Uma fábrica, em vez de um estimador já criado, garante uma instância sem estado para cada fold e para o ajuste final. Isso impede que parâmetros aprendidos em uma rodada sejam reutilizados na seguinte. O contrato previsto para o resultado é:

| Campo | Conteúdo |
| --- | --- |
| `pipeline` | pré-processador e estimador reajustados em todos os dados |
| `modelo` | nome estável da configuração avaliada |
| `alvo` | coluna de nível prevista |
| `metricas_folds` | Macro F1 e balanced accuracy de cada fold |
| `metricas_resumo` | média e desvio-padrão das métricas |
| `matriz_confusao` | matriz agregada das previsões fora da amostra |

O pipeline fará todo o pré-processamento necessário. Variáveis numéricas terão imputação pela mediana e padronização; variáveis categóricas terão imputação pelo valor mais frequente e codificação one-hot. Como essas etapas ficam no mesmo objeto que o estimador, o pipeline treinado poderá receber novas linhas no mesmo formato da tabela processada e executar `predict` sem preparação manual paralela.

### Justificativa do pré-processamento

A mediana é menos sensível a valores extremos que a média. A categoria mais frequente fornece um valor válido para respostas ausentes sem criar códigos numéricos artificiais. One-hot encoding evita impor uma ordem inexistente às alternativas dos questionários e deve ignorar categorias novas durante a predição, mantendo o esquema aprendido no treino.

A padronização é essencial para KNN, que usa distâncias, e melhora a otimização da regressão logística quando as escalas numéricas diferem. Árvores e florestas não dependem da escala, mas podem compartilhar a mesma transformação sem mudar a ordem dos valores. Todo pré-processamento deve ser ajustado dentro de cada fold: calcular medianas, categorias ou escalas antes da divisão permitiria que a validação influenciasse o treino.

Os modelos representam hipóteses diferentes. A regressão logística oferece uma referência linear e interpretável; KNN explora similaridade local; a árvore captura regras e interações não lineares; random forest reduz a variância de uma árvore por agregação; XGBoost constrói árvores sequencialmente para corrigir erros anteriores. Compará-los sob os mesmos atributos, folds e métricas separa melhor o efeito do algoritmo do efeito da avaliação.

## Validação cruzada

Cada configuração será avaliada por validação cruzada estratificada e agrupada em cinco folds, usando `ID_ESCOLA` como grupo e `random_state=42`. A estratificação tenta manter proporções semelhantes dos níveis de proficiência em cada fold. O agrupamento mantém todos os estudantes de uma escola no mesmo fold; assim, atributos repetidos da escola ou da direção não aparecem simultaneamente no treino e na validação.

Cinco folds oferecem um compromisso entre custo e estabilidade: cada ajuste usa aproximadamente 80% das escolas para treino e 20% para validação, repetindo o processo cinco vezes. Dez folds exigiriam aproximadamente o dobro de ajustes e deixariam conjuntos de validação menores. Uma única separação treino/teste seria mais barata, porém muito mais dependente da escolha aleatória das escolas.

Em cada rodada, quatro folds treinam o pipeline completo e o fold restante mede o desempenho. Imputadores, codificadores e escaladores são ajustados somente nos quatro folds de treino, evitando vazamento de informação. Após as cinco rodadas, a mesma configuração é reajustada com todas as linhas e armazenada em `ResultadoTreinamento` para uso em previsões.

A métrica principal será **Macro F1**: calcula-se o F1 de cada nível separadamente e depois a média simples, dando a níveis raros a mesma importância dos frequentes. Serão registrados também balanced accuracy, que calcula a média da taxa de acerto por classe, e a matriz de confusão, que mostra quais níveis são confundidos entre si. A comparação dos cinco algoritmos usará primeiro o Macro F1 médio e considerará seu desvio entre folds.

Para cada classe, precisão mede quantas previsões daquele nível estavam corretas e recall mede quantos estudantes daquele nível foram encontrados. O F1 combina as duas medidas e só é alto quando ambas são altas. A média macro impede que uma classe numerosa domine o resultado. Balanced accuracy complementa essa visão ao resumir apenas o recall médio, enquanto a matriz de confusão preserva os tipos de erro que os valores agregados escondem.

Os folds devem ser gerados uma única vez e reutilizados para todos os modelos. Cada classe precisa ocorrer em pelo menos cinco escolas distintas; caso contrário, a validação em cinco grupos não é válida e a execução deve falhar com uma mensagem clara, sem migrar silenciosamente para uma divisão por estudante. A matriz de confusão agregada deve usar somente previsões feitas quando cada observação estava fora do treino.

Não haverá ajuste automático de hiperparâmetros nesta etapa. Escolher hiperparâmetros com base nos mesmos cinco folds e depois apresentar suas métricas como estimativa final produziria uma avaliação otimista. Se busca de hiperparâmetros for adicionada, ela deverá usar validação aninhada ou um conjunto de teste externo mantido intocado.

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
