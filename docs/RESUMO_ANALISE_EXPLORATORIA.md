# Resumo da Análise Exploratória - Trabalho IML

## O que foi feito


### 1. **Visão Geral dos Dados**
- 1.514.448 amostras (alunos) com 82 features
- 43 features numéricas, 39 categóricas, 7 identificadores
- Qualidade dos dados: 93,05% completo (6,95% de valores ausentes)

### 2. **Análise das Variáveis Alvo**

#### NIVEL_LP (Língua Portuguesa) - 9 níveis
```
Nível 0: 293.071 alunos (19,35%) - Maior classe
Nível 8:   1.200 alunos ( 0,08%) - Menor classe
→ Desbalanceamento: 244x
```

#### NIVEL_MT (Matemática) - 11 níveis
```
Nível 2: 306.529 alunos (20,24%) - Maior classe
Nível 10:     79 alunos ( 0,01%) - Menor classe
→ Desbalanceamento: 3.880x (severo!)
```

### 3. **Features Mais Importantes**

**Para NIVEL_LP (Língua Portuguesa):**
1. PROFICIENCIA_LP_SAEB (r=0,975) - Correlação esperada
2. PROFICIENCIA_MT_SAEB (r=0,572) - Desempenho correlacionado
3. MEDIA_EM_LP (r=0,401) - Média da escola
4. TX_Q035 (r=0,102) - Feature de questionário do diretor

**Para NIVEL_MT (Matemática):**
1. PROFICIENCIA_MT_SAEB (r=0,982) - Correlação esperada
2. PROFICIENCIA_LP_SAEB (r=0,561) - Desempenho correlacionado
3. MEDIA_EM_MT (r=0,419) - Média da escola
4. TX_Q035 (r=0,098) - Feature de questionário do diretor

### 4. **Dados Ausentes Principais**

| Feature | % Ausente | Observação |
|---------|-----------|------------|
| MEDIA_EM_LP | 21,89% | Alguns estados/escolas |
| MEDIA_EM_MT | 21,89% | Alguns estados/escolas |
| TX_Q022 | 19,17% | Questionário do diretor |
| TX_Q087 | 19,13% | Questionário do diretor |
| Outras TX_Q* | ~18-19% | Respostas opcionais |

---

## Principais Insights

1. **As variáveis alvo têm forte correlação com os scores SAEB**
   - Isso é esperado, pois foram derivadas deles
   - Confirma a integridade dos dados

2. **LP e MT estão moderadamente correlacionadas (r=0,564)**
   - Alunos bons em uma disciplina tendem a ser bons na outra
   - Mas não perfeitamente (há 'especialistas' em cada área)

3. **Características da escola importam, mas pouco**
   - MEDIA_EM_LP/MT têm correlação baixa (0,40-0,42)
   - Features do diretor são ainda mais fracas (0,06-0,10)
   - **Insight**: Características individuais (respostas do questionário) importam mais que características da escola

4. **Dados ausentes são principalmente nas features do diretor**
   - 17-23% das escolas não têm diretor responsável
   - Isso foi tratado no pré-processamento (POSSUI_DIRETOR = 0/1)

5. **Classes severamente desbalanceadas em NIVEL_MT**
   - Nível 2 tem 306 mil alunos
   - Nível 10 tem apenas 79 alunos
   - Isso afetará a modelagem → usar validação cruzada estratificada

---

## Recomendações para Modelagem

### O que fazer

1. **Validação Cruzada Estratificada**
   - Usar `StratifiedGroupKFold` para respeitar:
     - Distribuição das classes (estratificação)
     - Grupos de escola (para não vazar dados)

2. **Pré-processamento**
   - Imputação por **mediana** para numéricas
   - Imputação por **moda** para categóricas
   - Normalização/Padronização para distâncias (KNN) e gradientes (LogReg)

3. **One-Hot Encoding**
   - Converter 39 variáveis categóricas em variáveis binárias
   - Usar `OneHotEncoder` do sklearn
   - Ignorar novas categorias na predição

4. **Seleção de Features**
   - Remover as notas brutas de proficiência (PROFICIENCIA_LP_SAEB, etc)
   - Remover IDs (não são features preditivas)
   - Considerar remover features muito colineares

### Cuidado com

- **Multicolinearidade**: TX_Q032 e TX_Q033 estão correlacionadas
- **Desbalanceamento**: Usar `class_weight='balanced'` em modelos como LogReg e DecisionTree
- **Data leakage**: Não incluir MEDIA_EM_LP/MT na previsão de NIVEL_LP
- **Valores ausentes**: Pipeline deve tratar antes de cada modelo





## Próximas Etapas

1. **Pré-processamento** (`src/modelagem.py`)
   - Implementar `separar_dados()` - remover variáveis inadequadas
   - Implementar `criar_pipeline()` - encadear transformações
   
2. **Validação Cruzada**
   - Implementar `avaliar_modelo()` - estratificada e agrupada
   - Calcular métricas (Macro F1, Balanced Accuracy)

3. **Treinamento**
   - Testar os 5 modelos em `modelos.py`
   - Comparar desempenho em ambos os alvos

4. **Documentação Final**
   - Criar matriz de confusão para cada modelo
   - Gerar relatório com resultados e conclusões

---

## Estatísticas Rápidas

| Métrica | Valor | Status |
|---------|-------|--------|
| Amostras | 1.514.448 | ✓ Grande |
| Features | 82 | ✓ Muitas |
| Completude | 93,05% | ! Necessário imputar |
| Classes LP | 9 | ✓ Multiclasse |
| Classes MT | 11 | ✓ Multiclasse |
| Maior desequilíbrio | 3.880x | ! Severo (NIVEL_MT) |
| Correlação LP-MT | 0,564 | ✓ Moderada |

---

**Análise Concluída:** 21/06/2026  
