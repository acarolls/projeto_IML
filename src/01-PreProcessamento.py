# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd

df_aluno_original = pd.read_csv('dados/TS_ALUNO_34EM.csv', encoding='latin-1', sep=';')
df_diretor_original = pd.read_csv('dados/TS_DIRETOR.csv', encoding='latin-1', sep=';')
df_escola_original = pd.read_csv('dados/TS_ESCOLA.csv', encoding='latin-1', sep=';')

# ================================================================
# Escolhemos a features que temos interesses e as features
# que usaremos para limpar linhas que não tenham dados relevantes
# ================================================================

colunas_de_interesse_aluno = ['ID_ESCOLA', 'ID_ALUNO', 'ID_UF', 'ID_AREA', 'IN_PUBLICA', 'ID_SERIE', 'PROFICIENCIA_LP_SAEB', 
'PROFICIENCIA_MT_SAEB', 'TX_RESP_Q01', 'TX_RESP_Q02', 'TX_RESP_Q03', 'TX_RESP_Q04', 'TX_RESP_Q05a', 'TX_RESP_Q05b',
'TX_RESP_Q05c', 'TX_RESP_Q06','TX_RESP_Q07a','TX_RESP_Q07b','TX_RESP_Q07c','TX_RESP_Q07d','TX_RESP_Q07e','TX_RESP_Q08','TX_RESP_Q09',
'TX_RESP_Q10a','TX_RESP_Q10b','TX_RESP_Q10c','TX_RESP_Q10d','TX_RESP_Q10e','TX_RESP_Q10f','TX_RESP_Q11a','TX_RESP_Q11b','TX_RESP_Q11c',
'TX_RESP_Q12b','TX_RESP_Q12c','TX_RESP_Q14','TX_RESP_Q15b','TX_RESP_Q16','TX_RESP_Q17','TX_RESP_Q18','TX_RESP_Q19','TX_RESP_Q21a',
'TX_RESP_Q21b','TX_RESP_Q21c','TX_RESP_Q21d','TX_RESP_Q21e','TX_RESP_Q23d','TX_RESP_Q24']

remover_aluno = {
    'IN_PRESENCA_LP': 0,                    # Remover alunos que não tenham respondido a prova de língua portuguesa
    'IN_PRESENCA_MT': 0,                    # Remover alunos que não tenham respondido a prova de matemática
    'IN_PROFICIENCIA_LP': 0,                # Remover alunos que não tenham proficiência em língua portuguesa
    'IN_PROFICIENCIA_MT': 0,                # Remover alunos que não tenham proficiência em matemática
    'IN_PREENCHIMENTO_QUESTIONARIO': 0}     # Remover alunos que não preencheram o questionário

colunas_de_interesse_diretor = ['ID_ESCOLA', 'ID_SERIE', 'TX_Q020','TX_Q022','TX_Q032','TX_Q033','TX_Q035',
'TX_Q036','TX_Q056','TX_Q057','TX_Q078','TX_Q079','TX_Q081','TX_Q082','TX_Q083', 'TX_Q085','TX_Q087',
'TX_Q108','TX_Q119','TX_Q129','TX_Q130','TX_Q139','TX_Q191','TX_Q194','TX_Q203','TX_Q205','TX_Q206','TX_Q207',
'TX_Q208','TX_Q209']

remover_diretor = {
    'IN_PREENCHIMENTO_QUESTIONARIO': [0],   # Remover diretores que não preencheram o questionário
    'ID_SERIE': [5, 9, 2]}

colunas_de_interesse_escola = ['ID_ESCOLA','PC_FORMACAO_DOCENTE_MEDIO',
                               'TAXA_PARTICIPACAO_EM','MEDIA_EM_LP','MEDIA_EM_MT']

remover_escola = {
    'NU_PRESENTES_EM': 0}                   # Remover escolas que não tenham alunos presentes no ensino médio

# ================================================================
# Aplicaremos as regras de limpeza de dados e selecionaremos as features
# ================================================================

print("\n# ================================================================\n")
df_aluno_limpo = df_aluno_original.copy()
df_diretor_limpo = df_diretor_original.copy()
df_escola_limpo = df_escola_original.copy()

for coluna, valor in remover_aluno.items():
    antes = len(df_aluno_limpo)
    df_aluno_limpo = df_aluno_limpo[df_aluno_limpo[coluna] != valor]
    removidas = antes - len(df_aluno_limpo)
    
    print(
        f"Alunos: removidas {removidas} linhas "
        f"pela regra: {coluna} != {valor}"
    )

for coluna, valor in remover_diretor.items():
    antes = len(df_diretor_limpo)
    df_diretor_limpo = df_diretor_limpo[~df_diretor_limpo[coluna].isin(valor)]
    removidas = antes - len(df_diretor_limpo)

    print(
        f"Diretores: removidas {removidas} linhas "
        f"pela regra: {valor} não estar em {coluna}"
    )

for coluna, valor in remover_escola.items():
    antes = len(df_escola_limpo)
    df_escola_limpo = df_escola_limpo[df_escola_limpo[coluna] != valor]
    removidas = antes - len(df_escola_limpo)

    print(
        f"Escolas: removidas {removidas} linhas "
        f"pela regra: {coluna} != {valor}"
    )

df_aluno_limpo = df_aluno_limpo[colunas_de_interesse_aluno]
df_diretor_limpo = df_diretor_limpo[colunas_de_interesse_diretor]
df_escola_limpo = df_escola_limpo[colunas_de_interesse_escola]

print("\n# ================================================================\n")
print(f'Alunos: {len(df_aluno_limpo)} linhas')
print(f'Diretores: {len(df_diretor_limpo)} linhas')
print(f'Escolas: {len(df_escola_limpo)} linhas')
print("\n# ================================================================\n")

# ================================================================
# Podemos ter problemas ao unificar as tabelas se houver escolas
# com o mesmo ID e multiplos diretores
# ================================================================

print("\n# ================================================================\n")
print(f"Alunos em mais de uma escola: {df_aluno_limpo['ID_ALUNO'].duplicated().sum()}")
print(f"Alunos duplicados: {df_aluno_limpo['ID_ALUNO'].duplicated().sum()}")


print(f"Escolas com ID_ESCOLA duplicados: {df_escola_limpo['ID_ESCOLA'].duplicated().sum()}")

escolas_com_diretores_duplicados = (
    df_diretor_limpo['ID_ESCOLA']
    .value_counts()
    .loc[lambda x: x > 1]
)

escolas_sem_diretores = (
    df_escola_limpo['ID_ESCOLA']
    .loc[~df_escola_limpo['ID_ESCOLA'].isin(df_diretor_limpo['ID_ESCOLA'])]
)

print(f'Escolas com mais de um diretor: {len(escolas_com_diretores_duplicados)}')
print("\n# ================================================================\n")

# ================================================================
# Vamos tratar as Features transformando de categóricas para numéricas quando possivel
# ================================================================

colunas_diretor_numericas = [
    'TX_Q020',
    'TX_Q022'
]

colunas_diretor_ordinais_AD = [
    'TX_Q056',
    'TX_Q057',
    'TX_Q108'
]

colunas_diretor_ordinais_AE = [
    'TX_Q032',
    'TX_Q033',
    'TX_Q035',
    'TX_Q036',
    'TX_Q191'
]

colunas_diretor_booleanas = [
    'TX_Q078', 'TX_Q079', 'TX_Q081', 'TX_Q082',
    'TX_Q083', 'TX_Q085', 'TX_Q087', 'TX_Q119',
    'TX_Q129', 'TX_Q130', 'TX_Q139', 'TX_Q194',
    'TX_Q203', 'TX_Q205', 'TX_Q206', 'TX_Q207',
    'TX_Q208', 'TX_Q209'
]

# ================================================================
# Mapeamentos normalizados [0,1]
# ================================================================

mapa_AD = {
    'A': 0 / 3,
    'B': 1 / 3,
    'C': 2 / 3,
    'D': 3 / 3
}

mapa_AE = {
    'A': 0 / 4,
    'B': 1 / 4,
    'C': 2 / 4,
    'D': 3 / 4,
    'E': 4 / 4
}

mapa_bool = {
    'A': 0.0,
    'B': 1.0
}

# ================================================================
# Conversão das respostas dos diretores
# ================================================================

for col in colunas_diretor_ordinais_AD:
    if col in df_diretor_limpo.columns:
        df_diretor_limpo[col] = (
            df_diretor_limpo[col]
            .map(mapa_AD)
            .astype('float32')
        )

for col in colunas_diretor_ordinais_AE:
    if col in df_diretor_limpo.columns:
        df_diretor_limpo[col] = (
            df_diretor_limpo[col]
            .map(mapa_AE)
            .astype('float32')
        )

for col in colunas_diretor_booleanas:
    if col in df_diretor_limpo.columns:
        df_diretor_limpo[col] = (
            df_diretor_limpo[col]
            .map(mapa_bool)
            .astype('float32')
        )

# Cada linha restante possui um diretor válido
df_diretor_limpo['POSSUI_DIRETOR'] = np.int8(1)

print("\n# ================================================================\n")
print('Registros duplicados de (ID_ESCOLA, ID_SERIE):',df_diretor_limpo.duplicated(subset=['ID_ESCOLA', 'ID_SERIE']).sum())

# ================================================================
# Estatísticas
# ================================================================

pares_aluno = (
    df_aluno_limpo[['ID_ESCOLA', 'ID_SERIE']]
    .drop_duplicates()
)

pares_diretor = (
    df_diretor_limpo[['ID_ESCOLA', 'ID_SERIE']]
)

pares_com_diretor = (
    pares_aluno.merge(
        pares_diretor,
        on=['ID_ESCOLA', 'ID_SERIE'],
        how='left',
        indicator=True
    )
)

qtd_com = (pares_com_diretor['_merge'] == 'both').sum()
qtd_sem = (pares_com_diretor['_merge'] == 'left_only').sum()

print(
    f'Pares escola-série com diretor: '
    f'{qtd_com:,} -> '
    f'{100*qtd_com/len(pares_com_diretor):.2f}%'
)

print(
    f'Pares escola-série sem diretor: '
    f'{qtd_sem:,} -> '
    f'{100*qtd_sem/len(pares_com_diretor):.2f}%'
)

# ================================================================
# Merge final
# ================================================================

df_unificado = (
    df_aluno_limpo
    .merge(
        df_diretor_limpo,
        on=['ID_ESCOLA', 'ID_SERIE'],
        how='left'
    )
    .merge(
        df_escola_limpo,
        on='ID_ESCOLA',
        how='left'
    )
)

# Escolas/séries sem diretor recebem 0
df_unificado['POSSUI_DIRETOR'] = (
    df_unificado['POSSUI_DIRETOR']
    .fillna(0)
    .astype(np.int8)
)

registros_com = int(df_unificado['POSSUI_DIRETOR'].sum())
registros_sem = len(df_unificado) - registros_com

pct_com = 100 * registros_com / len(df_unificado)
pct_sem = 100 - pct_com

print(
    f'Registros com diretor: '
    f'{registros_com:,} -> {pct_com:.2f}%'
)

print(
    f'Registros sem diretor: '
    f'{registros_sem:,} -> {pct_sem:.2f}%'
)

print(f'Linhas finais: {len(df_unificado):,}')
print("\n# ================================================================\n")

# Adicionando a classificação de nível de acordo com a escala SAEB
df_final = df_unificado.copy()

def classificar_nivel_LP(nota_proficiencia):
    if nota_proficiencia < 225:
        return 0
    elif 225 <= nota_proficiencia < 250:
        return 1
    elif 250 <= nota_proficiencia < 275:
        return 2
    elif 275 <= nota_proficiencia < 300:
        return 3
    elif 300 <= nota_proficiencia < 325:
        return 4
    elif 325 <= nota_proficiencia < 350:
        return 5
    elif 350 <= nota_proficiencia < 375:
        return 6
    elif 375 <= nota_proficiencia < 400:
        return 7
    elif nota_proficiencia >= 400:
        return 8
    
def classificar_nivel_MT(nota_proficiencia):
    if nota_proficiencia < 225:
        return 0
    elif 225 <= nota_proficiencia < 250:
        return 1
    elif 250 <= nota_proficiencia < 275:
        return 2
    elif 275 <= nota_proficiencia < 300:
        return 3
    elif 300 <= nota_proficiencia < 325:
        return 4
    elif 325 <= nota_proficiencia < 350:
        return 5
    elif 350 <= nota_proficiencia < 375:
        return 6
    elif 375 <= nota_proficiencia < 400:
        return 7
    elif 400 <= nota_proficiencia < 425:
        return 8
    elif 425 <= nota_proficiencia < 450:
        return 9
    elif nota_proficiencia >= 450:
        return 10

# Adicionando os níveis de proficiência de cada aluno e removendo as colunas originais de proficiência
df_final['NIVEL_LP'] = df_final['PROFICIENCIA_LP_SAEB'].apply(classificar_nivel_LP)
df_final['NIVEL_MT'] = df_final['PROFICIENCIA_MT_SAEB'].apply(classificar_nivel_MT)
df_final = df_final.drop(columns=['PROFICIENCIA_LP_SAEB', 'PROFICIENCIA_MT_SAEB'])

df_final.to_csv('dados/TS_FINAL.csv', index=False)
