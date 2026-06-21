# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

colunas_de_interesse_diretor = ['ID_ESCOLA', 'TX_Q020','TX_Q022','TX_Q032','TX_Q033','TX_Q035',
'TX_Q036','TX_Q056','TX_Q057','TX_Q078','TX_Q079','TX_Q081','TX_Q082','TX_Q083', 'TX_Q085','TX_Q087',
'TX_Q108','TX_Q119','TX_Q129','TX_Q130','TX_Q139','TX_Q191','TX_Q194','TX_Q203','TX_Q205','TX_Q206','TX_Q207',
'TX_Q208','TX_Q209']

remover_diretor = {
    'IN_PREENCHIMENTO_QUESTIONARIO': [0]}     # Remover diretores que não preencheram o questionário

colunas_de_interesse_escola = ['ID_ESCOLA','PC_FORMACAO_DOCENTE_MEDIO','TAXA_PARTICIPACAO_EM',
    'MEDIA_EMT_LP','MEDIA_EMT_MT','MEDIA_EMI_LP','MEDIA_EMI_MT','MEDIA_EM_LP','MEDIA_EM_MT']

remover_escola = {
    'NU_PRESENTES_EM': 0}                   # Remover escolas que não tenham alunos presentes no ensino médio


# ================================================================
# Aplicaremos as regras de limpeza de dados e selecionaremos as features
# ================================================================

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


# ================================================================
# Podemos ter problemas ao unificar as tabelas se houver escolas
# com o mesmo ID e multiplos diretores
# ================================================================

print(f"Alunos em mais de uma escola: {df_aluno_limpo['ID_ALUNO'].duplicated().sum()}")
print(f"Alunos duplicados: {df_aluno_limpo['ID_ALUNO'].duplicated().sum()}")

print("\n# ================================================================\n")

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
print(f'Escolas sem diretor: {len(escolas_sem_diretores)}')


# ================================================================
# Vamos tratar os casos de escolas com mais de um diretor
# ================================================================
colunas_numericas = [
    'TX_Q020',
    'TX_Q022'
]

colunas_ordinais_AD = [
    'TX_Q056',
    'TX_Q057',
    'TX_Q108'
]

colunas_ordinais_AE = [
    'TX_Q032',
    'TX_Q033',
    'TX_Q035',
    'TX_Q036',
    'TX_Q191'
]

colunas_booleanas = [
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
    'A': 0/3,
    'B': 1/3,
    'C': 2/3,
    'D': 3/3
}

mapa_AE = {
    'A': 0/4,
    'B': 1/4,
    'C': 2/4,
    'D': 3/4,
    'E': 4/4
}

mapa_bool = {
    'A': 0.0,
    'B': 1.0
}

# ================================================================
# Conversão das respostas
# ================================================================

for col in colunas_ordinais_AD:
    if col in df_diretor_limpo.columns:
        df_diretor_limpo[col] = (
            df_diretor_limpo[col]
            .map(mapa_AD)
            .astype('float32')
        )

for col in colunas_ordinais_AE:
    if col in df_diretor_limpo.columns:
        df_diretor_limpo[col] = (
            df_diretor_limpo[col]
            .map(mapa_AE)
            .astype('float32')
        )

for col in colunas_booleanas:
    if col in df_diretor_limpo.columns:
        df_diretor_limpo[col] = (
            df_diretor_limpo[col]
            .map(mapa_bool)
            .astype('float32')
        )

# ================================================================
# Regras de agregação
# ================================================================

agg = {}

for col in colunas_numericas:
    if col in df_diretor_limpo.columns:
        agg[col] = 'mean'

for col in colunas_ordinais_AD + colunas_ordinais_AE:
    if col in df_diretor_limpo.columns:
        agg[col] = 'mean'

for col in colunas_booleanas:
    if col in df_diretor_limpo.columns:
        agg[col] = 'max'

# ================================================================
# Uma linha por escola
# ================================================================

df_diretor_agregado = (
    df_diretor_limpo
    .groupby('ID_ESCOLA', as_index=False)
    .agg(agg)
)

df_diretor_agregado['POSSUI_DIRETOR'] = np.int8(1)

print(f'Escolas com diretor: {len(df_diretor_agregado)} -> {100* len(df_diretor_agregado)/len(df_escola_limpo):.2f}%')
print(f'Escolas sem diretor: {len(df_escola_limpo) - len(df_diretor_agregado)} -> {100* (len(df_escola_limpo) - len(df_diretor_agregado))/len(df_escola_limpo):.2f}%')

# ================================================================
# Merge final
# ================================================================

df_unificado = (
    df_aluno_limpo
    .merge(
        df_diretor_agregado,
        on='ID_ESCOLA',
        how='left'
    )
    .merge(
        df_escola_limpo,
        on='ID_ESCOLA',
        how='left'
    )
)

# Escolas sem diretor recebem 0
df_unificado['POSSUI_DIRETOR'] = (
    df_unificado['POSSUI_DIRETOR']
    .fillna(0)
    .astype(np.int8)
)

print(f'Registros com diretor: {df_unificado["POSSUI_DIRETOR"].sum():} -> {100* df_unificado["POSSUI_DIRETOR"].sum()/len(df_unificado):.2f}%')
print(f'Registros sem diretor: {len(df_unificado) - df_unificado["POSSUI_DIRETOR"].sum():} -> {100* (len(df_unificado) - df_unificado["POSSUI_DIRETOR"].sum())/len(df_unificado):.2f}%')
print(f'Linhas finais: {len(df_unificado)}')