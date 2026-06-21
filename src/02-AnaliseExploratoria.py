df_analise = df_unificado.copy()

# proporção de alunos por ID_UF
proporcao_alunos_por_uf = (
    df_analise['ID_UF']
    .value_counts(normalize=True)
    .sort_index()
)
print(proporcao_alunos_por_uf)