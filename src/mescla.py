import pandas as pd
import os

print("Carregando e unindo os dados...")
df_fake = pd.read_csv("dados/saude_fake_bruto.csv")
df_fato = pd.read_csv("dados/saude_fatos_bruto.csv")

# 1. Une as bases
df_completo = pd.concat([df_fake, df_fato], ignore_index=True)

# 2. Mantém APENAS as colunas que importam para a IA (O texto e o rótulo)
df_weka = df_completo[["texto", "classe"]].copy()

# 3. Embaralha os dados para o modelo aprender corretamente
df_weka = df_weka.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Salva em um CSV formatado para o Weka
caminho_weka = "dados/base_para_weka.csv"

# O parâmetro quoting=1 obriga o Pandas a colocar aspas duplas em todos os textos.
# Isso é FUNDAMENTAL para o Weka não "quebrar" ao ler vírgulas no meio das notícias.
df_weka.to_csv(caminho_weka, index=False, quoting=1)

print(
    f"\n[SUCESSO] Base finalizada! O arquivo '{caminho_weka}' está pronto para ser aberto no Weka."
)
