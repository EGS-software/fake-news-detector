import pandas as pd

# Carrega os arquivos
df_fake = pd.read_csv("dados/saude_fake_bruto.csv")
df_fato = pd.read_csv("dados/saude_fatos_bruto.csv")

# Une os arquivos e embaralha as linhas
df_completo = pd.concat([df_fake, df_fato], ignore_index=True)
df_completo = df_completo.sample(frac=1, random_state=42).reset_index(drop=True)

# Imprime os dados para o seu relatório
print("--- COMPOSIÇÃO DA BASE DE DADOS ---")
print(f"Total de textos: {len(df_completo)}")
print(df_completo['classe'].value_counts())