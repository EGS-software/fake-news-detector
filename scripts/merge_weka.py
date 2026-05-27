import pandas as pd
import os

print("Iniciando o processamento e unificação das bases...")

# 1. Carrega os arquivos CSV locais
df_fake = pd.read_csv("dados/saude_fake_bruto.csv") 
df_fato = pd.read_csv("dados/saude_fatos_bruto.csv")

# 2. Une as bases e isola as colunas necessárias
df_completo = pd.concat([df_fake, df_fato], ignore_index=True)
df_weka = df_completo[['texto', 'classe']].copy()

# NORMALIZAÇÃO: LOWERCASE DESDE O INÍCIO
# Converte todo o texto para minúsculo e garante que seja tratado como String
df_weka['texto'] = df_weka['texto'].astype(str).str.lower()

# 3. Limpeza de caracteres de controle e aspas que quebram o Weka
df_weka['texto'] = df_weka['texto'].str.replace('\n', ' ', regex=False)
df_weka['texto'] = df_weka['texto'].str.replace('\r', ' ', regex=False)
df_weka['texto'] = df_weka['texto'].str.replace('"', "'", regex=False)

# 4. Embaralha as linhas de forma reprodutível (random_state fixo)
df_weka = df_weka.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Exporta o CSV 
caminho_weka = "dados/base_para_weka.csv"
df_weka.to_csv(caminho_weka, index=False, quoting=1)

print(f"\n[SUCESSO] Base gerada e normalizada em lowercase!")
print(f"Arquivo salvo pronto para o Weka em: '{caminho_weka}'")