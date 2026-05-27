import feedparser
import pandas as pd
from bs4 import BeautifulSoup
import os

# 1. Configurações Iniciais
# Vamos usar o feed RSS de Saúde/Bem-Estar de um portal confiável (ex: G1 Saúde)
# Isso garante textos com linguagem jornalística, revisada e factual.
URL_FEED = "https://g1.globo.com/dynamo/saude/rss2.xml"

dados_coletados = []

def extrair_noticias_fato():
    print(f"Lendo o Feed RSS oficial de Saúde: {URL_FEED}")
    
    try:
        # O feedparser lê o XML e transforma em um dicionário Python
        feed = feedparser.parse(URL_FEED)
        
        print(f"Encontrados {len(feed.entries)} artigos recentes.")
        
        for post in feed.entries:
            titulo = post.title
            link = post.link
            
            # O resumo (description) geralmente vem com tags HTML, então usamos BeautifulSoup para limpar
            resumo_html = post.get('description', '')
            texto_limpo = BeautifulSoup(resumo_html, 'html.parser').get_text(strip=True)
            
            # Filtramos para pegar apenas textos que tenham conteúdo relevante
            if len(texto_limpo) > 30:
                dados_coletados.append({
                    "titulo": titulo,
                    "texto": texto_limpo,
                    "classe": "Fato", # Rotulando conforme exigência do professor
                    "fonte": "G1 Saúde / Oficial",
                    "link": link
                })
                
    except Exception as e:
        print(f"Erro ao processar o feed RSS: {e}")

# 2. Execução
if __name__ == "__main__":
    extrair_noticias_fato()
    
    # 3. Salvamento dos Dados
    if dados_coletados:
        df = pd.DataFrame(dados_coletados)
        os.makedirs('dados', exist_ok=True)
        
        caminho_arquivo = 'dados/saude_fatos_bruto.csv'
        df.to_csv(caminho_arquivo, index=False, encoding='utf-8')
        print(f"\nSucesso! {len(dados_coletados)} textos coletados e salvos em {caminho_arquivo}")
    else:
        print("\nNenhum dado foi coletado. Verifique a URL do Feed.")