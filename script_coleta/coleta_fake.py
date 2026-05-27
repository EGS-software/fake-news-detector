import feedparser
import pandas as pd
from bs4 import BeautifulSoup
import os

# 1. O TRUQUE DE MESTRE: Disfarçar o feedparser para evitar bloqueios de robôs
feedparser.USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# 2. Feeds Ampliados (O filtro abaixo fará o trabalho de limpar)
FEEDS_FAKES = [
    "https://www.boatos.org/category/saude/feed", # Categoria oficial de saúde
    "https://g1.globo.com/rss/g1/fato-ou-fake/",  # Fato ou Fake do G1 (Geral)
    "https://www.e-farsas.com/feed"               # E-farsas (Geral)
]

# 3. Filtro estrito: O texto DEVE conter termos do nosso nicho
TERMOS_SAUDE = [
    "emagrece", "emagrecimento", "dieta", "receita", "peso", "barriga", 
    "cápsula", "suplemento", "gordura", "chá", "detox", "médico", 
    "anvisa", "cura", "doença", "câncer", "diabetes"
]

dados_coletados = []

def extrair_fakes_saude():
    print("Iniciando coleta anti-bloqueio de Fake News (Filtrando apenas Saúde)...")
    
    for url in FEEDS_FAKES:
        print(f"\nLendo a fonte: {url}")
        
        try:
            feed = feedparser.parse(url)
            
            # Se ainda vier 0, a URL pode estar instável, mas o script continua pras outras
            quantidade = len(feed.entries)
            print(f"[{feed.feed.get('title', 'Fonte')}] - Encontrados {quantidade} artigos para analisar.")
            
            for post in feed.entries:
                titulo = post.title
                link = post.link
                
                # Extrai o texto ignorando o HTML
                conteudo_bruto = post.get('content', [{'value': post.get('description', '')}])[0]['value']
                texto_limpo = BeautifulSoup(conteudo_bruto, 'html.parser').get_text(strip=True)
                
                texto_teste = (titulo + " " + texto_limpo).lower()
                
                # A mágica do filtro: Verifica se o texto tem relação com o nosso tema
                if any(termo in texto_teste for termo in TERMOS_SAUDE):
                    
                    # Ignora textos muito curtos e propagandas vazias
                    if len(texto_limpo) > 50:
                        dados_coletados.append({
                            "titulo": titulo,
                            "texto": texto_limpo,
                            "classe": "Fake",
                            "fonte": "RSS Checkers (Saúde)",
                            "link": link
                        })
                        
        except Exception as e:
            print(f"Erro ao ler o feed {url}: {e}")

# 4. Execução
if __name__ == "__main__":
    extrair_fakes_saude()
    
    # 5. Salvamento dos Dados
    if dados_coletados:
        df = pd.DataFrame(dados_coletados)
        os.makedirs('dados', exist_ok=True)
        
        caminho_arquivo = 'dados/fakes_somente_saude.csv'
        
        # Remove eventuais notícias duplicadas que saíram em mais de um feed
        df = df.drop_duplicates(subset=['titulo'])
        
        df.to_csv(caminho_arquivo, index=False, encoding='utf-8')
        print(f"\n[SUCESSO] {len(df)} Fake News puramente de Saúde salvas em: {caminho_arquivo}")
        
        print("\nExemplo dos primeiros títulos capturados:")
        for i, titulo in enumerate(df['titulo'].head(5)):
            print(f"{i+1}. {titulo}")
            
    else:
        print("\nNenhum dado capturado que corresponda ao filtro de saúde.")