import feedparser
import pandas as pd
from bs4 import BeautifulSoup
import os

feedparser.USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

FEEDS_FAKES = [
    "https://www.boatos.org/category/saude/feed",
    "https://g1.globo.com/rss/g1/fato-ou-fake/",
    "https://www.e-farsas.com/feed",
]

TERMOS_SAUDE = [
    "emagrece",
    "emagrecimento",
    "dieta",
    "receita",
    "peso",
    "barriga",
    "cápsula",
    "suplemento",
    "gordura",
    "chá",
    "detox",
    "médico",
    "anvisa",
    "cura",
    "doença",
    "câncer",
    "diabetes",
]

DEFAULT_OUTPUT_CSV = os.path.join("dados", "saude_fake_bruto.csv")
DEFAULT_PREVIEW_CSV = os.path.join("dados", "fakes_somente_saude.csv")


def ensure_data_dir():
    os.makedirs("dados", exist_ok=True)


def collect_fake_news(
    force=False, output_csv=DEFAULT_OUTPUT_CSV, preview_csv=DEFAULT_PREVIEW_CSV
):
    ensure_data_dir()  # Garante que o diretório de dados exista antes de salvar arquivos
    if os.path.exists(output_csv) and not force:
        print(f"Arquivo de Fake News já existe em '{output_csv}', pulando coleta.")
        return pd.read_csv(output_csv)

    print("Iniciando coleta de Fake News de saúde via RSS...")
    feedparser.USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    records = []
    # Itera sobre cada feed de Fake News definido na lista
    for url in FEEDS_FAKES:
        print(f"Lendo a fonte: {url}")
        try:  # Tenta ler o feed RSS e processar os artigos
            feed = feedparser.parse(url)
            quantidade = len(feed.entries)
            print(
                f"[{feed.feed.get('title', 'Fonte')}] - Encontrados {quantidade} artigos para analisar."
            )

            # Processa cada artigo do feed, extraindo título, link e conteúdo, e filtrando por termos relacionados à saúde
            for post in feed.entries:
                titulo = post.get("title", "")
                link = post.get("link", "")
                conteudo_bruto = post.get(
                    "content", [{"value": post.get("description", "")}]
                )[0]["value"]
                texto_limpo = BeautifulSoup(conteudo_bruto, "html.parser").get_text(
                    strip=True
                )
                texto_teste = (titulo + " " + texto_limpo).lower()

                if (
                    any(termo in texto_teste for termo in TERMOS_SAUDE)
                    and len(texto_limpo) > 50
                ):
                    records.append(
                        {
                            "titulo": titulo,
                            "texto": texto_limpo,
                            "classe": "Fake",
                            "fonte": "RSS Checkers (Saúde)",
                            "link": link,
                        }
                    )
        except Exception as exc:
            print(f"Erro ao ler o feed {url}: {exc}")

    if not records:
        if os.path.exists(output_csv):
            print("Nenhuma Fake News nova coletada; usando arquivo existente.")
            return pd.read_csv(output_csv)
        raise RuntimeError(
            "Não foi possível coletar Fake News e o arquivo base não existe."
        )

    df = pd.DataFrame(records).drop_duplicates(subset=["titulo"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    df.to_csv(preview_csv, index=False, encoding="utf-8")
    print(
        f"Fake News salvas em '{output_csv}'.\nArquivo de preview salvo em '{preview_csv}'."
    )
    return df


def print_collection_summary(df):
    print(f"\nTotal de Fake News coletadas: {len(df)}")
    print(df[["titulo", "classe"]].head(5).to_string(index=False))


if __name__ == "__main__":
    df_fake = collect_fake_news(force=False)
    print_collection_summary(df_fake)
