import feedparser
import pandas as pd
from bs4 import BeautifulSoup
import os

URL_FEED = "https://g1.globo.com/dynamo/saude/rss2.xml"
DEFAULT_OUTPUT_CSV = os.path.join("dados", "saude_fatos_bruto.csv")


def ensure_data_dir():
    os.makedirs("dados", exist_ok=True)


def collect_fact_news(force=False, output_csv=DEFAULT_OUTPUT_CSV):
    ensure_data_dir()
    if os.path.exists(output_csv) and not force:
        print(f"Arquivo de fatos já existe em '{output_csv}', pulando coleta.")
        return pd.read_csv(output_csv)

    print(f"Lendo o Feed RSS oficial de Saúde: {URL_FEED}")
    records = []
    try:
        feed = feedparser.parse(URL_FEED)
        print(f"Encontrados {len(feed.entries)} artigos recentes.")
        for post in feed.entries:
            titulo = post.get("title", "")
            link = post.get("link", "")
            resumo_html = post.get("description", "")
            texto_limpo = BeautifulSoup(resumo_html, "html.parser").get_text(strip=True)
            if len(texto_limpo) > 30:
                records.append(
                    {
                        "titulo": titulo,
                        "texto": texto_limpo,
                        "classe": "Fato",
                        "fonte": "G1 Saúde / Oficial",
                        "link": link,
                    }
                )
    except Exception as exc:
        print(f"Erro ao processar o feed RSS: {exc}")

    if not records:
        if os.path.exists(output_csv):
            print("Nenhuma notícia nova coletada; usando arquivo existente.")
            return pd.read_csv(output_csv)
        raise RuntimeError(
            "Não foi possível coletar fatos e o arquivo base não existe."
        )

    df = pd.DataFrame(records).drop_duplicates(subset=["titulo"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Notícias de fato salvas em '{output_csv}'.")
    return df


def print_collection_summary(df):
    print(f"\nTotal de fatos coletados: {len(df)}")
    print(df[["titulo", "classe"]].head(5).to_string(index=False))


if __name__ == "__main__":
    df_fato = collect_fact_news(force=False)
    print_collection_summary(df_fato)
