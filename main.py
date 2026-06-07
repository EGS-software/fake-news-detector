import argparse
import os

import pandas as pd
from src.service.coleta_fake import collect_fake_news
from src.service.coleta_fato import collect_fact_news
from src.treinar_modelo import build_dataset, count_dataset_predictions, train_model

DATA_DIR = "dados"
FAKE_CSV = os.path.join(DATA_DIR, "saude_fake_bruto.csv")
FACT_CSV = os.path.join(DATA_DIR, "saude_fatos_bruto.csv")
WEKA_CSV = os.path.join(DATA_DIR, "base_para_weka.csv")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)


def summarize_dataset(df):
    print("\n--- COMPOSIÇÃO DA BASE DE DADOS ---")
    print(f"Total de textos: {len(df)}")
    print(df["classe"].value_counts().to_string())


def main(args=None):  # Comandos para controle de fluxo via linha de comando
    parser = argparse.ArgumentParser(
        description="Centraliza a coleta de dados e o treinamento do modelo."
    )
    parser.add_argument(
        "--force-collect",
        action="store_true",
        help="Força coleta de RSS mesmo quando os arquivos já existem.",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Usa apenas os arquivos existentes e não executa coleta de RSS.",
    )
    parser.add_argument(
        "--skip-train", action="store_true", help="Pula o treinamento do modelo."
    )
    parsed = parser.parse_args(args=args)

    ensure_data_dir()  # Força existência de path e files necessários

    if (
        parsed.skip_collect
    ):  # Com flag de skip, não tenta coletar e apenas carrega os arquivos existentes
        print("Pulando coleta de RSS e usando dados existentes.")
        fake_df = load_csv(FAKE_CSV)
        fact_df = load_csv(FACT_CSV)
    else:  # Coleta os dados normalmente, respeitando a flag de força coleta
        fake_df = collect_fake_news(force=parsed.force_collect)
        fact_df = collect_fact_news(force=parsed.force_collect)

    # Construção da base de dados final e resumo
    df = build_dataset(
        fake_df, fact_df, save_path=WEKA_CSV
    )  # Salva a base final para uso em Weka
    summarize_dataset(df)

    if (
        not parsed.skip_train
    ):  # Treina o modelo e exibe resultados, a menos que a flag de skip seja usada
        model, vectorizer, accuracy = train_model(df)
        # Contagem de previsões do modelo na base completa para análise adicional
        prediction_counts = count_dataset_predictions(model, vectorizer, df)
        print("\n--- CONTAGEM DE PREVISÕES DO MODELO ---")
        for label, count in prediction_counts.items():
            print(f"{label}: {count}")
        print(f"\nAcurácia do teste: {accuracy * 100:.2f}%")
    else:
        print("Treinamento do modelo foi pulado.")

    print("\nFluxo concluído com sucesso.")


if __name__ == "__main__":
    main()
