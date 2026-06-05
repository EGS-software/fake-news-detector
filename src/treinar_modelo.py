import os

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATA_DIR = "dados"
CONFUSION_PNG = os.path.join(DATA_DIR, "matriz_confusao.png")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def normalize_text(df, text_column="texto"):
    df = df.copy()
    df[text_column] = df[text_column].astype(str).str.lower()
    df[text_column] = df[text_column].str.replace("\n", " ", regex=False)
    df[text_column] = df[text_column].str.replace("\r", " ", regex=False)
    df[text_column] = df[text_column].str.replace('"', "'", regex=False)
    return df


def build_dataset(fake_df, fact_df, save_path=None):
    ensure_data_dir() # Garante que o diretório de dados exista antes de salvar arquivos
    df_completo = pd.concat([fake_df, fact_df], ignore_index=True)
    df_completo = normalize_text(df_completo, "texto")

    # Embaralha os dados para evitar qualquer ordenação que possa influenciar o modelo
    df_completo = df_completo.sample(frac=1, random_state=42).reset_index(drop=True)
    dataset = df_completo[["texto", "classe"]].copy() # Seleciona apenas as colunas necessárias para o modelo
    if save_path:
        dataset.to_csv(save_path, index=False, quoting=1)
        print(f"Conjunto de dados salvo em '{save_path}'.")
    return dataset


def train_model(df, save_confusion_path=CONFUSION_PNG):
    ensure_data_dir()
    print("Treinando o modelo de classificação...\n")
    nltk.download("stopwords", quiet=True)
    stop_words_pt = stopwords.words("portuguese") # Lista de stopwords em português para melhorar a vetorização

    # Prepara os dados para treinamento e teste
    X = df["texto"]
    y = df["classe"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Vetorização dos textos usando CountVectorizer com remoção de stopwords
    vectorizer = CountVectorizer(stop_words=stop_words_pt)

    # Ajusta o vetor de treinamento e transforma o vetor de teste usando o mesmo vocabulário
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB() # Cria o modelo de classificação
    model.fit(X_train_vec, y_train) # Treina o modelo com os dados de treinamento

    y_pred = model.predict(X_test_vec) # Faz previsões com os dados de teste
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 40)
    print(f"🎯 Acurácia final do modelo: {accuracy * 100:.2f}%")
    print("=" * 40 + "\n")

    # Gera e salva a matriz de confusão para análise visual do desempenho do modelo
    labels = model.classes_
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusão - Fake News (Saúde)")
    plt.savefig(save_confusion_path, bbox_inches="tight")
    plt.close()
    print(f"Imagem de matriz de confusão salva em: '{save_confusion_path}'")

    return model, vectorizer, accuracy


def count_dataset_predictions(model, vectorizer, df):
    """Função para contar quantas vezes o modelo prevê cada classe na base completa, 
    útil para análise de distribuição de previsões."""
    X = df["texto"]
    y_pred = model.predict(vectorizer.transform(X))
    counts = pd.Series(y_pred).value_counts()
    return counts.to_dict()
