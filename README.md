# 🩺 Pipeline de Detecção de Fake News na Saúde

Este projeto implementa um pipeline automatizado de ponta a ponta (end-to-end) em Python para a coleta de dados, processamento de texto, treinamento de modelo e avaliação estatística para classificação de notícias em **Fato** ou **Fake**, focando na área da saúde.

O sistema utiliza técnicas de Processamento de Linguagem Natural (PLN/NLP) e algoritmos de Machine Learning clássicos para identificar desinformação de maneira rápida e com alta acurácia.

---

## 🚀 Funcionalidades

- **Coleta de Dados Automatizada**: Coleta notícias verdadeiras e falsas a partir de feeds RSS e técnicas de web scraping (utilizando BeautifulSoup de forma implícita).
- **Interface de Linha de Comando (CLI)**: Centralização do fluxo por meio do `argparse`, permitindo pular ou forçar etapas por meio de flags no terminal.
- **Normalização Textual**: Limpeza e padronização dos textos (remoção de quebras de linha, padronização de aspas, conversão para minúsculas).
- **Extração de Features (NLP)**: Tokenização e vetorização textual com remoção inteligente de *stopwords* em português via `CountVectorizer`.
- **Inteligência Artificial**: Treinamento do modelo probabilístico `Multinomial Naive Bayes` (altamente adequado para contagem de frequências textuais).
- **Métricas de Desempenho**: Geração automática de relatórios de acurácia e exportação visual da **Matriz de Confusão** (em tons de azul).
- **Compatibilidade WEKA**: Exportação automática da base de dados unificada no formato CSV adaptado para mineração de dados no software WEKA.

---

## 📁 Estrutura do Projeto

Abaixo está a disposição sugerida e implícita da arquitetura do projeto:

```text
├── dados/
│   ├── saude_fake_bruto.csv     # Dados coletados de fontes falsas
│   ├── saude_fatos_bruto.csv    # Dados coletados de fontes confiáveis
│   ├── base_para_weka.csv       # Dataset processado e unificado
│   └── matriz_confusao.png      # Gráfico de desempenho gerado no treino
├── src/
│   ├── service/
│   │   ├── coleta_fake.py       # Lógica de scraping/RSS de fakes
│   │   └── coleta_fato.py       # Lógica de scraping/RSS de fatos
│   └── treinar_modelo.py        # Módulo de NLP e IA (fase de treino/teste)
└── main.py                      # Script maestro / Centralizador do fluxo

```

---

## 💻 Como Executar

O fluxo principal é controlado pelo arquivo principal (`main.py`) e pode ser parametrizado no terminal.

### 1. Execução Padrão (Fluxo Completo)

Para rodar o ciclo completo (Coleta -> Processamento -> Treinamento -> Avaliação), execute sem nenhuma flag:

```bash
python main.py

```

### 2. Pular a Coleta de Dados

Se você já possui os arquivos brutos baixados na pasta `dados/` e deseja apenas treinar o modelo:

```bash
python main.py --skip-collect

```

### 3. Forçar Nova Coleta (Atualizar RSS)

Caso queira obrigar o sistema a rodar o scraping e buscar as atualizações mais recentes do feed RSS:

```bash
python main.py --force-collect

```

### 4. Apenas Coletar e Processar (Pular Treino)

Se o seu objetivo for apenas atualizar a base de dados unificada sem disparar o treinamento da inteligência artificial:

```bash
python main.py --skip-train

```

---

## 🧠 Como o Modelo Funciona por trás dos panos

1. **Remoção de Stopwords**: Palavras neutras (como "o", "da", "em", "que") são deletadas porque ocorrem igualmente em textos falsos e verdadeiros, não agregando valor preditivo.
2. **Vetorização (Saco de Palavras)**: O texto é quebrado em tokens (palavras) e transformado em uma matriz matemática onde as colunas são as palavras do vocabulário e as linhas registram a frequência delas em cada notícia.
3. **Divisão de Dados (Validação Cruzada)**: O pipeline divide a base em **75% para Treino** (onde a IA constrói o dicionário probabilístico) e **25% para Teste** (onde o modelo é desafiado a classificar sem ver as respostas).
4. **Cálculo Probabilístico**: O algoritmo `MultinomialNB` calcula a probabilidade estatística de um texto pertencer à classe *Fake* ou *Fato* baseado no peso das palavras encontradas.

---

## 📊 Resultados e Matriz de Confusão

Ao fim do treinamento, o modelo exporta uma imagem avaliativa (`matriz_confusao.png`) que detalha matematicamente os acertos:

* **Verdadeiros Positivos / Negativos**: Diagonais escuras que representam quando a IA acertou o palpite (ex: notícias fakes classificadas como fakes).
* **Falsos Positivos / Negativos**: Quadrantes claros que mostram os erros cometidos pelo modelo (como alarmes falsos de notícias verdadeiras classificadas como suspeitas).

O projeto atinge níveis elevados de acurácia (frequentemente superiores a 95%), validando a eficácia de abordagens probabilísticas no combate à desinformação na saúde.

---

## 🛠️ Pré-requisitos e Dependências

Para rodar este ecossistema, certifique-se de ter instalado os pacotes Python descritos abaixo:

```bash
pip install pandas matplotlib nltk scikit-learn

```