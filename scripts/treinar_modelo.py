import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# 1. Carrega e prepara os dados
print("Carregando e misturando os dados...")
df_fake = pd.read_csv("dados/saude_fake_bruto.csv") 
df_fato = pd.read_csv("dados/saude_fatos_bruto.csv")

df_completo = pd.concat([df_fake, df_fato], ignore_index=True)
df_completo = df_completo.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Pré-processamento e Divisão (Treino/Teste)
print("Vetorizando textos e removendo stopwords...")
nltk.download('stopwords', quiet=True)
stop_words_pt = stopwords.words('portuguese')

X = df_completo['texto']
y = df_completo['classe']

# 75% Treino e 25% Teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

vetorizador = CountVectorizer(stop_words=stop_words_pt)
X_treino_vetorizado = vetorizador.fit_transform(X_treino)
X_teste_vetorizado = vetorizador.transform(X_teste)

# 3. Treinamento do Modelo Naive Bayes
print("Treinando o modelo de Inteligência Artificial...")
modelo_nb = MultinomialNB()
modelo_nb.fit(X_treino_vetorizado, y_treino)

# 4. Avaliação e Métricas
y_previsto = modelo_nb.predict(X_teste_vetorizado)
acuracia = accuracy_score(y_teste, y_previsto)

print("\n" + "="*40)
print(f"🎯 ACURÁCIA FINAL DO MODELO: {acuracia * 100:.2f}%")
print("="*40 + "\n")

# 5. Gera e salva a Matriz de Confusão
classes_reais = modelo_nb.classes_
matriz = confusion_matrix(y_teste, y_previsto, labels=classes_reais)

disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=classes_reais)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - Fake News (Saúde)")

caminho_imagem = "dados/matriz_confusao.png"
plt.savefig(caminho_imagem, bbox_inches='tight')
print(f"📊 Imagem da Matriz de Confusão salva com sucesso em: {caminho_imagem}")

# 6. Demonstração Final (Para apresentar ao professor)
textos_demonstracao = [
    "Boato – O oncologista Rafael Onuki Sato gravou um vídeo explicando como o limão pode prevenir o câncer e combater as células cancerígenas.Chega um momento em que as fake news começam, de forma cada vez mais constante, a se repetir. O caso de hoje no Boatos.org fala de uma história com conteúdo muito parecido com desmentidos que já fizemos por aqui, uma mesma vítima de boato e uma forma “só um pouquinho” diferente.Começou a circular com força na internet um vídeo de uma pessoa falando que o limão é o principal aliado contra o câncer. De acordo com o vídeo, o limão é um alimento que deixa o sangue alcalino e faz com que as células do câncer tenham a morte acelerada. Por isso, a receita para prevenir o câncer e combater a doença seria comer dois limões com a casca por dia e tomar o óleo da casca da fruta.Mas não para por aí. O vídeo está sendo acompanhado da informação de que a dica não foi passada “por uma pessoa qualquer”. A filmagem aponta que a pessoa do vídeo é o médico oncologista Rafael Onuki Sato. Leia a mensagem que acompanha o vídeo (optamos por não exibir o vídeo aqui):Dr. Rafael Onuki Sato *ONCOLOGISTA EM LONDRINA* Formado em Medicina pela Universidade Estadual de Londrina (UEL), possui residência em Cirurgia Geral pela Faculdade de Medicina de Marília (FAMEMA) e em Cirurgia Oncológica pelo Hospital de Câncer de Barretos. *PRESTE ATENÇÃO NA FALA DO MÉDICO* [vídeo].Médico oncologista disse que limão combate e mata as células do câncer?A mensagem está circulando muito na internet. Mas será mesmo que o Dr. Rafael Onuki Sato gravou um vídeo recomendando o limão para combater o câncer? E será mesmo que a dica é tão válida assim? A resposta é não. Para você entender tudo, vamos aos fatos.Vamos ao mais simples primeiro. Se você está pensando em seguir a dica do limão achando que nunca terá câncer ou (muito pior) deseja parar um tratamento contra a doença só para usar o limão porque um médico recomendou, calma lá. A pessoa do vídeo não é um médico oncologista.O nome do próprio Dr. Rafael já foi utilizado em outro boato desmentido por aqui: o que fala que o chá da folha de graviola cura o câncer. À época, o próprio médico não só desmentiu que havia gravado um áudio recomendando o uso do chá da fruta contra o câncer como falou a que era contra receitas alternativas. ",
    "A Agência Nacional de Vigilância Sanitária (Anvisa) determinou a apreensão dos medicamentos Gluconex e Tirzedral, produzidos por empresa não identificada. A medida também proíbe a comercialização, a distribuição, a importação e o uso dos produtos. Amplamente divulgados na internet e vendidos como medicamentos injetáveis de GLP-1, os produtos são conhecidos popularmente como canetas emagrecedoras, mas não têm registro, notificação ou cadastro na Anvisa. Em nota, a Anvisa destacou que, por se tratarem de produtos irregulares e de origem desconhecida, não há qualquer garantia quanto ao seu conteúdo ou à sua qualidade. Por isso, não devem ser utilizados em nenhuma hipótese. Profissionais de saúde e pacientes que identificarem produtos das marcas e lotes citados podem entrar em contato com a agência, por meio dos canais de atendimento, ou com a vigilância sanitária local, utilizando os contatos disponíveis no portal da Anvisa."

]  

print("\n--- DEMONSTRAÇÃO AO VIVO (Textos Inéditos) ---\n")
textos_novos_vet = vetorizador.transform(textos_demonstracao)
previsoes = modelo_nb.predict(textos_novos_vet)

for i in range(len(textos_demonstracao)):
    print(f"TEXTO: '{textos_demonstracao[i]}'")
    print(f"-> CLASSIFICAÇÃO DA IA: {previsoes[i]}\n")