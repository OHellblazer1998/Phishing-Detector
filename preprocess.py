import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer


def preprocess_data(file_path):
    # Carregar o dataset
    data = pd.read_csv(file_path)

    # Remover valores nulos
    data.dropna(subset=['text'], inplace=True)

    # Separar as variáveis independentes (X) e dependentes (y)
    X = data['text']  # Texto do e-mail
    y = data['label']  # Spam ou Ham

    # Dividir o dataset em treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Converter o texto para uma representação numérica usando CountVectorizer
    vectorizer = CountVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test, y_train, y_test, vectorizer
