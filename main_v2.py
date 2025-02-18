import pandas as pd
from preprocess import preprocess_data
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Caminho para o dataset
file_path = 'data/spam_ham_dataset.csv'

# Pré-processar o dataset
X_train, X_test, y_train, y_test, vectorizer = preprocess_data(file_path)
print("Pré-processamento completo!")

# Treinar um modelo Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Exibir métricas de avaliação
print("\nMétricas de Avaliação:")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Testar com uma nova mensagem
new_email = ["Congratulations! You've won a free iPhone. Claim now."]
new_email_transformed = vectorizer.transform(new_email)
prediction = model.predict(new_email_transformed)
print("\nPrevisão para nova mensagem:", "Spam" if prediction[0] == "spam" else "Ham")
