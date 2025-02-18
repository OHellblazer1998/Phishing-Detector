import pandas as pd

# Carregar o dataset
data = pd.read_csv('data/spam_ham_dataset.csv')
file_path = 'data/spam_ham_dataset.csv'

# Visualizar as primeiras linhas
print("Primeiras linhas do dataset:")
print(data.head())

# Informações básicas do dataset
print("\nInformações gerais do dataset:")
print(data.info())

# Verificar a distribuição das classes
print("\nDistribuição das classes (Phishing vs Legítimos):")
print(data['label'].value_counts())