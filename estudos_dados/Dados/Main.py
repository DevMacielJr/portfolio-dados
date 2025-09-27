import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar um arquivo CSV
df = pd.read_csv('D:\para_edson\dhav2.csv')

# Visualizar as primeiras linhas do DataFrame
print(df.head())

print('---------------------------------------------------------------------------------------')

# Remover linhas com valores nulos
df.dropna(inplace=True)

# Remover duplicatas
df.drop_duplicates(inplace=True)

# Verificar informações sobre os dados
print(df.info())

print('---------------------------------------------------------------------------------------')

# Estatísticas descritivas
print(df.describe())

print('---------------------------------------------------------------------------------------')

# Ver distribuição de uma coluna
print(df['coluna_de_interesse'].value_counts())

print('---------------------------------------------------------------------------------------')

# Exemplo de gráfico de barras
sns.barplot(x='coluna_x', y='coluna_y', data=df)
plt.show()

# Exemplo de histograma
df['coluna_de_interesse'].hist()
plt.show()

# Matriz de correlação
correlacao = df.corr()
print(correlacao)

print('---------------------------------------------------------------------------------------')

# Mapa de calor da correlação
sns.heatmap(correlacao, annot=True, cmap='coolwarm')
plt.show()

