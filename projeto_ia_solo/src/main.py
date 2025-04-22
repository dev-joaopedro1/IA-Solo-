import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Criar a pasta 'visualization' caso ela não exista
os.makedirs("../visualization", exist_ok=True)

# Carregar os dados
df = pd.read_csv('../data/dados_solo_plantas.csv')

# Padronizar os nomes das colunas
df.columns = df.columns.str.strip().str.lower()

# Verificar colunas disponíveis
print("Colunas disponíveis:", df.columns.tolist())

# Selecionar colunas para o modelo (exceto a altura da planta)
X = df.drop(columns=["altura_planta"], errors='ignore')

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Criar e treinar o modelo KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df["grupo"] = kmeans.fit_predict(X_scaled)

# Salvar CSV com os grupos na pasta 'visualization'
df.to_csv('../visualization/dados_com_grupos.csv', index=False)

# Estilo para os gráficos
sns.set(style="whitegrid")

# Gráfico 1: Altura da planta por grupo
sns.boxplot(data=df, x="grupo", y="altura_planta")
plt.title("Altura da Planta por Grupo")
plt.savefig("../visualization/boxplot_grupo_altura.png")
plt.clf()

# Gráfico 2: pH vs Altura da Planta
if "ph" in df.columns:
    sns.scatterplot(data=df, x="ph", y="altura_planta", hue="grupo", palette="Set2")
    plt.title("Agrupamento por pH e Altura da Planta")
    plt.savefig("../visualization/ph_vs_altura.png")
    plt.clf()

# Gráfico 3: Umidade vs Altura da Planta
if "umidade" in df.columns:
    sns.scatterplot(data=df, x="umidade", y="altura_planta", hue="grupo", palette="Set2")
    plt.title("Agrupamento por Umidade e Altura da Planta")
    plt.savefig("../visualization/umidade_vs_altura.png")
    plt.clf()