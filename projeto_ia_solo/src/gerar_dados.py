import pandas as pd
import numpy as np

# Garante resultados iguais em execuções
np.random.seed(42)

# Gerar dados simulados
n = 100
ph = np.random.normal(loc=6.5, scale=0.4, size=n).clip(5.5, 7.5)
umidade = np.random.normal(loc=40, scale=10, size=n).clip(10, 60)

# Simular altura da planta com base em ph e umidade
altura_planta = (
    50 +
    (umidade - 20) * 0.3 +
    (-(ph - 6.5)**2 * 5) +  # máximo da parábola em ph = 6.5
    np.random.normal(0, 2, size=n)  # ruído
).clip(50, 75)

# Montar DataFrame
df = pd.DataFrame({
    "ph": ph.round(2),
    "umidade": umidade.round(1),
    "altura_planta": altura_planta.round(1)
})

# Salvar
df.to_csv('../data/dados_solo_plantas.csv', index=False)
print("Arquivo gerado com sucesso!")
