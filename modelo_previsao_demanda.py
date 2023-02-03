import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Carregando o conjunto de dados
df = pd.read_excel("/home/alexandre/Downloads/Dados.xlsx", parse_dates=["Data"], index_col="Data")

# Agrupar os dados por dia
df = df.resample("D").sum()

# Verificando se os dados são estacionários
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['Vendas'])
print("ADF p-value: ", result[1])

# Verificando se p-value é maior que 0.05, significa que os dados não são estacionários
if result[1] > 0.05:
    # Aplicando diferenciação para tornar os dados estacionários
    df['Vendas_diff'] = df['Vendas'] - df['Vendas'].shift()
    result = adfuller(df['Vendas_diff'].dropna())
    print("ADF p-value after differencing: ", result[1])

# Aplicando o modelo de média móvel
model = ExponentialSmoothing(df['Vendas'], seasonal_periods=7, trend="add", seasonal="add").fit()

# Fazendo previsões para os próximos 5 dias
forecast = model.forecast(steps=5)
print("Previsão de demanda para os próximos 5 dias:")
print(forecast)

# Plotando previsões
plt.plot(df['Vendas'], label='Dados')
plt.plot(forecast, label='Previsão', marker="o")
plt.legend(loc='best')
plt.show()
