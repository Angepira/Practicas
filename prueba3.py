#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# Generación de datos de entrada
data = {'Pelicula': ['Robocop', 'El padrino', 'Wonka', 'Star Wars'],
        'Action': [1, 0, 0, 1],
        'Drama': [0, 1, 1, 0],
        'Comedy': [0, 0, 1, 0],
        'Sci-Fi': [1, 0, 0, 1],
        'Rating': [4.5, 3.0, 2.0, 4.0]}
df = pd.DataFrame(data)

# Captura de las preferencias del usuario
preferencias_usuario = list(map(int, input().split()))



X = df[['Action', 'Drama', 'Comedy', 'Sci-Fi']]
y = df['Rating']
feature_names = ['Robocop', 'El padrino', 'Wonka', 'Star Wars']  

scaler = StandardScaler(with_mean=False, with_std=False)
X_scaled = scaler.fit_transform(X)


preferencias_usuario_matriz = np.array(preferencias_usuario).reshape(1, -1)
preferencias_usuario_matriz_escalada = scaler.fit_transform(preferencias_usuario_matriz)


# Entrenamiento del Modelo Predictivo
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Regresión Lineal
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Red Neuronal
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)

# Predicción de Películas Recomendadas
predicciones = {}
for i, (genero, preferencia) in enumerate(zip(feature_names, preferencias_usuario_matriz_escalada[0])):
    preferencia_matriz_escalada = np.zeros((1, 4))
    preferencia_matriz_escalada[0, i] = preferencia
    lr_pred = lr_model.predict(preferencia_matriz_escalada)[0]
    mlp_pred = mlp_model.predict(preferencia_matriz_escalada)[0]
    predicciones[genero] = {'Regresion_Lineal': lr_pred, 'Perceptron_Multicapa(MLP)': mlp_pred}


# Ordenar el diccionario por la calificación predicha por el modelo de regresión lineal
predicciones = {k: v for k, v in sorted(predicciones.items(), key=lambda item: item[1]['Regresion_Lineal'], reverse=True)}

# Imprimir las predicciones
print("PeliculaRegresion LinealPerceptron Multicapa (MLP)")
for i, (genero, pred) in enumerate(predicciones.items()):
    print(f"{i} {genero} {pred['Regresion_Lineal']:.1f} {pred['Perceptron_Multicapa(MLP)']:.1f}")

