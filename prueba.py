#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos desde un archivo de Excel
data = pd.read_excel('Prueba_Analista.xlsx', sheet_name='Consultas')



# In[33]:


# Información general sobre los datos
data = data.drop('Fecha Autorización', axis=1)
informacion_general = data.info()



# In[34]:


# Estadísticas descriptivas
estadisticas_descriptivas = data.describe()
pd.options.display.float_format = '{:.0f}'.format
print(estadisticas_descriptivas)


# In[35]:


frecuencia = data['Fabricante_x'].value_counts()

# Calcular los porcentajes
porcentaje = data['Fabricante_x'].value_counts(normalize=True) * 100

# Crear un DataFrame con las estadísticas
estadisticas = pd.DataFrame({'Frecuencia': frecuencia, 'Porcentaje': porcentaje})

print(estadisticas)


# In[28]:


# Verificar valores nulos
valores_nulos = data.isnull().sum()

print(valores_nulos)


# In[5]:


# Verificar integridad de los datos
integridad = data.apply(lambda x: np.logical_not(x.isnull().any()), axis=1).sum() / len(data) * 100

print(integridad)


# In[6]:


placas_aseguradas = pd.read_excel('Prueba_Analista.xlsx', sheet_name='Placas_Aseguradas')
talleres_autorizados = pd.read_excel('Prueba_Analista.xlsx', sheet_name='Talleres_Autorizados')



# In[7]:


# Verificar si la ciudad del taller está en la hoja de Talleres_Autorizados
talleres_no_autorizados = data[~data['Razon Social Taller'].isin(talleres_autorizados['Taller_Reparaciones'])]['Razon Social Taller'].tolist()

# Verificar si la placa está en la hoja de Placas_Aseguradas
placas_no_aseguradas = data[~data['Placa'].isin(placas_aseguradas['PLACAS_ASEGURADAS'])]['Placa'].tolist()


# In[8]:


valores_unicos = set(talleres_no_autorizados)
print(valores_unicos)


# In[9]:


# Crear un DataFrame vacío para almacenar los resultados
resultados = pd.DataFrame(columns=['Razon Social Taller', 'Cantidad', 'Valor_Total_Sum'])

# Iterar sobre los valores únicos y buscar en la columna 'Razon Social Taller'
for valor in valores_unicos:
    # Filtrar la data por el valor único
    filtrado = data[data['Razon Social Taller'] == valor]
    
    # Contar las veces que aparece y sumar el valor total
    cantidad = len(filtrado)
    valor_total_sum = filtrado['Valor_Total'].sum()
    
    # Crear un nuevo DataFrame con los resultados
    nuevo_df = pd.DataFrame({'Razon Social Taller': [valor], 'Cantidad': [cantidad], 'Valor_Total_Sum': [valor_total_sum]})
    
    # Concatenar el nuevo DataFrame al DataFrame de resultados
    resultados = pd.concat([resultados, nuevo_df], ignore_index=True)

# Mostrar los resultados
pd.options.display.float_format = '{:.2f}'.format
resultados = resultados.sort_values(by='Cantidad', ascending=False)
print(resultados)


# In[10]:


valores_placas = set(placas_no_aseguradas)
print(valores_placas)


# In[11]:


# Crear un DataFrame vacío para almacenar los resultados
resultados_placa = pd.DataFrame(columns=['Placa', 'Cantidad', 'Valor_Total_Sum'])

# Iterar sobre los valores únicos y buscar en la columna 'Razon Social Taller'
for valor in valores_placas:
    # Filtrar la data por el valor único
    filtrado = data[data['Placa'] == valor]
    
    # Contar las veces que aparece y sumar el valor total
    cantidad = len(filtrado)
    valor_total_sum = filtrado['Valor_Total'].sum()
    
    # Crear un nuevo DataFrame con los resultados
    nuevo_df = pd.DataFrame({'Placa': [valor], 'Cantidad': [cantidad], 'Valor_Total_Sum': [valor_total_sum]})
    
    # Concatenar el nuevo DataFrame al DataFrame de resultados
    resultados_placa = pd.concat([resultados_placa, nuevo_df], ignore_index=True)

# Mostrar los resultados
pd.options.display.float_format = '{:.2f}'.format
resultados_placa = resultados_placa.sort_values(by='Cantidad', ascending=False)
print(resultados_placa)


# In[ ]:





# In[12]:


# Visualización de la distribución de los datos
data.hist(bins=20, figsize=(10, 6))
plt.tight_layout()
plt.show()



# In[13]:


# Crear un informe
informe = f"""
Información general:
{informacion_general}

Estadísticas descriptivas:
{estadisticas_descriptivas}

Valores nulos:
{valores_nulos}

Integridad de los datos (% de filas sin valores nulos):
{integridad:.2f}%



Visualización de la distribución de los datos:
(Se adjunta gráfico)
"""

# Guardar el informe en un archivo
with open('informe.txt', 'w') as file:
    file.write(informe)


# In[ ]:




