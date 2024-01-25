import pandas as pd
import statsmodels.api as sm

# Datos proporcionados
data = {
    'x1': [85.1, 106.3, 50.2, 130.6, 54.8, 30.3, 79.4, 91.0, 135.4, 89.3],
    'x2': [8.5, 12.9, 5.2, 10.7, 3.1, 3.5, 9.2, 9.0, 15.1, 10.2],
    'x3': [5.1, 5.8, 2.1, 8.4, 2.9, 1.2, 3.7, 7.6, 7.7, 4.5],
    'x4': [4.7, 8.8, 15.1, 12.2, 10.6, 3.5, 9.7, 5.9, 20.8, 7.9]
}

df = pd.DataFrame(data)

# Agrega una columna de unos para el intercepto
df['intercept'] = 1

# Variables independientes (X)
X = df[['intercept', 'x1', 'x2', 'x4']]

# Variable dependiente (y)
y = df['x3']

# Ajusta el modelo de regresión lineal con statsmodels
model = sm.OLS(y, X).fit()

# Predice los costos de promoción para la nueva película
new_X = pd.DataFrame({'intercept': [1], 'x1': [100], 'x2': [12], 'x4': [9.2]})
prediction = model.get_prediction(new_X)
pred_mean = prediction.predicted_mean

print("Predicción de costos de promoción:", pred_mean.values[0])

#Scikit-Learn

from sklearn.linear_model import LinearRegression

# Variables independientes (X)
X_scikit = df[['x1', 'x2', 'x4']]

# Variable dependiente (y)
y_scikit = df['x3']

# Inicializa y ajusta el modelo de regresión lineal con Scikit-Learn
model_scikit = LinearRegression()
model_scikit.fit(X_scikit, y_scikit)

# Retorna los coeficientes del modelo
coefficients = model_scikit.coef_

print("Coeficientes del modelo:", coefficients)
