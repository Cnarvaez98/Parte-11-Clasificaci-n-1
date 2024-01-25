import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Cargar el dataset (asegúrate de tener el nombre correcto del archivo CSV)
df = pd.read_csv('nombre_del_archivo.csv')

# Eliminar la columna 'categoria_edad'
df = df.drop('categoria_edad', axis=1)

# Separar las características (X) y la variable objetivo (y)
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Partición estratificada en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Ajustar un modelo de Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred = rf_model.predict(X_test)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(conf_matrix)

# Calcular el accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del modelo: {accuracy}')

# Calcular el F1-Score
f1 = f1_score(y_test, y_pred)
print(f'F1-Score del modelo: {f1}')

# Puedes experimentar con los parámetros del Random Forest para optimizar el rendimiento
# Por ejemplo, puedes ajustar n_estimators, max_depth, min_samples_split, etc.
