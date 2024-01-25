import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar el dataset (asegúrate de tener el nombre correcto del archivo CSV)
df = pd.read_csv('nombre_del_archivo.csv')

# Eliminar la columna 'categoria_edad'
df = df.drop('categoria_edad', axis=1)

# Visualizar la distribución de clases
plt.figure(figsize=(8, 6))
df['DEATH_EVENT'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribución de Clases')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.xticks([0, 1], ['No Fallecido', 'Fallecido'], rotation=0)
plt.show()

# Separar las características (X) y la variable objetivo (y)
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Partición estratificada en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Ajustar un árbol de decisión
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular el accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del modelo: {accuracy}')
