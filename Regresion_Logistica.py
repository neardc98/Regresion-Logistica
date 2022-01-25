from numpy.core import fromnumeric
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns

heart = pd.read_csv('heart.csv')  # llamar al archivo CSV
print(heart.shape)  # valor
print(heart.columns)  # columnas

# y es quien va a predecir si es que tiene o no heart
#  x (conjunto de etiquetas)

caracteristicas = ['age', 'sex', 'cp',
                   'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
x = heart[caracteristicas]
y = heart.target

# dividir la información para aprendizaje y validacion del modelo
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.27, random_state=0)

# modelo de prediccion
log_regresion = LogisticRegression()  # usamos el metodo
log_regresion.fit(x_train, y_train)  # entranamos

y_predic = log_regresion.predict(x_test)
print(y_predic)

# hacer la matriz de confusion
matriz_confusion = metrics.confusion_matrix(y_test, y_predic)

# hacer grafico
clase_nombre = [0, 1]
figura, axisas = plt.subplots()
ticks_marker = np.arange(len(clase_nombre))
plt.xticks(ticks_marker, clase_nombre)
plt.yticks(ticks_marker, clase_nombre)

sns.heatmap(pd.DataFrame(matriz_confusion),
            annot=True, cmap="Blues_r", fmt='g')
axisas.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Matriz de Confusión', y=1.1)
plt.ylabel("Etiqueta de actual")
plt.xlabel("Etiqueta de predicción")
plt.show()
print("Exactitud =>", metrics.accuracy_score(y_test, y_predic))
