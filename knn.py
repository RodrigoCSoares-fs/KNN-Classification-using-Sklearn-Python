# -*- coding: utf-8 -*-


from sklearn.datasets import load_iris
iris = load_iris()

caracteristicas = iris.data
print(caracteristicas)
caracteristicas.dtype

caracteristicas.shape

classes = iris.target
print(classes)

print(classes.shape)

import pandas as pd
caracteristicas_df = pd.DataFrame(caracteristicas, columns = ["at_sepal", "larg_sepal", "at_petal", "larg_petal"])
caracteristicas_df.describe()

classes_df = pd.DataFrame(classes, columns = ["classes"])
classes_df

caracteristicas_df.info()

data_df = pd.concat([caracteristicas_df, classes_df], axis=1)
print(data_df)

import seaborn as sns

sns.scatterplot(data=data_df,x="at_sepal", y="larg_sepal", hue= classes)

sns.set_theme(style="ticks")
sns.pairplot(data_df, hue="classes")

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
scaler.fit(caracteristicas_df)
caracteristicas_norm = scaler.transform(caracteristicas_df)
print(caracteristicas_norm)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(caracteristicas_norm, classes, test_size=0.2)

print(y_train)
print(y_train.shape)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print(y_test)

print(pred)

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average='macro')
prec = precision_score(y_test, pred, average='macro')
recall = recall_score(y_test, pred, average='macro')

print("Acuracia: ", str(acc))
print("Precisao: ", str(prec))
print("F1: ", str(f1))
print("Recall: ", str(recall))