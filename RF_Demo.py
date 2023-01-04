import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import  *
import seaborn as sns

penguins_ds = sns.load_dataset('penguins')
#print(penguins_ds)
# print(penguins_ds.info())
# print(type(penguins_ds))

#print(penguins_ds.isnull().sum())
penguins_ds.dropna(inplace=True)
#print(penguins_ds.isnull().sum())

gender = pd.get_dummies(penguins_ds['sex'], drop_first=True)
#print(gender)

island_new = pd.get_dummies(penguins_ds['island'], drop_first=True)
#print(island_new)


penguins_ds = pd.concat([penguins_ds,gender, island_new], axis=1)
#print(penguins_ds)

penguins_ds.drop(['island', 'sex' ], axis=1, inplace=True)
print(penguins_ds.info())

Y = penguins_ds['species']
#print(Y)
X = penguins_ds.drop(['species'], axis=1)
print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)


model = RandomForestClassifier(n_estimators=4, criterion='entropy', random_state=1)
#print(dir(RandomForestClassifier))

model.fit(X_train, Y_train)

Y_predict = model.predict(X_test)
print(Y_predict)

score = accuracy_score(Y_test, Y_predict)
print(score)


