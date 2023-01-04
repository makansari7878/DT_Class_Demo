
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import  tree

comp_data = pd.read_csv(r"C:\Users\Personal\Desktop\New folder\companies_degree.csv")
#print(comp_data)

Y = comp_data['salary_more_then_100k']
X = comp_data.drop('salary_more_then_100k' , axis=1)
# print(Y)
# print(X)

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

X['company_new'] = le_company.fit_transform(comp_data['company'])
X['job_new']  = le_job.fit_transform(comp_data['job'])
X['degree_new'] = le_degree.fit_transform(comp_data['degree'])

#print(X)

X_new = X.drop(['company', 'job', 'degree'], axis=1)
print(X_new)

model = tree.DecisionTreeClassifier()
model.fit(X_new, Y)
print(model)

print(dir(model))

print(model.score(X_new, Y))

print(model.predict([[2,2,0]]))





