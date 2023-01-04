import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import  *
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

cancer_ds = load_breast_cancer()
# print(cancer_ds)
# print(type(cancer_ds))
#
# print(cancer_ds.feature_names)

# print(cancer_ds.data)
# print("----------------------------------")
# print(cancer_ds.target)

cancer_df = pd.DataFrame(np.c_[cancer_ds.data, cancer_ds.target], columns=[list(cancer_ds.feature_names) + ['target']])
# print(cancer_df)
# print(type(cancer_df))

Y = cancer_df['target']
print(Y.shape)

X = cancer_df.drop(['target'], axis=1)
print(X.shape)

X_train,X_test, Y_train,Y_test = train_test_split(X,Y, test_size=0.25, random_state=1)

model = BernoulliNB()
model.fit(X_train, Y_train)
print(model)

print(model.score(X_test, Y_test))

Y_predict = model.predict(X_test)
print("accuracy score of my model :", accuracy_score(Y_test,Y_predict))


cancerdata = cancer_df.to_csv("cancerdata")

patient1_data = [13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,
            0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]

patient1 = np.array([patient1_data])
result_of_patient1 = model.predict(patient1)

if result_of_patient1[0] == 1:
    print("patient 1 doesnt have cancer ")
else:
    print("patient 1 has cancer")


patient2_data = [17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,
            0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]

patient2 = np.array([patient2_data])
result_of_patient2 = model.predict(patient2)

if result_of_patient2[0] == 1:
    print("patient 2 doesnt have cancer ")
else:
    print("patient 2 has cancer")