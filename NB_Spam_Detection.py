import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import  *
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# data = ['This is the first document',
#         'This document is the second document '
#         'And this is the third document',
#         'is this the first document']
#
# vec = CountVectorizer()
# res = vec.fit_transform(data)
# print(res)
# print(res.toarray())
# print(vec.get_feature_names_out())

spam_data = pd.read_csv(r"C:\Users\Personal\Desktop\New folder\spam_ham_dataset.csv")
#print(spam_data)

#print(spam_data.groupby('Category').describe())
category_new = pd.get_dummies(spam_data['Category'], drop_first=True)
#print(category_new)

spam_data = pd.concat([spam_data, category_new], axis=1)
spam_data = spam_data.drop(['Category'], axis=1)
spam_data = spam_data.rename(columns={'spam':'Category'})
print(spam_data)

Y = spam_data['Spam']
X = spam_data['Message']

# print(Y)
# print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=1)

vecotrize = CountVectorizer()
X_train_count = vecotrize.fit_transform(X_train.values)
res = X_train_count.toarray()
#print(res)

model = MultinomialNB()
model = model.fit(res, Y_train)

print(model)


print(model.score(X_train_count, Y_train))

checkemail = ["Hello Guys, How are you doing today",
              "get 20% discount on all the sales and advertisement"]

emails_count = vecotrize.transform(checkemail)
prediction = model.predict(emails_count)
print(prediction)


if prediction[1] == 1:
    print("spam email ")
else:
    print("not spam")