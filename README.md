# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.

2.Import the dataset to operate on.

3.Split the dataset.

4.Predict the required output.

5.End the program. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..

Developed by: S.LOKESH SAI DILEEP

RegisterNumber: 212221230111 
*/
```
import pandas as pd

data=pd.read_csv("spam.csv",encoding='Windows-1252')

import chardet

file='spam.csv'

with open(file, 'rb') as rawdata:

    result = chardet.detect(rawdata.read(10000))
    
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 

cv=CountVectorizer()

x_train=cv.fit_transform(x_train)

x_test=cv.transform(x_test)

from sklearn.svm import SVC

svc=SVC()

svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)

y_pred

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy

## Output:
![image](https://user-images.githubusercontent.com/94883079/204546230-0a003d20-eaa5-4176-b5f7-d83a8671bbad.png)
![image](https://user-images.githubusercontent.com/94883079/204546340-f4ba8079-4ca8-4ae9-b5b6-8f00d4857f48.png)
![image](https://user-images.githubusercontent.com/94883079/204546501-eb51b07e-ab32-435a-98fc-6b6cf781e714.png)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
