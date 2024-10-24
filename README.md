# EX 10 Implementation of SVM For Spam Mail Detection
## DATE:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. SVM aims to find the optimal hyperplane that maximizes the margin between two classes (spam vs. non-spam).
2. For spam detection, you typically use a dataset of emails with a label (spam or ham/non-spam) and the email text. 
3. SVM model on the training data. For text classification, linear SVM often works well
4. After training the model, we’ll make predictions on the test set and evaluate its performance.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection.
Developed by: 
RegisterNumber:  
*/
```import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("/content/spamEX10.csv",encoding='ISO-8859-1')
df.head()

vectorizer = CountVectorizer()
x=vectorizer.fit_transform(df['v2'])
y=df['v1']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=42)

model = svm.SVC()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print("ACCUARACY:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))

def predict_message(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return prediction[0]

new_message = "Congratulations!"
result = predict_message(new_message)
print(f"The message: '{new_message}' is classified as: {result}")
```
## Output:
![image](https://github.com/user-attachments/assets/c83d7dcf-9ec0-48fa-85d8-43cc388830b6)
![image](https://github.com/user-attachments/assets/02600a18-9f25-4cc5-91f5-e65fd34c3d60)
![image](https://github.com/user-attachments/assets/1bafa057-db2f-4aa0-930b-b4eb9a3fc265)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
