

import numpy as np
import pandas as pd
import warnings
from joblib import dump
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import pathlib
warnings.filterwarnings("ignore", category=FutureWarning)



Diabetes_df=pd.read_csv('data/diabetes_prediction_dataset.csv')
Diabetes_df.head()


print(Diabetes_df.describe())
print("Valores únicos en gender:", Diabetes_df['gender'].unique())
print("Valores únicos en smoking_history:", Diabetes_df['smoking_history'].unique())




gender_encode = {
    'Male': 0,
    'Female': 1,
    'Other': 2   
}

smoking_encode = {
    'never': 0,
    'not current': 1,
    'former': 2,
    'current': 3,
    'ever': 4,
    'No Info': 5
}
Diabetes_df['gender']=Diabetes_df['gender'].map(gender_encode)
Diabetes_df['smoking_history']=Diabetes_df['smoking_history'].map(smoking_encode)
print(Diabetes_df.head())


#dividir el dataframe en x e y
X=Diabetes_df.drop('diabetes', axis=1)
y=Diabetes_df['diabetes']

X_train, X_test,y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)
print(f'Training Data Shape: X_train {X_train.shape}, y_train{y_train.shape}')
print(f'Testing Data Shape: X_test {X_test.shape}, y_test {y_test.shape}')



#oversampling o como se llame -- aqui empienza


train_df=X_train.copy()
train_df['diabetes']=y_train

diabetes_majority=train_df[train_df['diabetes']==0]
diabetes_minority=train_df[train_df['diabetes']==1]

diabetes_oversample_minority=resample(
    diabetes_minority,
    replace=True,
    n_samples=len(diabetes_majority),
    random_state=42
)
train_balanced=pd.concat([diabetes_majority,diabetes_oversample_minority])
train_balanced=train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X_train=train_balanced.drop(columns='diabetes',axis=1)
y_train=train_balanced['diabetes']
y_train.value_counts()



#aqui acaba--

#entrenamiento: 



Rf=RandomForestClassifier().fit(X_train, y_train)
eval_train_rf=Rf.predict(X_train)
eval_test_rf=Rf.predict(X_test)

print(X_test)
print(y_test)

acc_rf=accuracy_score(y_test, eval_test_rf)
precision_rf=precision_score(y_test, eval_test_rf)
recall_rf=recall_score(y_test, eval_test_rf)
f1_rf=f1_score(y_test, eval_test_rf)




result_df=({
    "Accuracy":[acc_rf],
    "Precisious":[precision_rf],
    "Recall":[recall_rf],
    "F1 Score":[f1_rf]
})
result_df=pd.DataFrame(result_df, index=['Random Forest'])
print(result_df)

dump(Rf, pathlib.Path('model/diabetesv1.joblib'))







