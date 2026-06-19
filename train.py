import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 
import joblib 
from sklearn.datasets import load_breast_cancer 

data_c = load_breast_cancer() 
df = pd.DataFrame(data_c.data , columns=data_c.feature_names)

x = df.values
y = data_c.target

X_train , X_test, Y_train ,Y_test = train_test_split (
    x,y,test_size = 0.20,random_state = 82
)

model = RandomForestClassifier(
    max_depth = 6, criterion = 'gini' ,min_samples_split = 5,n_estimators =100 ,random_state = 32
)

model.fit(X_train , Y_train)
prediection = model .predict(X_test)

joblib.dump(model , "breastCancerPrediction.joblib")

acc = accuracy_score(Y_test , prediection)

print(acc)