from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,confusion_matrix,classification_report
import numpy as np


def train_xgboost_model(X_train_scaled,y_train):
    model=XGBClassifier(random_state=42,eval_metric='logloss')
    parameters = {
        "max_depth": np.arange(4,6,1),
        "n_estimators": np.arange(150,300,50),
        "subsample":[0.5,0.7,0.9,1],
        "learning_rate":[0.01,0.1,0.2,0.05],
        "gamma":[0,1,3],
        "colsample_bytree":[0.5,0.7,0.9,1],
        "colsample_bylevel":[0.5,0.7,0.9,1]
    }
    random_obj=RandomizedSearchCV(model,parameters,scoring='recall',cv=5,n_iter=200,n_jobs=-1,verbose=2)
    random_obj.fit(X_train_scaled,y_train)
    
    xgb_tuned=random_obj.best_estimator_
    xgb_tuned.fit(X_train_scaled,y_train)
    return xgb_tuned

def get_performance_measure(model,predictor,target,model_name):
    y_pred=model.predict(predictor)
    accuracy=accuracy_score(target,y_pred)
    recall=recall_score(target,y_pred)
    f1=f1_score(target,y_pred)
    precision=precision_score(target,y_pred)
    print(f"\n{model_name}")
    #df_score=pd.DataFrame({'Accuracy':accuracy,'Recall':recall,'F1_score':f1,'Precision':precision},index=[0])
    #display(df_score)
    print(f"Accuracy score : {accuracy}")
    report=classification_report(target,y_pred)
    print(f"Classification report: \n{report}")
    