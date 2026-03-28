from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,root_mean_squared_error,mean_absolute_error

def train_linear_model(X_train,y_train):
    lr_model=LinearRegression()
    lr_model.fit(X_train,y_train)
    return lr_model


def train_dtree_model(X_train,y_train):
    dtree_model=DecisionTreeRegressor(random_state=42)
    dtree_model.fit(X_train,y_train)
    return dtree_model


def train_rf_model(X_train,y_train):
    rf_model=RandomForestRegressor(random_state=42)
    rf_model.fit(X_train,y_train)
    return rf_model


def get_performance_metrics(model,predictor,target,model_name):
    y_pred=model.predict(predictor)
    r2=r2_score(target,y_pred)
    mae=mean_absolute_error(target,y_pred)
    rmse=root_mean_squared_error(target,y_pred)
    print(f"\n{model_name} model results:\n")
    print("R2 score",r2)
    print("Mean absolute error",mae)
    print("Root mean squared error",rmse)
    return r2,mae,rmse
    