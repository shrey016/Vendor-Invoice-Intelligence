from data_load_preprocessing import load_db,prepare_features,split_data
from train_performance_model import train_linear_model,train_dtree_model,train_rf_model,get_performance_metrics
import joblib
from pathlib import Path
import os

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    db_path = os.path.join(PROJECT_ROOT, "data", "inventory.db")
    model_dir = os.path.join(PROJECT_ROOT, "models")
    
    df=load_db(db_path)

   

    X,y=prepare_features(df)
    X_train,X_test,y_train,y_test=split_data(X,y)
    lr_model=train_linear_model(X_train,y_train)
    dtree_model=train_dtree_model(X_train,y_train)
    rf_model=train_rf_model(X_train,y_train)

    
    lr_r2,lr_mae,lr_rmse=get_performance_metrics(lr_model,X_test,y_test,'Linear Regression')
    dtree_r2,dtree_mae,dtree_rmse=get_performance_metrics(dtree_model,X_test,y_test,'Decision Tree Regression')
    rf_r2,rf_mae,rf_rmse=get_performance_metrics(rf_model,X_test,y_test,'Random Forest Regression')
    
    model_results={
        "Linear Regression":(lr_model,lr_mae),
        "Decision Tree Regression":(dtree_model,dtree_mae),
        "Random forest Regresion":(rf_model,rf_mae)
    }

    best_model_name,(best_model,best_mae)=min(model_results.items(),key=lambda x:x[1][1])
    model_path=model_dir + '/predict_frieght_cost_model.pkl'
    joblib.dump(best_model,model_path)

    print(f"Best model : {best_model_name} with {best_mae} MAE saved !!\n")
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    main()

