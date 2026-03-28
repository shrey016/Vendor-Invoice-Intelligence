from data_preprocessing_split_scale import load_db,create_flag_invoice,split_data,scale_data
from train_evaulate_models import train_xgboost_model,get_performance_measure
import joblib
import os


features =['invoice_quantity', 'invoice_dollars', 'Freight','total_qty', 'total_dollars']

target = "flag_invoice"

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    db_path = os.path.join(PROJECT_ROOT, "data", "inventory.db")
    model_dir = os.path.join(PROJECT_ROOT, "models")

    df=load_db(db_path)

    df=create_flag_invoice(df)

    X=df[features]
    y=df[target]

    X_train,X_test,y_train,y_test=split_data(X,y)
    X_train_scaled,X_test_scaled=scale_data(X_train,X_test)

    model=train_xgboost_model(X_train_scaled,y_train)
    get_performance_measure(model,X_test_scaled,y_test,'XGBClassifier')

    model_path=model_dir + '/predict_invoice_flag_model.pkl'
    joblib.dump(model,model_path)

    print("XGB model saved !!\n")
    print(f"Model path: {model_path}")

if __name__ == "__main__":
    main()

    