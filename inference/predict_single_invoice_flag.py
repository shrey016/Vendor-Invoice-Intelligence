import joblib
import pandas as pd

model_path='/Users/shrey/Great_learning/frieght_cost_classficaion_project/Inventory-Invoice-Analytics/models/predict_invoice_flag_model.pkl'
scaler_path='/Users/shrey/Great_learning/frieght_cost_classficaion_project/Inventory-Invoice-Analytics/models/scaler.pkl'

def load_model(model_path,scaler_path):
    with open(model_path,'rb') as f:
        model=joblib.load(f)
    with open(scaler_path,'rb') as a:
        scaler=joblib.load(a)
    return model,scaler

def predict_invoice_flag(input_data):
    model,scaler=load_model(model_path,scaler_path)
    df=pd.DataFrame(input_data)
    df_scaled=scaler.transform(df)
    df['invoice_flag']=model.predict(df_scaled)
    return df
    
if __name__=="__main__":
    sample_data = {'invoice_quantity': [50,34773],
                 'invoice_dollars': [352.95, 225706.96],
                 'Freight': [1.73, 1196.25],
                 'total_item_quantity': [162, 34773],
                 'total_item_dollars': [2476, 225706]}

    prediction=predict_invoice_flag(sample_data)
    print(prediction)