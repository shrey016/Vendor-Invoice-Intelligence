import joblib
import pandas as pd

model_path='/Users/shrey/Great_learning/frieght_cost_classficaion_project/Inventory-Invoice-Analytics/models/predict_frieght_cost_model.pkl'

def load_model(model_path):
    with open(model_path,'rb') as f:
        model=joblib.load(f)
    return model
    

def predict_frieght_cost(input_data):
    model=load_model(model_path)
    df=pd.DataFrame(input_data)
    df['Predicted Frieght']=model.predict(df).round()
    return df


if __name__=="__main__":
    sample_data={
        'Dollars':[18200,900]
    }
    frieght_cost=predict_frieght_cost(sample_data)
    print(frieght_cost)