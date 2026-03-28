import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split



def load_db(db_path):
    conn=sqlite3.connect(db_path)
    vendor_df=pd.read_sql_query("Select * from vendor_invoice",conn)
    conn.close()
    return vendor_df

def prepare_features(df):
    X=df[['Dollars']]
    y=df['Freight']
    return X,y

def split_data(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test
    