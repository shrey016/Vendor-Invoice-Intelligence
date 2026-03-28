import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler


def load_db(db_path):
    conn=sqlite3.connect(db_path)
    query="""WITH purchase_agg AS
    (SELECT PONumber,COUNT(DISTINCT(Brand)) AS total_brands,
    SUM(QUANTITY) AS total_qty,
    SUM(Dollars) AS  total_dollars,
    round(AVG(JULIANDAY(ReceivingDate) - JULIANDAY(PODate)),2) as avg_receiving_delay
    from purchases 
    group by PONumber)
    
    SELECT
    vi.PONumber,
    vi.Quantity as invoice_quantity,
    vi.Dollars as invoice_dollars,
    vi.Freight,
    (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
    (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
    pa.total_brands,
    pa.total_qty,
    pa.total_dollars,
    pa.avg_receiving_delay
    from vendor_invoice vi LEFT JOIN purchase_agg pa ON vi.PONumber=pa.PONumber
    """
    df=pd.read_sql_query(query,conn)
    conn.close()
    return df
    
def create_invoice_risk_label(row):
    if(abs(row['invoice_dollars']-row['total_dollars'])>5):
        return 1
    if(row['avg_receiving_delay']>10):
        return 1
    else:
        return 0
        
def create_flag_invoice(df):
    df['flag_invoice']=df.apply(create_invoice_risk_label,axis=1)
    return df

def split_data(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test

def scale_data(X_train,X_test):
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return X_train_scaled,X_test_scaled