import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from src.data_loader import load_datasets

categ_cols = ["protocol_type", "service", "flag"]
label_cols = ["label", "binary_label", "attack_cat", "difficulty"]
cont_cols = []

def preprocess(train_df, test_df):
    cont_cols = [col for col in train_df.columns if col not in categ_cols + label_cols]
    train, val = train_test_split(train_df, test_size=0.2, random_state=42)
    
    X_train = train[categ_cols + cont_cols]
    X_val = val[categ_cols + cont_cols]
    X_test = test_df[categ_cols + cont_cols]
    
    y_train_bin = train["binary_label"]
    y_val_bin = val["binary_label"]
    y_test_bin = test_df["binary_label"]
    
    y_train_cat = train["attack_cat"]
    y_val_cat = val["attack_cat"]
    y_test_cat = test_df["attack_cat"]
    
    preprocessor = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categ_cols),
                                                   ("num", StandardScaler(), cont_cols)])
    
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)
    
    print("Feature dim: ", X_train.shape)
    print("X_train size: ", X_train.shape[0])
    print("X_val size: ", X_val.shape[0])
    print("X_test size: ", X_test.shape[0])
    
    return X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, y_train_cat, y_val_cat, y_test_cat, preprocessor
   
   
if __name__ == "__main__":    
    train_df, test_df = load_datasets("./data/NSL-KDD/KDDTrain+.txt", "./data/NSL-KDD/KDDTest+.txt") 
    (X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, y_train_cat, y_val_cat, y_test_cat, preprocessor) = preprocess(train_df, test_df)