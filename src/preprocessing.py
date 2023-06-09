from typing import Tuple
import itertools
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
        
    all_values = working_train_df.select_dtypes(include=['object']).nunique().index.tolist()
    ohe_values = []
    ordinal_values = []
    for value in all_values:
        if working_train_df[value].nunique() < 3:
            ordinal_values.append(value)
        else:
            ohe_values.append(value)
    
    ohe = OneHotEncoder(drop=None)
    ohe_df = ohe.fit_transform(working_train_df[ohe_values]).toarray()
    ohe_df_2 = ohe.transform(working_test_df[ohe_values]).toarray()
    ohe_df_3 = ohe.transform(working_val_df[ohe_values]).toarray()
    
    oe = OrdinalEncoder()
    ordinal_df = oe.fit_transform(working_train_df[ordinal_values])
    ordinal_df_2 = oe.transform(working_test_df[ordinal_values])
    ordinal_df_3 = oe.transform(working_val_df[ordinal_values])
    
    working_train_df= working_train_df.drop(columns=ordinal_values+ohe_values).to_numpy()
    working_test_df= working_test_df.drop(columns=ordinal_values+ohe_values).to_numpy()
    working_val_df= working_val_df.drop(columns=ordinal_values+ohe_values).to_numpy()
    
    working_train_df= np.concatenate((working_train_df, ordinal_df, ohe_df),axis=1)
    working_test_df= np.concatenate((working_test_df, ordinal_df_2, ohe_df_2),axis=1)
    working_val_df= np.concatenate((working_val_df, ordinal_df_3, ohe_df_3),axis=1)
    
    imputer=SimpleImputer()
    working_train_df=imputer.fit_transform(working_train_df)
    working_test_df=imputer.transform(working_test_df)
    working_val_df=imputer.transform(working_val_df)
    
    scaler = MinMaxScaler()
    working_train_df= scaler.fit_transform(working_train_df)
    working_test_df= scaler.transform(working_test_df)
    working_val_df= scaler.transform(working_val_df)
    
    return working_train_df, working_val_df, working_test_df

