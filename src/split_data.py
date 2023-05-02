import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
def SplitData(df):
    numpy_df = np.array(df)
    X = numpy_df[:,:-1]
    y = numpy_df[:,-1]
    ros = RandomOverSampler()
    X,y = ros.fit_resample(X,y)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    one_hot = OneHotEncoder()
    y_train = one_hot.fit_transform(y_train.reshape(-1,1)).toarray()
    y_test  = one_hot.transform(y_test.reshape(-1,1)).toarray()

    X_train = torch.tensor(X_train,dtype=torch.float32)
    X_test  = torch.tensor(X_test,dtype=torch.float32)
    y_train = torch.tensor(y_train,dtype=torch.float32)
    y_test  = torch.tensor(y_test,dtype=torch.float32)

    return X_train,X_test,y_train,y_test


