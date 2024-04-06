"""# 2. add volume"""
import pandas as pd
# from numba import jit, cuda
# from mpi4py import MPI
import numba
appl_price=pd.read_csv("AAPL_new.csv")
appl_tw=pd.read_csv('data_AAPL.csv',lineterminator='\n')

# comm = MPI.COMM_WORLD 
# rank = comm.Get_rank()
# size = comm.Get_size()

appl_price['Date']=pd.to_datetime(appl_price['Date'],format='%Y.%m.%d')
appl_price.rename(columns={'Unnamed: 0':'ID'},inplace = True)
appl_price.info()
appl_price

appl_tw.rename(columns={'Created_at':'Date','Unnamed: 0':'ID'},inplace = True)
appl_tw['Date']=pd.to_datetime(appl_tw['Date']).dt.date  #.view('int64')#.floor('d').view('int64')
appl_tw['Date']=pd.to_datetime(appl_tw['Date'],format='%Y.%m.%d')
appl_tw.info()
appl_tw

appl_mergeDp=pd.merge(appl_price,appl_tw,left_on='Date', right_on='Date')
#appl_predict_pd=appl_mergeDp.drop(columns=['ID_x','Open','High','Low','Close','Volume','Normalized','Standardized','ID_y','User','Tweet','Tokens'])
#appl_predict_pd
#display(appl_mergeDp)
appl_predict_pd=appl_mergeDp.drop(columns=['ID_x','ID_y','User','Tweet','Tokens','Date','High','Low'])
#appl_predict_pd=appl_mergeDp.drop(columns=['ID_x','ID_y','User','Date'])
appl_predict_pd

breakTime=appl_predict_pd['Open'][0:]
print(type(breakTime))
#breakTime.append(0.0)
print(breakTime)
for i in range(0,breakTime.size-1):
    breakTime.iloc[i]=breakTime.iloc[i+1]
appl_predict_pd['Price']=appl_predict_pd['Open'][0:]
appl_predict_pd


# @numba.jit(parallel=True)
# def get_breaktime(predic_df):
#   breakTime=predic_df
#   # i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
#   display(type(breakTime))
#   #breakTime.append(0.0)
#   display(breakTime)
#   for i in range(0,breakTime.size-1):
#       breakTime.iloc[i]=breakTime.iloc[i+1] 
     
# appl_predict_pd['Price']=get_breaktime(appl_predict_pd['Open'][0:])
# appl_predict_pd

appl_predict_pd['Price']



appl_mergeDp['Price']=breakTime
appl_mergeDp

y=appl_predict_pd[['Price']]
X=appl_predict_pd[['Open','Close','Adj Close','Normalized','Standardized','Score','Volume']]
print(y.shape, X.shape)

X

# Normalization
# ref: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#regression
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

@numba.jit(parallel=True,forceobj = True)
def build_model():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[7]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model
model = build_model()
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: 
            print('')
        print('.', end='')

EPOCHS = 50
model.fit(
  X_train, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=1,
  callbacks=[PrintDot()])

from sklearn.metrics import mean_squared_error

test_predictions = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, test_predictions)
print('mse={0}'.format(mse))
predict_df = pd.DataFrame(test_predictions)

print(predict_df)

y_test