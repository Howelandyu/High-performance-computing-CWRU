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

tokens_set = set([])
tokens_list = []

for token_list in appl_mergeDp["Tokens"].to_list():
    token_list = token_list.replace('[','').replace(']','').replace('"','').split(",")
    token_list = [t.strip() for t in token_list if t!='']
    tokens_list.append(token_list)
    for token in token_list:
        token = token.strip()
        if token!='':
            tokens_set.add(token)

print('nodup token count = {0}'.format(len(tokens_set)))

tokens_list[:5]

list(tokens_set)[:200]

"""## 3.2. train word2vec model by tokens"""

# ! pip install gensim==4.2.0

from gensim.models import Word2Vec
from gensim.test.utils import common_texts

vsize = 50

# train word2vec model
model = Word2Vec(sentences=tokens_list, vector_size=vsize, window=5, min_count=1, workers=4)
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")

# test
vector = model.wv['oldies']
vector.shape

"""## 3.3 tokens to vector"""

# max token count

token_len_list = []
for t in tokens_list:
    token_len_list.append(len(t))
    
max_token_count = max(token_len_list)
print(max_token_count)




from tqdm import *

sentence_vector_list = []
for tokens in tqdm(tokens_list):
    sentence_vector = []
    for w in tokens:
        vector = model.wv[w]
        sentence_vector = sentence_vector+list(vector)
    sentence_vector = sentence_vector + [0]*(max_token_count-len(sentence_vector)//vsize)*(vsize)
    sentence_vector_list.append(sentence_vector)    

print(len(sentence_vector_list))

len(sentence_vector)

y=appl_predict_pd[['Price']]
X=appl_predict_pd[['Open','Close','Adj Close','Normalized','Standardized','Score','Volume']]
print(y.shape, X.shape)

import numpy as np

x_data = np.array(X)
y_data = np.array(y)
print('before, x_data.shape = {0}'.format(x_data.shape) )
x_data = np.hstack((x_data, sentence_vector_list))
print('after, x_data.shape = {0}'.format(x_data.shape) )

# Normalization
# ref: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
x_data.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#回归
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
    layers.Dense(64, activation='relu', input_shape=[x_data.shape[1]]),
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

EPOCHS = 10
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