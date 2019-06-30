from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Input, Embedding, Dense,Flatten, concatenate, Reshape,Dropout
from keras.models import Model
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from collections import OrderedDict
import itertools as it
import matplotlib.pyplot as plt
from keras.utils import plot_model

out_folder = './model'
data_folder ='./data'
records = pd.read_csv(data_folder + '/newdataset.csv')
print(records.head())
#missing values
#print(records.isnull().values.sum())
#print(records.isnull().sum())

#size preproces below 20,50,100
records.size_below20=""
records.size_below50=""
records.size_below100=""
records['size_below20'] = ['1' if x <=20 else '0' for x in records['Size']]
records['size_below50'] = ['1' if x <=50 and x>20 else '0' for x in records['Size']]
records['size_below100'] =['1' if  x>50 else '0' for x in records['Size']]
#category one hot encode
records = pd.get_dummies(records, columns=['Category'], prefix = ['C'])
#ratting - above 3, above 4

records.rating_above3=""
records.rating_above4=""
records.rating_anything=""
records['rating_above3'] = ['1' if x >=3 else '0' for x in records['Rating']]
records['rating_above4'] = ['1' if x >=4  else '0' for x in records['Rating']]
records['rating_anything']=['0'for x in records['Rating'] ]
#content --- age group one hot encode
records = pd.get_dummies(records, columns=['Content Rating'], prefix = ['age_group'])


#records.id = records.id.astype('category').cat.codes.values
#print(records.id)

u= len(records.id.unique())

print(u)


# drop colunmns

records.drop(columns='App',axis=1,inplace=True)
records.drop(columns='Rating',axis=1,inplace=True)
records.drop(columns='Size',axis=1,inplace=True)
records.drop(columns='Installs',axis=1,inplace=True)
#records.drop(columns='Android Ver',axis=1,inplace=True)
records.Android = records.Android.astype('category').cat.codes.values
records.to_csv(r'E:\projectM3\work\data\processed.csv')

#train test set
train, test = train_test_split(records, test_size=0.2, random_state=0)
appid = train['id']
testid = test.id
trainLabel = train.label
testLabel = test.label
train.to_csv(r'E:\projectM3\final ranking\data\train.csv')
test.to_csv(r'E:\projectM3\final ranking\data\test.csv')
train.drop(columns='id',axis=1,inplace=True)
test.drop(columns='id',axis=1,inplace=True)

train.drop(columns='label',axis=1,inplace=True)
test.drop(columns='label',axis=1,inplace=True)





##model

main_input = Input(shape=(1,), name='main_input')
x = Embedding(output_dim=20, input_dim=u+2,)(main_input)
flat = Flatten(name='flattelayer')(x)

ax_input = Input(shape=(29,),name='aux_input')
concat =  concatenate([flat,ax_input],name='concat')
#ax_input = Input(shape=(12,),name='aux_input')
drop = Dropout(0.2)(concat)
#dense = Dense(128,activation='relu',name='FullyConnected1')(concat)
dense_2 = Dense(64,activation='relu',name='FullyConnected2')(drop)
dense_3 = Dense(32,activation='relu',name='FullyConnected3')(dense_2)
dense_4 = Dense(18,activation='relu',name='FullyConnected4')(dense_3)
output = Dense(1,activation='sigmoid',name='FullyConnected5')(dense_4)

model = keras.Model([main_input,ax_input] , output)
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

##training
h = model.fit([appid,train],trainLabel,epochs=100,batch_size=32,validation_data=(([testid[:600],test[:600]]),testLabel[:600]))


######save model#####
#model.save('model.h5')
#print(model.predict([testid[301:342],test[301:342]]))
#print(testLabel[301:342])
#print(model.predict([testid,test]))
##test

""""
print(m.predict([appid[0:1],train[0:1]]))

print(trainLabel[0:1])
print(m.predict([appid[1:2],train[1:2]]))
print(trainLabel[1:2])
print(m.predict([appid[2:3],train[2:3]]))
print(trainLabel[2:3])
print(m.predict([appid[4:5],train[4:5]]))
print(trainLabel[4:5])
"""

#a = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,1,1,1,0,0])
#ap= pd.Series()
#ap= a.reshape(1,27)

#print(ap.shape)

#test_loss, test_acc = model.evaluate(([testid,test]),testLabel)
#print(test_loss, test_acc )
history_dict = h.history
history_dict.keys()
[u'acc', u'loss', u'val_acc', u'val_loss']

acc = h.history['acc']
val_acc = h.history['val_acc']
loss = h.history['loss']
val_loss = h.history['val_loss']
epochs = range(1, len(acc) + 1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf()
plt.plot(epochs, acc, 'bo', label='Training accuracy')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

#print(testLabel[0:15])
#print(model.predict([pd.Series(id),ap]))
