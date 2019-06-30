from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix
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
from sklearn.utils.multiclass import unique_labels


out_folder = './model'
data_folder ='./data'
records = pd.read_csv(data_folder + '/newdataset.csv')
print(records.head())
#missing values
#print(records.isnull().values.sum())
#print(records.isnull().sum())

#records.id = records.Category.astype('category').cat.codes.values

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




# drop colunmns

records.drop(columns='App',axis=1,inplace=True)
records.drop(columns='Rating',axis=1,inplace=True)
records.drop(columns='Size',axis=1,inplace=True)
records.drop(columns='Installs',axis=1,inplace=True)
#records.drop(columns='Android Ver',axis=1,inplace=True)
records.Android = records.Android.astype('category').cat.codes.values
#records.to_csv(r'E:\projectM3\work\data\processed.csv')

#train test set
train, test = train_test_split(records, test_size=0.2, random_state=0)
appid = train['id']
testid = test.id
trainLabel = train.label
testLabel = test.label
#train.to_csv(r'E:\projectM3\final ranking\data\train.csv')
#test.to_csv(r'E:\projectM3\final ranking\data\testanly.csv')
train.drop(columns='id',axis=1,inplace=True)
test.drop(columns='id',axis=1,inplace=True)

train.drop(columns='label',axis=1,inplace=True)
test.drop(columns='label',axis=1,inplace=True)

m =keras.models.load_model('model.h5')
p= m.predict([testid[0:100],test[0:100]])


cm = confusion_matrix(testLabel[0:100],p.round())
print(testLabel.shape)
ap =0
for i in range(0,100):
    if list(testLabel)[i]== 1:
        ap = ap +1
print("number of tottal possitive case",ap)
print(cm)
a=np.array(testLabel)
TP = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[0][0]

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)
print(TPR, "recall")

# Specificity or true negative rate
#TNR = TN / (TN + FP)
#print(TNR, "TNR")

# Precision or positive predictive value
PPV = TP / (TP + FP)
print(PPV, "precision")
# Negative predictive value
#NPV = TN / (TN + FN)
#print(NPV, "NPV")
# Fall out or false positive rate
#FPR = FP / (FP + TN)
#print(FPR, "FPR")
# False negative rate
#FNR = FN / (TP + FN)
#print(FNR, "FNR")
# False discovery rate
#FDR = FP / (TP + FP)
#print(FDR, "FDR")
# Overall accuracy
ACC = (TP + TN) / (TP + FP + FN + TN)
print(ACC, "ACC")

c=0
for i in range(len(a)):
    if(a[i]== 0):
        c=c+1;
print(c)
accuracy = np.trace(cm) / float(np.sum(cm))
misclass = 1 - accuracy
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[y_true, y_pred]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
plot_confusion_matrix(testLabel[0:100],p.round(), classes=[0,1], normalize=False,title='confusion matrix')
plt.show()