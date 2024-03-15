from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.models import load_model
from keras.utils import image_dataset_from_directory
from sklearn.metrics import (
    accuracy_score , classification_report , 
    confusion_matrix  , mean_squared_error)
from GRN import ResNet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


data_train , data_test = image_dataset_from_directory('Model_Picture\dataset' , 
                                                   label_mode='int',
                                                   image_size=(100,100),
                                                   batch_size=64,
                                                   subset='both' , 
                                                   validation_split=0.2,
                                                   seed=1233)

data_train_aug = data_train.map(lambda x , y : (x/255 , y))

train_ds = data_train_aug
test_ds = data_test

ex = train_ds.as_numpy_iterator()
ex = ex.next()
print(ex[1])

Model = ResNet(64)

Model.compile(optimizer=Adam() , 
              loss=SparseCategoricalCrossentropy(),
              metrics=['Accuracy'])
#Model.fit(train_ds , batch_size=32 , epochs=25)
#Model.save('GRN1.tf')

#Evaluation 
Model_load = load_model('GRN1.tf')
test_ds = test_ds.map(lambda x , y : (x/255 , y))
test_ds = test_ds.as_numpy_iterator()
test_ds = test_ds.next()
print(test_ds[1])
hasil = Model_load.predict(test_ds[0])
convert = [np.argmax(h) for h in hasil]

print(f'MSE : {np.round(mean_squared_error(test_ds[1] , convert) , decimals =2)}')
print(f'Acc : {np.round(accuracy_score(test_ds[1] , convert) , decimals = 2)}')
confusion = confusion_matrix(test_ds[1] , convert)
print(confusion)
print(classification_report(test_ds[1] , convert))
d = pd.DataFrame({'True' : test_ds[1] , 'Pred' : convert})
print(d)

def ReadIMG(url = 'Model_Picture\dataset\date\IMG_20230810_125546.jpg'): 
    img = tf.keras.preprocessing.image.load_img(url , target_size=(100,100))
    array = tf.keras.preprocessing.image.img_to_array(img)/255
    array = tf.expand_dims(array,0)
    real = tf.keras.preprocessing.image.load_img(url , target_size=(250,250))
    real = tf.keras.preprocessing.image.img_to_array(real)/255
    return array , real

def converts(val:int): 
    if val == 0 : return 'Date'
    elif val == 1 : return 'kuliah'
    elif val == 2 : return 'sma'
    elif val == 3 : return 'yudo'

img , real = ReadIMG()
title = Model_load.predict(img)
plt.axis('off')
plt.title(converts(np.argmax(title)))
plt.imshow(real)
plt.show()
