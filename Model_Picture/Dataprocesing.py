import tensorflow as tf 

def Augmented_img(img): 
    layer = [
        tf.keras.layers.RandomRotation(factor=0.15) , 
        tf.keras.layers.RandomFlip(seed=123)
    ]
    for l in layer : 
        imgs = l(img)
    return imgs

data_train = tf.keras.utils.image_dataset_from_directory('Model_Picture\dataset' , 
                                                   label_mode='categorical',
                                                   image_size=(64,64),
                                                   batch_size=64)

data_train_aug = data_train.map(lambda x , y : (x/255 , y))

train_ds = data_train_aug

ex = train_ds.as_numpy_iterator()
ex = ex.next()