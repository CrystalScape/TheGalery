#from keras.backend import clear_session
from keras.layers import (Conv2D , Dense , Flatten , 
                          BatchNormalization , AveragePooling2D, 
                          Layer , ReLU , Softmax , Dropout)
from keras.layers import add
from keras import Model
from keras.regularizers import l2
from tensorflow import (math , reduce_mean , 
                        convert_to_tensor , random)
import numpy as np
    
class Residual_Block(Layer): 
    
    def __init__(self, units , reg = 0.0001, epsilon = 2e-1 , momentum = 0.9 , 
                stride = 1 , reduction = False , **kwargs):
        super(Residual_Block , self).__init__(**kwargs)
        self.batch1 = BatchNormalization(epsilon=epsilon , momentum=momentum)
        self.batch2 = BatchNormalization(epsilon=epsilon , momentum=momentum)
        self.batch3 = BatchNormalization(epsilon=epsilon , momentum=momentum)
        self.conv1 = Conv2D(units , (1,1) , stride , padding='same' , 
                            use_bias=False , kernel_regularizer=l2(reg))
        self.conv2 = Conv2D(units , (3,3) , stride , padding='same' , 
                            use_bias=False , kernel_regularizer=l2(reg))
        self.conv3 = Conv2D(units , (1,1) , stride , padding='same' , 
                            use_bias=False , kernel_regularizer=l2(reg))
        self.shortcut = Conv2D(units , (1,1) , stride , padding='same' , 
                            use_bias=False , kernel_regularizer=l2(reg))
        self.rel = ReLU()
        self.reducs = reduction
        
    def call(self, inputs):
        actually = inputs 
        # block 1
        b1 = self.batch1(inputs)
        act1 = self.rel(b1)
        c1 = self.conv1(act1)
        # block 2
        b2 = self.batch2(c1)
        act2 = self.rel(b2)
        c2 = self.conv2(act2)
        # block 3 
        b3 = self.batch3(c2)
        act3 = self.rel(b3)
        c3 = self.conv2(act3)
        if self.reducs : 
            actually = self.shortcut(act1)
        else : 
            actually = actually
        x = add([c3 , actually])
        return x 
    
class ResNet(Model): 
    
    def __init__(self, units:tuple , reg = 0.0001 , epsilon = 2e-1 , moment = 0.9 , 
                 filters = (32 , 64 , 128), stage = (2,3,4) ,
                 lable_num = 4  , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense = Dense(units , activation='relu',
                           kernel_regularizer=l2(reg))
        self.batch1 = BatchNormalization(momentum=moment , epsilon=epsilon)
        self.batch2 = BatchNormalization(momentum=moment , epsilon=epsilon)
        self.avrg= AveragePooling2D((8,8))
        self.drop = Dropout(0.5)
        self.flat = Flatten()
        self.out = Dense(lable_num , 
                         kernel_regularizer=l2(reg))
        self.soft = Softmax()
        self.rel = ReLU()
        self.Block1 = [Residual_Block(f , reduction=True) for f in filters]
        self.Block2 = [Residual_Block(f) for f in filters]
        self.stage = stage
        
    def call(self, inputs):
        x = self.batch1(inputs)
        for i in range(len(self.stage)):
            x = self.Block1[i](x)
            for _ in range(self.stage[i]):
                x = self.Block2[i](x)
        x = self.batch2(x)
        x = self.rel(x)
        x = self.avrg(x)
        x = self.flat(x)
        x = self.dense(x)
        x = self.drop(x)
        x = self.out(x)
        x = self.soft(x)
        return x 
                
    
dumm = convert_to_tensor(
    np.random.randint(0 , 255 , 
    size = (1 , 100 , 100 , 3)) / 255
)
        