from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

def build_model(input_shape, num_classes):
    ''' This method builds a very basic cnn image classifier which was inspired by many tutorials 
        one of which we like is Anuj Shah's tutorial: https://www.youtube.com/watch?v=yDVap0lpYKg  
    '''
    model = Sequential()
    #First layer
    model.add(Conv2D(32, (3,3),input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    #second layer
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    #third layer
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    #Softmax layer
    model.add(Activation('softmax'))
    
    #compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])
    return model