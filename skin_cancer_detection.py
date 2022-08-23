import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense , MaxPooling2D,Dropout,BatchNormalization
from keras.models import Sequential

label={
    ' Actinic keratoses':0,
    'Basal cell carcinoma':1,
    'Benign keratosis-like lesions':2,
    'Dermatofibroma':3,
    'Melanocytic nevi':4,
    'Melanoma':6,
    'Vascular lesions':5
}
def output():
    model=Sequential()

    model.add(Conv2D(64,(2,2),input_shape=(28,28,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(512,(2,2),input_shape=(28,28,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(Conv2D(1024,(2,2),input_shape=(28,28,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(Conv2D(1024,(1,1),input_shape=(28,28,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(BatchNormalization())
    #
    model.add(Dropout(0.3))
    model.add(Conv2D(1024,(1,1),input_shape=(28,28,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(BatchNormalization())

    #
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))


    model.add(Dense(7,activation='softmax'))

    return model




# model.load_weights("modelv1.h5")