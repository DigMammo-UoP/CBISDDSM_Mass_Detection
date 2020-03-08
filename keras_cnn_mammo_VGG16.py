from keras import backend as K
from keras import regularizers
from keras.models import Model
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import *
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from keras.applications import vgg16
from sklearn import *


ROWS=224
COLS=224
target = (ROWS,COLS)

#load model
input_shape = (ROWS,COLS,3)
vgg=vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
vgg_model=Model(vgg.input, output=vgg.layers[-1].output)

#freeze all but last layers
# 0-15: train only the last convolutional layer (block5_conv3, conv2, conv1) 
for layer in vgg_model.layers[:15]:
	layer.trainable = False
vgg_model.summary()


# This has the paths of the database images
train_path = '/Research/Mammography/CBIS-DDSM/MYROIS/TRAIN/'
test_path = '/Research/Mammography/CBIS-DDSM/MYROIS/TEST/'

# TRAIN / TEST GENERATORS
train_datagen = ImageDataGenerator(
#	featurewise_center=True,
	samplewise_center = True,
	samplewise_std_normalization = True,
	shear_range=0.5,
	rotation_range= 40,
	width_shift_range=0.25,
	height_shift_range=0.25,
	zoom_range= [0.5,1.5],
	horizontal_flip=True,	
	fill_mode='nearest',
#	preprocessing_function=preprocess_input
)
train_datagen.mean = [103.939, 116.779, 123.68]  #this is the global mean of the training set calculated by MATLAB

test_datagen = ImageDataGenerator(
#	preprocessing_function=preprocess_input,
	samplewise_center = True,
	samplewise_std_normalization = True,
#	featurewise_center=True
)
test_datagen.mean = [103.939, 116.779, 123.68]  #this is the global mean of the training set calculated by MATLAB

# This makes the batches for training / validating / testing
# The validation images are taken from the train images in a split using subset attribute
train_batches = train_datagen.flow_from_directory(train_path, target_size = target, batch_size=32, class_mode='categorical',shuffle=True)
valid_batches = test_datagen.flow_from_directory(test_path, target_size = target, batch_size=32, class_mode='categorical',shuffle=True)
test_batches = test_datagen.flow_from_directory(test_path, target_size = target, batch_size=1, shuffle=False, class_mode='categorical')


# Model

model = Sequential()
model.add(vgg_model)
model.add(Flatten())


model.add(Dense(4096,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024,activation='relu', kernel_regularizer=regularizers.l2(1)))
#model.add(Dense(256,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.summary()

# Optimization - Fillting
optimizer = Adam(lr=1e-4)
model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = ['acc'])

STEP_SIZE_TRAIN=train_batches.samples//train_batches.batch_size
STEP_SIZE_VALID=valid_batches.samples//valid_batches.batch_size

#this is to save the results
file_path='weights.best.vgg16.hdf5'

# Learning Rate Reducer
learn_control = ReduceLROnPlateau(monitor='val_acc', patience=10,verbose=1,factor=0.1, min_lr=1e-7)
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor='val_acc', mode='max', patience=20)
callbacks_list = [learn_control, checkpoint, early]

history = model.fit_generator(train_batches, steps_per_epoch = STEP_SIZE_TRAIN, validation_data = valid_batches, validation_steps=STEP_SIZE_VALID, shuffle=True, callbacks=callbacks_list, epochs=150, verbose=1)
#history = model.fit_generator(train_batches, steps_per_epoch = STEP_SIZE_TRAIN, validation_data = valid_batches, validation_steps=STEP_SIZE_VALID, shuffle=True, epochs=200, verbose=1)

#Check testing using the best weights from the file
model.load_weights(file_path)

result=model.evaluate_generator(generator=test_batches,steps=1)
print('Loss: ',result[0])
print('Accuracy: ',result[1])

test_batches.reset()
pred = model.predict_generator(test_batches, verbose=True, workers=2)

predicted_class_indices=np.argmax(pred,axis=1)
labels = (test_batches.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

true_classes = test_batches.classes
class_labels = list(test_batches.class_indices.keys())
report = metrics.classification_report(true_classes, predicted_class_indices, target_names=class_labels)
print(report) 
cmatrix = metrics.confusion_matrix(true_classes, predicted_class_indices)
print(cmatrix)


#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Validation'], loc='upper left')
#plt.show()

##plot training & validation loss values
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Validation'], loc='upper left')
#plt.show()

