import os
import tensorflow as tf
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.metrics import Recall

train_dir = '../train/'
test_dir = '../test/'
val_dir = '../Validation/'

#Train Generator
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

#Test Generator
test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

#Validation Generator
val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory(train_dir, 
                                                 target_size=(200,150),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

test_generator=test_datagen.flow_from_directory(test_dir, 
                                                 target_size=(200,150),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)
validation_generator=val_datagen.flow_from_directory(val_dir, 
                                                 target_size=(200,150),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)


conv_base=VGG19( weights='imagenet',include_top=False,input_shape=(200,150,3))

for layer in base_model.layers:
    layer.trainable=False
model = tf.keras.models.Sequential()

# add convolutional base as layer 
model.add(conv_base)
# flatting layer
model.add(tf.keras.layers.GlobalAveragePooling2D())
# dense layers
model.add(tf.keras.layers.BatchNormalization(axis=1))
model.add(tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)))
model.add(tf.keras.layers.BatchNormalization(axis=1))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(.5))
model.add(tf.keras.layers.BatchNormalization(axis=1))
model.add(tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)))
model.add(tf.keras.layers.Dense(3,activation="softmax"))


for layer in model.layers[17:]:
    layer.trainable=True

def print_layers(model):
    for idx, layer in enumerate(model.layers):
        print("layer {}: {}, trainable: {}".format(idx, layer.name, layer.trainable))

print_layers(model)
rmsprop = RMSprop(lr = 0.0001)
model.compile(loss='categorical_crossentropy', metrics=[Recall(class_id=2),'accuracy'], 
	optimizer=rmsprop)

model_chkpoint = ModelCheckpoint(filepath='Xray_best_model', save_best_only=True, save_weights_only=True)

step_size_train=train_generator.n//train_generator.batch_size
step_size_val=validation_generator.n//validation_generator.batch_size

model.fit_generator(train_generator, epochs=20, steps_per_epoch=step_size_train, 
                    callbacks=[model_chkpoint], validation_data=validation_generator,
                    validation_steps=step_size_val)

model_json = model.to_json()
with open("Xray_model.json", "w") as json_file:
    json_file.write(model_json)

