from keras.models import load_model
from keras.models import model_from_json
from keras.metrics import Recall
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import preprocess_input
from keras.optimizers import RMSprop
from sklearn.metrics import classification_report
test_dir = '../test/'
# load json and create model
json_file = open('../Xray_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("../Xray_best_model")

rmsprop = RMSprop(lr = 0.0001)
model.compile(loss='categorical_crossentropy', metrics=[Recall(class_id=2)], optimizer=rmsprop)
#print(loaded_model.summary())

#Test Generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(test_dir, 
                                                 target_size=(200,150),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=False)

STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
#loss, acc=model.evaluate_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)

Y_pred = model.predict_generator(test_generator, STEP_SIZE_TEST)
y_pred = np.argmax(Y_pred, axis=1)
report_matrix = classification_report(test_generator.classes, y_pred)



