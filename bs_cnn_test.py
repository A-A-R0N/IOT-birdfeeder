from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

model = load_model('bird_squirrel.hdf5')

model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])


def predict_image(image_file_path):
    img = cv2.imread(image_file_path)
    img = cv2.resize(img, (200, 200))
    # cv2.imshow('Original Image', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray Image',img)
    img = np.reshape(img, [-1, 200, 200, 1])
    class_result = model.predict_classes(img)
    prediction = model.predict(img)
    print('image:',image_file_path)
    print('class:', class_result)
    print('prediction:', prediction)
    if class_result == [[0]]:
        print('bird')
    if class_result == [[1]]:
        print('squirrel')

model.summary()
model.layers[0].get_weights()

print('Bird Prediction Examples:')
predict_image('Dataset/validation/Bird/e8e80cdf72a9db06.jpg')
predict_image('Dataset/validation/Bird/e8286561e708febc.jpg')
predict_image('Dataset/validation/Bird/fdc88afb6f6fa1bd.jpg')
predict_image('Dataset/validation/Bird/f9699ba8aeea3b5e.jpg')
predict_image('Dataset/validation/Bird/e42348412dba9ebc.jpg')
predict_image('Dataset/validation/Bird/eed4c401a6420501.jpg')
predict_image('Dataset/validation/Bird/e63df715e0d35a10.jpg')

print('Squirrel Prediction Examples:')
predict_image('Dataset/validation/Squirrel/897c809b02010f67.jpg')
predict_image('Dataset/validation/Squirrel/1b9e69c8308f0a50.jpg')
predict_image('Dataset/validation/Squirrel/7b990a6edf11def6.jpg')
predict_image('Dataset/validation/Squirrel/56e0708d6bc71e66.jpg')
predict_image('Dataset/validation/Squirrel/78642e1159ffdca9.jpg')
predict_image('Dataset/validation/Squirrel/e64e492bec7500bb.jpg')

