from picamera import PiCamera
from time import sleep
from datetime import datetime
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

# function that receives image filename and determins whether it is a bird or a squirrel
def predict_image(image_file_path):
    img = cv2.imread(image_file_path)
    img = cv2.resize(img, (200, 200))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.reshape(img, [-1, 200, 200, 1])
    class_result = model.predict_classes(img)
    prediction = model.predict(img)
    print('image:',image_file_path)
    print('class:', class_result)
    print('prediction:', prediction)
    if class_result == [[0]]:
        return 'bird'
    if class_result == [[1]]:
        return 'squirrel'

# Take Picture
camera = PiCamera()
now = datetime.now()
image_name = '/home/pi/Desktop/GroupProjectCode/' + str(now) + '.jpg'
print('Capturing image now. Saving to ' + image_name)
camera.capture(image_name)

# This calls our classification function with the recently captured image. The returned value could be used to actuate something.
image_animal = predict_image(image_name)
print('It looks like a ' + image_animal)