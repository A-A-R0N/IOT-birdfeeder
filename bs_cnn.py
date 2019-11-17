import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, rmsprop

#Variable Declaration
batch_size = 50
training_img_nmb = 3620
test_img_nmb = 180
image_size = (200, 200)
input_size = image_size + (1,)
epochs = 500

#Model Definition
model = Sequential()
model.add(Conv2D(32, (7, 7), input_shape=input_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (7, 7)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(128, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
# model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])


# File Paths
training_dir = os.getcwd() + "./Dataset/train"
test_dir = os.getcwd() + "./Dataset/test"


# Augmenting Training Data
training_data_generator = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# Rescaling Test Data
test_data_generator = ImageDataGenerator()

training_data_iterator = training_data_generator.flow_from_directory(
    training_dir,
    image_size,
    class_mode='binary',
    batch_size=batch_size,
    color_mode='grayscale')

test_data_iterator = test_data_generator.flow_from_directory(
    test_dir,
    image_size,
    class_mode='binary',
    batch_size=batch_size,
    color_mode='grayscale')

model.fit_generator(
    training_data_iterator,
    steps_per_epoch=training_img_nmb // batch_size,
    epochs=epochs,
    validation_data=test_data_iterator,
    validation_steps=test_img_nmb // batch_size
)

model.save('bird_squirrel.hdf5')
