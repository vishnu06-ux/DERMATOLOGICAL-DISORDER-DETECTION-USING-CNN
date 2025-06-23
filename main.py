from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
model = Sequential()
model.add(Convolution2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))
model.add(Convolution2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.summary()
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64,activation = 'relu'))
model.add(Dense(5,activation = 'softmax'))
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, height_shift_range = 0.2, width_shift_range = 0.2,
                                                                                                horizontal_flip = True, vertical_flip = True)
test_datagen = ImageDataGenerator(rescale = 1)
x_train = train_datagen.flow_from_directory(r'D:\Skin-Diseases-Identification-master\Skin Diseases\Skin Diseases\Skin Diseases\train',
                                                                 target_size = (64, 64), batch_size = 32, class_mode = 'categorical')
x_test = test_datagen.flow_from_directory(r'D:\Skin-Diseases-Identification-master\Skin Diseases\Skin Diseases\Skin Diseases\test',
                                                                 target_size = (64, 64), batch_size = 32, class_mode = 'categorical')
x_train.class_indices
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit_generator(x_train, steps_per_epoch =100, epochs = 150, validation_data = x_test, validation_steps = 63)

