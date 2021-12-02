import cv2 as cv                                                                                    # py -m pip install opencv-python
import numpy as np
import matplotlib.pyplot as plt                                                                     # py -m pip install matplotlib
import tensorflow as tf                                                                             # py -m pip install tensorflow

mnist = tf.keras.datasets.mnist                                                                     # Theres already existing libraries of training data, so i just grabbed some rather than making my own
(x_train, y_train), (x_test, y_test) = mnist.load_data()                                            # Here the dataset is placed in its proper catergories

x_train = tf.keras.utils.normalize(x_train, axis=1)                                                 
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()                                                                # Here we make the Neural Net following a sequential pattern
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))                                             # The first layer of our Neural Net is the input layer, where we load the images. We limit image to 28px * 28px (784px) to lower the amount of neurons we're dealing with. 
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))                                  # Second "hidden" layer of the neural net, i have no fucking clue what goes on here... but it seems to work. relu or "rectified linear unit" is used instead of sigmoid as it acchieves virtualy the same results but faster
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))                                  # Third "hidden" layer of the neural net, here we use relu again for the same reasons as in the second "hidden" layer
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))                                # Fourth and final layer, here we ofcourse limit the neurons to just 10 so we can match them with the number it is supposed to guess.

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])       # Here we compile the model to our specefied settings

model.fit(x_train, y_train, epochs=5)                                                               # Set to 3 if everything fucks up

loss, accuracy = model.evaluate(x_test, y_test)                                                     # Here we get the models Accuracy, and Loss
print(accuracy)                                                                                     # Here we print our accuracy from the previous line (higher is better)
print(loss)                                                                                         # Here we print the loss value from the second previous line (lower is better)

model.save('digits.model')                                                                          # Here we save the model as 'digits.model'

for x in range(1,10):                                                                               # Since theres 9 images (1 through 9) we run a loop going through each image, one after another
    img = cv.imread(f'{x}.png')[:,:,0]                                                              # Here the images are taken from the main folder, and parsed into the program
    img = np.invert(np.array([img]))                                                                # Here we invert the images, since it for some reason has already been inverted. And our AI has been trained on black on white background
    prediction = model.predict(img)                                                                 # Here we get what the model's/neural net's/ai's prediction
    print(f'Guessed Number: {np.argmax(prediction)}')                                               # Here we spit out the model's/neural net's/ai's prediction
    plt.imshow(img[0], cmap=plt.cm.binary)                                                          # Here our image viewing program fetches the image
    plt.show()                                                                                      # Here we load our image viewing program