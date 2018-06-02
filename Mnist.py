#import liberary
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
#------------------------------------------------------------
#load Data
(train_images, train_labels), (test_images, test_labels)=mnist.load_data()
#------------------------------------------------------------
#PreProcessing
train_images = train_images.reshape((60000,28*28))
test_images = test_images.reshape((10000,28*28))

#normalize 
train_images = train_images.astype('float32')/255
test_images = test_images.astype("float32")/255

# labels normalize
test_labels = to_categorical(test_labels)
train_labels = to_categorical(train_labels)
#------------------------------------------------------------

#Network Architector
network = Sequential()
network.add(Dense(512, activation='relu', input_shape=(784,)))
network.add(Dense(10, activation='softmax'))

#------------------------------------------------------------

#optemizer setting
network.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

#------------------------------------------------------------

#Training
network.fit(train_images,train_labels ,epochs=5,batch_size=128)

#------------------------------------------------------------

# evaluation
network.evaluate(test_images,test_labels)

#------------------------------------------------------------

# Show Result
(loss , accuracy ) = network.evaluate(test_images, test_labels)
print ('--------------------------------------')
print ('your eavaluate accuracy is : ',accuracy)
print ('--------------------------------------')
