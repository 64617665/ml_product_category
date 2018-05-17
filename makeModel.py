import numpy as np

training = np.genfromtxt('C:/Users/Dave/Downloads/mltest.csv', delimiter=',', skip_header=1, usecols=(0, 1), dtype=None, encoding='latin1', case_sensitive=False)
#could also use converters property to keep text and then change into 0,1,2,3, etc... 
#https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.genfromtxt.html

#training = np.loadtxt('C:/Users/Dave/Downloads/mltest.csv', delimiter=',', skiprows=1, usecols=(1, 2), dtype=None, encoding='utf-8')

#[x.encode('utf-8') for x in training]
train_x = [str(x[0]) for x in training]
train_y = np.asarray([x[1] for x in training])
#may need to adjust train_y to numbers but would ideally like to leave as string? performance loss?

import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer

max_words = 3000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_x)


dictionary = tokenizer.word_index
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)


def convert_text_to_index_array(text):
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

allWordIndices = []

for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)


allWordIndices = np.asarray(allWordIndices)


train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
train_y = keras.utils.to_categorical(train_y, 5) #red, white, rose, sparkling, specialty [0,1,2,3,4]


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

#model parameters
model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax')) #array size here must match train_y array size

#setup equations
model.compile(loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])


#train model, adjust these values for best fit
model.fit(train_x, train_y,
  batch_size=32,
  epochs=5,
  verbose=1,
  validation_split=0.1,
  shuffle=True)

#save model, adjust filenames for each model run

name_json = 'model_20180517.json'
name_h5   = 'model_20180517.h5'

model_json = model.to_json()
with open(name_json, 'w') as json_file:
    json_file.write(model_json)

model.save_weights(name_h5)