from tensorflow.keras.preprocessing.text import one_hot
docs = ['Well done!',
 'Good work',
 'Great effort',
 'nice work',
 'Excellent!',
 'Weak',
 'Poor effort!',
 'not good',
 'poor work',
 'Could have done better.']

vocab_size = 50
encoded_docs = [one_hot(i, vocab_size) for i in docs]

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, Flatten

max_len = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_len, padding='post')

from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length = max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

import numpy as np
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
model.fit(padded_docs, labels, epochs=50, verbose=0)
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print("Accuracy: %f" % (accuracy*100))
print(model.predict(padded_docs))
