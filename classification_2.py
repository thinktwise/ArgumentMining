#import numpy as np
import os

import nltk
import numpy as np
import tensorflow_hub as hub
from keras import layers, Model, optimizers
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import Preprocessing.wikipedia_dataset
import tensorflow as tf
import keras.layers
import keras.callbacks
import keras.metrics
from sklearn.preprocessing import OneHotEncoder

import pickle

X, Y = Preprocessing.wikipedia_dataset.getDataLabelledSentences()
X_updated = []

#for sentence in X:
#    tagged_sentence = nltk.tag.pos_tag(sentence.split())
#    print(tagged_sentence)
#    edited_sentence = [word for word,tag in tagged_sentence if tag != 'NN' and tag != 'NNPS' and tag != 'NNP']

#    print(' '.join(edited_sentence))
#    X_updated.append(' '.join(edited_sentence))

#print("X")
#print(X)

#print("Y")
#print(Y)
#Y = Y[0]
print(len(X))
print(len(Y))

#X = X[0:1000]
#Y = Y[0:1000]
train_len = int(len(X)/100 * 80)

print("train_len")
print(train_len)

embed = hub.load("universal-sentence-encoder_4")

#X_embed = embed(X)
#print(len(X_embed))

type_one_hot = OneHotEncoder(sparse=False).fit_transform(
    np.array(Y).reshape(-1, 1)
)

#Y = np.array(Y)

#X_sent_train = X[0:train_len]
#X_sent_test = X[train_len:len(X)-1]
#Y_train = Y[0:train_len]
#Y_test = Y[train_len:len(Y)-1]

Reload_embeddings = False

X_train = []
X_test = []

# does not work yet, do not use this code
if not Reload_embeddings:
    with open('sentences_embedded_train_2.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('sentences_embedded_test_2.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('Y_train_2.pkl', 'rb') as f:
        Y_train = pickle.load(f)
    with open('Y_test_2.pkl', 'rb') as f:
        Y_test = pickle.load(f)

else:
    X_sent_train, X_sent_test, Y_train, Y_test = \
        train_test_split(
            X,
            type_one_hot,
            test_size=.1
        )

    with open('Y_train_2.pkl', 'wb') as f:
        pickle.dump(Y_train, f)
    with open('Y_test_2.pkl', 'wb') as f:
        pickle.dump(Y_test, f)

    for r in X_sent_train:
      #print(r)
      emb = embed([r])
      review_emb = tf.reshape(emb, [-1]).numpy()
      X_train.append(review_emb)

    X_train = np.array(X_train)
    with open('sentences_embedded_train_2.pkl', 'wb') as f:
        pickle.dump(X_train, f)

    for r in X_sent_test:
      emb = embed([r])
      review_emb = tf.reshape(emb, [-1]).numpy()
      X_test.append(review_emb)

    X_test = np.array(X_test)
    with open('sentences_embedded_test_2.pkl', 'wb') as f:
        pickle.dump(X_test, f)


print(X_train.shape, Y_train.shape)

import random

X_train_diminued = []
Y_train_diminued = []

print("before")
print(Y_test)

# This slow and poorly written function set the ratio of positive and negative example
# TODO vectorize this
percentage_chance = 0
i = 0
for idx, sent in enumerate(X_train):
    #print(random.random())
    if random.random() < percentage_chance and Y_train[idx] == 0:
        #print("we skip")
        continue
    else:
        X_train_diminued.append(sent)
        Y_train_diminued.append(Y_train[idx])
        i = i + 1
        #print("we keep")



X_train = np.array(X_train_diminued)
Y_train = np.array(Y_train_diminued)

print("after")
print(Y_test)



print(X_train.shape, Y_train.shape)

#testing purpose: less data to speed up
#X_train = X[0:80]
#X_test = X[80:100]

#Y_train = Y[0:80]
#Y_test = Y[80:100]


# You must download https://tfhub.dev/google/universal-sentence-encoder/4 and extract it  

model = keras.Sequential()
model.add(
  keras.layers.Dense(
    units=256,
    input_shape=(X_train.shape[1], ),
    #activation='relu',
    #kernel_initializer='lecun_normal',
    activation='relu'
  )
)
model.add(
  keras.layers.Dropout(rate=0.5)
)
model.add(
  keras.layers.Dense(
    units=128,
    #activation='relu',
    #kernel_initializer='lecun_normal',
    activation='relu'
  )
)
#model.add(
#  keras.layers.Dropout(rate=0.1)
#)
model.add(keras.layers.Dense(2, activation='sigmoid'))

#model = keras.Sequential()
#model.add(layers.Embedding(28650, 512, input_length=1,  name="w2v_embedding"))
#model.add(keras.layers.Dense(64, activation='relu'))
#model.add(keras.layers.Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy',keras.metrics.Precision(), keras.metrics.Recall()])


#checkpoint_path = "model_weights"

# Create a callback that saves the model's weights
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                 save_weights_only=True,
#                                                 verbose=1)

es = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=0, mode='auto')

print("test")
history = model.fit(
    X_train, Y_train,
    epochs=100,
    callbacks=[es],
    batch_size=16,
    validation_split=0.1,
    verbose=1,
    shuffle=True,
    class_weight= {0: 1.,
                1: 3}

)

#model.fit(X,
#          Y,
#          epochs=2)
          #callbacks=[cp_callback])
          #validation_data=(x_test, y_test))



score=model.evaluate(X_test, Y_test, verbose=2)

print(score)

predictions = model.predict(X_test)

#for idx, prediction in enumerate(predictions):
#    print(X_sent_test[idx])
#    if np.argmax(prediction) == 0:
#        print("not claim")
#    else:
#        print("A claim")
#    print(prediction)

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

print("y_pred")
print(y_pred)
#y_pred [y_pred[0] > 0.4] = 1
#y_pred [y_pred[0] <= 0.4] = 0

thresh = 0.5
#y_pred = [1 if a_ > thresh else 0 for a_ in y_pred]

y_pred = np.argmax(y_pred, axis=1)

print("y_pred")
print(y_pred)

print("Y_test")
print(Y_test)

Y_test = np.argmax(Y_test, axis=1)

print("Y_test")
print(Y_test)


print(classification_report(Y_test, y_pred, target_names=["Not claim", "Claim"]))
#print(classification_report(Y_test, y_pred, target_names=['not claim', 'claim']))

directory = os.fsencode("Datasets/Wikipedia/test")


#for file in os.listdir(directory):
#    filename = os.fsdecode(file)
#    print(filename)
#    if filename.endswith(".txt") or filename.endswith(".py"):
#        with open(os.path.join(directory, file), 'r') as txt_file:
#            txt = txt_file.read().replace('\n', '')
#            # print(txt)
#            sentences = nltk.tokenize.sent_tokenize(txt)
#            sentences_emb = []
#            for sentence in sentences:
#                # print(r)
#                emb = embed([sentence])
#                sent_emb = tf.reshape(emb, [-1]).numpy()
#                sentences_emb.append(sent_emb)

#            sentences_emb = np.array(sentences_emb)

#            predictions = model.predict(sentences_emb)
#            predictions = [1 if a_ > thresh else 0 for a_ in predictions]

#            for idx, prediction in enumerate(predictions):
#                if prediction == 0:
 #                   continue
 #                   print("not claim")
 #               else:
 #                   print(sentences[idx])
#                    print("A claim")
                #print(prediction)



