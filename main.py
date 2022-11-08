import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import keras
from keras import Sequential

from keras_preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')



stop_words = stopwords.words("english")
def remove_stopwords(text):
    no_stop = []
    for word in text.split(' '):
        if word not in stop_words:
            no_stop.append(word)
    return " ".join(no_stop)

    

def clean_text(text):
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'https?://[^\s\n\r]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^0-9a-zA-Z:,]+', ' ', text)
    return text

def prepare_data(train, is_train):
    train['processed_text'] = train['text'].apply(lambda x: x.lower())


    lemm = WordNetLemmatizer()
    train['processed_text'] = train['processed_text'].apply(lambda x: lemm.lemmatize(x))

    train['processed_text'] = train['processed_text'].apply(remove_stopwords)

    train['processed_text'] = train['processed_text'].apply(clean_text)



    X = train['processed_text']
    if is_train:
        Y = train['target']



    token = Tokenizer(num_words=10000,oov_token="<OOV>")

    token.fit_on_texts(X)

    word_index = token.word_index

    training_seq = token.texts_to_sequences(X)

    train_padded = pad_sequences(training_seq,padding="post",truncating="post",maxlen=50)

    if is_train:
        return [train_padded, Y]
    else:
        return train_padded

X_train,X_valid,y_train,y_valid = train_test_split(prepare_data(train, True)[0],prepare_data(train, True)[1] ,test_size=0.2,train_size=0.8)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid ,test_size=0.5,train_size=0.5)
model = Sequential()



model.add(keras.layers.Embedding(10000,128))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=True)))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1,activation="sigmoid"))



model.compile("SGD", "binary_crossentropy", metrics=["accuracy"])
from keras import backend as K
K.set_value(model.optimizer.learning_rate, 0.01)
def train():
    history = model.fit(X_train,y_train,epochs=10,  batch_size= 2, validation_data=(X_valid, y_valid), validation_batch_size=1000)


    fig , ax = plt.subplots(1,2)
    fig.set_size_inches(12,4)

    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('Training Accuracy vs Validation Accuracy')
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Training Loss vs Validation Loss')


    plt.show()

    model.save('./model')


def pred():
    loaded = keras.models.load_model('./model')
    print(loaded.summary())
    loaded.evaluate(X_test, y_test)
    

train()
pred()