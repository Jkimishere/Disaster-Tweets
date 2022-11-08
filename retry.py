import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
from keras_preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import keras
import keras.backend as K
from keras import Sequential
from keras.layers import  Dense, Bidirectional, Embedding, Dropout, LSTM
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

count0 = (train['target'] == 0).sum()
count1 = (train['target']==1).sum()

stop_words = stopwords.words('english')
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



train['processed_text'] = train['text'].apply(lambda x: x.lower())
lemm = WordNetLemmatizer()
train['processed_text'] = train['processed_text'].apply(lambda x: lemm.lemmatize(x))
train['processed_text'] = train['processed_text'].apply(remove_stopwords)
train['processed_text'] = train['processed_text'].apply(clean_text)

test['processed_text'] = test['text'].apply(lambda x: x.lower())
test['processed_text'] = test['processed_text'].apply(lambda x: lemm.lemmatize(x))
test['processed_text'] = test['processed_text'].apply(remove_stopwords)
test['processed_text'] = test['processed_text'].apply(clean_text)

X_train = train['processed_text']
X_test = test['processed_text']
y_train = train['target']

token = Tokenizer(num_words=10000,oov_token="<OOV>")

token.fit_on_texts(X_train)
token.fit_on_texts(X_test)

word_index = token.word_index

training_seq= token.texts_to_sequences(X_train)
testing_seq = token.texts_to_sequences(X_test)

train_padded = pad_sequences(training_seq,padding="post",truncating="post",maxlen=50)
test_padded = pad_sequences(testing_seq,padding="post",truncating="post",maxlen=50)


X_train, X_valid, y_train, y_valid = train_test_split(train_padded, y_train,test_size=0.1,train_size=0.9, shuffle=True)




model = Sequential()
model.add(Embedding(1000, 100))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer= 'SGD', loss= keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])

K.set_value(model.optimizer.learning_rate, 0.001)

print(X_train, y_train)
history = model.fit(X_train, y_train, batch_size=20, epochs=10, validation_data=(X_valid, y_valid))

model.save('./model')

fig , ax = plt.subplots(1,2)
fig.set_size_inches(12,4)

ax[0].plot(history.history['binary_accuracy'])
ax[0].plot(history.history['val_binary_accuracy'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Training Loss vs Validation Loss')