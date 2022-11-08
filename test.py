import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import keras
from keras_preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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


train['processed_text'] = train['text'].apply(lambda x: x.lower())


lemm = WordNetLemmatizer()
train['processed_text'] = train['processed_text'].apply(lambda x: lemm.lemmatize(x))

train['processed_text'] = train['processed_text'].apply(remove_stopwords)

train['processed_text'] = train['processed_text'].apply(clean_text)



X = train['processed_text']

Y = train['target']


token = Tokenizer(num_words=10000,oov_token="<OOV>")

token.fit_on_texts(X)

word_index = token.word_index

training_seq = token.texts_to_sequences(X)

train_padded = pad_sequences(training_seq,padding="post",truncating="post",maxlen=50)


X_train,X_valid,y_train,y_valid = train_test_split(train_padded,Y,test_size=0.2,train_size=0.8)
model = Sequential()


model = keras.models.Sequential()
model.add(Embedding(10000,500))
model.add(LSTM(64))
model.add(LSTM(64))
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(1,activation="sigmoid"))



model.compile("rmsprop", "binary_crossentropy", metrics=["accuracy"])


model = keras.models.load_model('./model')

results = model.evaluate(X_valid, y_valid, batch_size=128)

print(results)