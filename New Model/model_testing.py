import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_tweet(data):
    tweets = data['text']
    labels = data['label']
    labels = labels.replace(["0","1","2","3","4","5"],['sadness','joy','love','anger','fear','surprise'])
    return tweets, labels

def get_sequences(tokenizer, tweets):
    maxlen = 50
    sequences = tokenizer.texts_to_sequences(tweets)
    padded = pad_sequences(sequences, truncating='post' , padding='post', maxlen = maxlen)
    return padded

def main():

    model = load_model('models/SA_Model_Final_08012022_23_46_27')
    train = pd.read_csv("~/development/anjana/datasets/ECNG3020_Train_Dataset.csv")
   

    tweets, labels = get_tweet(train)
    classes = set(labels)

    tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
    tokenizer.fit_on_texts(tweets)
    tokenizer.texts_to_sequences([tweets[1]])

    class_to_index = dict((c,i) for i, c in enumerate(classes))
    index_to_class = dict((v, k) for k, v in class_to_index.items())

    msg = ["When, as a child, I was nearly knocked down by a car."]
    msg_seq = get_sequences(tokenizer, msg)

    p = model.predict(msg_seq)[0]
    #p = model.predict(np.expand_dims(msg_seq[0], axis=0))[0]
    pred_class = index_to_class[np.argmax(p).astype('uint8')]

    print('Predicted Emotion:', pred_class)

if __name__ == '__main__':
    main()