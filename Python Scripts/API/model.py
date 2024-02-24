import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class SA_Model():
    def __init__(self):
        self.model = load_model('SA_Model_Final_v8')
        self.classes = ['joy', 'fear', 'anger', 'sadness', 'love', 'surprise']
        self.class_to_index = dict((c,i) for i, c in enumerate(self.classes))
        self.index_to_class = dict((v, k) for k, v in self.class_to_index.items())
        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def get_sequences(self, msg):
        maxlen = 50
        sequences = self.tokenizer.texts_to_sequences(msg)
        padded = pad_sequences(sequences, truncating='post' , padding='post', maxlen = maxlen)
        return padded
        
    def get_emotion(self, msg):
        msg_seq = self.get_sequences([msg])

        p = self.model.predict(msg_seq)[0]
        pred_class = self.index_to_class[np.argmax(p).astype('uint8')]

        return pred_class
