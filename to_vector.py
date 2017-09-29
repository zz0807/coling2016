from gensim.models import KeyedVectors
import pickle
import numpy as np

def word_to_vec(sentence):
    sentence_vec_list = []
    for word in sentence:
        if word in model.wv.vocab:
            sentence_vec_list.append(model[word].tolist())
    return sentence_vec_list

model = KeyedVectors.load_word2vec_format('model.bin', binary=True)
all_sentences_vector = []

with open('test_data.pkl', 'rb') as file:
    all_data = pickle.load(file)

for single in all_data:
    sentence_vector = word_to_vec(single['text'].split())
    all_sentences_vector.append(sentence_vector)

all_sentences_vector = np.array(all_sentences_vector)
np.save("sw_test_x.npy", all_sentences_vector)

