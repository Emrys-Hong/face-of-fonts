import pandas as pd
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense
from keras.models import Sequential
import gensim
from collections import defaultdict
import numpy as np
from nltk import word_tokenize
from scipy.spatial.distance import cosine
from keras.models import load_model
import tensorflow as tf
import operator


df = pd.read_excel('fonts.xlsx')
classdict = {}
for i in df.index:
    try:
        classdict[df.ix[i, 'Font']].append(df.ix[i, 'Mission_statement'])
    except:
        classdict[df.ix[i, 'Font']] = [df.ix[i, 'Mission_statement']]

# just to randomly shuffle the dataset
#to get 80-20 training and test split
classdict_train = {}
classdict_test = {}
for keys in classdict.keys():
    classdict[keys] = np.random.permutation(classdict[keys])
    classdict_train[keys] = classdict[keys][:int(len(classdict[keys])*4/5)]
    classdict_test[keys] = classdict[keys][int(len(classdict[keys])*4/5):]


wvmodel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)


class SumEmbeddedVecClassifier:
    def __init__(self, wvmodel, classdict, vecsize=300):
        self.wvmodel = wvmodel
        self.classdict = classdict
        self.vecsize = vecsize
        self.trained = False

    def train(self):
        self.addvec = defaultdict(lambda: np.zeros(self.vecsize))
        for classtype in self.classdict:
            for shorttext in self.classdict[classtype]:
                self.addvec[classtype] += self.shorttext_to_embedvec(shorttext)
            self.addvec[classtype] /= np.linalg.norm(self.addvec[classtype])
        self.addvec = dict(self.addvec)
        self.trained = True

    def shorttext_to_embedvec(self, shorttext):
        vec = np.zeros(self.vecsize)
        tokens = word_tokenize(shorttext)
        for token in tokens:
            if token in self.wvmodel:
                vec += self.wvmodel[token]
        norm = np.linalg.norm(vec)
        if norm != 0:
            vec /= np.linalg.norm(vec)
        return vec

    def score(self, shorttext):

        vec = self.shorttext_to_embedvec(shorttext)
        scoredict = {}
        for classtype in self.addvec:
            try:
                scoredict[classtype] = 1 - cosine(vec, self.addvec[classtype])
            except ValueError:
                scoredict[classtype] = np.nan
        return scoredict




class CNNEmbeddedVecClassifier:
    def __init__(self,
                 wvmodel,
                 classdict,
                 n_gram,
                 vecsize=300,
                 nb_filters=1200,
                 maxlen=200):
        self.wvmodel = wvmodel
        self.classdict = classdict
        self.n_gram = n_gram
        self.vecsize = vecsize
        self.nb_filters = nb_filters
        self.maxlen = maxlen
        self.trained = False

    def convert_trainingdata_matrix(self):
        classlabels = self.classdict.keys()
        lblidx_dict = dict(zip(classlabels, range(len(classlabels))))

        # tokenize the words, and determine the word length
        phrases = []
        indices = []
        for label in classlabels:
            for shorttext in self.classdict[label]:
                category_bucket = [0] * len(classlabels)
                category_bucket[lblidx_dict[label]] = 1#generating the one hot vector
                indices.append(category_bucket)
                phrases.append(word_tokenize(shorttext))

        # store embedded vectors
        train_embedvec = np.zeros(shape=(len(phrases), self.maxlen, self.vecsize))
        for i in range(len(phrases)):
            for j in range(min(self.maxlen, len(phrases[i]))):
                train_embedvec[i, j] = self.word_to_embedvec(phrases[i][j])
        indices = np.array(indices, dtype=np.int)

        return classlabels, train_embedvec, indices

    def train(self):
        # convert classdict to training input vectors
        self.classlabels, train_embedvec, indices = self.convert_trainingdata_matrix()

        # build the deep neural network model
        model = Sequential()
        model.add(Convolution1D(nb_filter=self.nb_filters,
                                filter_length=self.n_gram,
                                border_mode='valid',
                                activation='relu',
                                input_shape=(self.maxlen, self.vecsize)))
        model.add(MaxPooling1D(pool_length=self.maxlen - self.n_gram + 1))
        model.add(Flatten())
        model.add(Dense(len(self.classlabels), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        # train the model
        model.fit(train_embedvec, indices)

        # flag switch
        self.model = model
        self.trained = True

    def word_to_embedvec(self, word):
        return self.wvmodel[word] if word in self.wvmodel else np.zeros(self.vecsize)

    def shorttext_to_matrix(self, shorttext):
        tokens = word_tokenize(shorttext)
        matrix = np.zeros((self.maxlen, self.vecsize))
        for i in range(min(self.maxlen, len(tokens))):
            matrix[i] = self.word_to_embedvec(tokens[i])
        return matrix

    def save_model(self):
        self.model.save('mymodel.h5')

    def load_model(self):
        self.model = load_model('mymodel.h5')

    def score(self, shorttext):


        # retrieve vector
        matrix = np.array([self.shorttext_to_matrix(shorttext)])

        # classification using the neural network
        predictions = self.model.predict(matrix)

        # wrangle output result
        scoredict = {}
        for idx, classlabel in zip(range(len(self.classlabels)), self.classlabels):
            scoredict[classlabel] = predictions[0][idx]
        return scoredict

    def compute_accuracy(self):
        classlabels, test_embedvec, indices = CNNEmbeddedVecClassifier(wvmodel,classdict_test,n_gram=5).convert_trainingdata_matrix()
        predictions = self.model.predict(test_embedvec)
        correct_prediction = tf.equal(tf.argmax(predictions,1),tf.argmax(indices,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        with tf.Session() as sess:
            accuracy = sess.run(accuracy)
        return accuracy

    def printing_results(self):
        classlabels, test_embedvec, indices = CNNEmbeddedVecClassifier(wvmodel, classdict_test,
                                                                       n_gram=5).convert_trainingdata_matrix()
        predictions = self.model.predict(test_embedvec)
        list = []
        for i in predictions:
            index, value = max(enumerate(i), key=operator.itemgetter(1))
            list.append(index)

        return list, indices


        # loss, accuracy = self.model.evaluate(test_embedvec,indices)
        # return loss, accuracy


# average = SumEmbeddedVecClassifier(wvmodel, classdict_train, vecsize=300)
# average.train()
# print(average.addvec)
CNN = CNNEmbeddedVecClassifier(wvmodel=wvmodel,classdict=classdict_train,vecsize=300,n_gram=5,nb_filters=2000,maxlen=500)
CNN.train()
# CNN.save_model()
print(CNN.compute_accuracy())
print(CNN.printing_results())