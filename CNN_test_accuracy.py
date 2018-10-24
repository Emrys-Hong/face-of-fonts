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
import keras

###make a dictionary like this:
###{'helvetica':["mission 1','mission 2',...],'arial':['mission 1','mission 2']...}
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

###load the gensim model
wvmodel = gensim.models.KeyedVectors.load_word2vec_format('/home/emrys/Desktop/Glove')



class CNNEmbeddedVecClassifier:
    ###inputs
    def __init__(self,
                 wvmodel,
                 classdict,
                 n_gram,
                 vecsize=300,
                 nb_filters=6000,
                 maxlen=200):
        self.wvmodel = wvmodel
        self.classdict = classdict
        self.n_gram = n_gram
        self.vecsize = vecsize
        self.nb_filters = nb_filters
        self.maxlen = maxlen
        self.trained = False

    def convert_trainingdata_matrix(self):
        ###the output is a dictionary like:
        ###{'helvetica': 'a matrix ', ' arial' :'a matrix'}
        classlabels = self.classdict.keys()
        lblidx_dict = dict(zip(classlabels, range(len(classlabels))))

        # tokenize the words, and determine the word length
        phrases = []
        indices = []
        for label in classlabels:
            for shorttext in self.classdict[label]:
                category_bucket = [0] * len(classlabels)
                category_bucket[lblidx_dict[label]] = 1 #generating the one hot vector for labels
                ###so helvitica will be like[ 0 1 0 0 0 0 0 0 0]

                indices.append(category_bucket)
                phrases.append(word_tokenize(shorttext))

        # store embedded vectors, the shape is (len(phrases), self.maxlen(the longest word in a sentence), self.vecsize=300)
        train_embedvec = np.zeros(shape=(len(phrases), self.maxlen, self.vecsize))
        for i in range(len(phrases)):
            for j in range(min(self.maxlen, len(phrases[i]))):
                train_embedvec[i, j] = self.word_to_embedvec(phrases[i][j])
        indices = np.array(indices, dtype=np.int)

        return classlabels, train_embedvec, indices

    def train(self):
        ### add in tensorboard function
        tb = keras.callbacks.TensorBoard(log_dir='/tmp/', histogram_freq=0,
                                    write_graph=True, write_images=True)

        # convert classdict to training input vectors
        self.classlabels, train_embedvec, indices = self.convert_trainingdata_matrix()

        # build the deep neural network model using the keras model, this is how a normal convolutional neural network looks like using keras
        model = Sequential()
        model.add(Convolution1D(nb_filter=self.nb_filters, filter_length=self.n_gram, border_mode='valid', activation='relu', input_shape=(self.maxlen, self.vecsize)))

        model.add(MaxPooling1D(pool_length=self.maxlen - self.n_gram + 1))
        model.add(Flatten())
        model.add(Dense(len(self.classlabels), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


        # train the model, using keras inate functions
        model.fit(train_embedvec, indices, callbacks=[tb],epochs=10)

        # save the model
        self.model = model


    def word_to_embedvec(self, word):
        ###convert a specific work into a vector with a size of 300
        return self.wvmodel[word] if word in self.wvmodel else np.zeros(self.vecsize)

    def shorttext_to_matrix(self, shorttext):
        ###convert a short text into a matrix,
        ###each row stands for one word, and the number of columns is the vector size which is 300
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
        #to predict how many percentage does each font have
        ###the output will be a dictionary
        ###{'helvetica':0.9, 'arial':0.05,...}
        ###this is only to predict for specific text, to test for the general accuracy, need to use compute_accuracy


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
        ###compute the accuracy the mothod used here is similar to how mnist calculate the accuracy
        classlabels, test_embedvec, indices = CNNEmbeddedVecClassifier(wvmodel,classdict_test,n_gram=2,nb_filters=6000,maxlen=200).convert_trainingdata_matrix()
        ###keras have the inate function that can give you accuracy
        predictions = self.model.predict(test_embedvec)
        correct_prediction = tf.equal(tf.argmax(predictions,1),tf.argmax(indices,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        with tf.Session() as sess:
            accuracy = sess.run(accuracy)
        return accuracy
        # loss, accuracy = self.model.evaluate(test_embedvec,indices)
        # return loss, accuracy


# average = SumEmbeddedVecClassifier(wvmodel, classdict_train, vecsize=300)
# average.train()
# print(average.addvec)

CNN = CNNEmbeddedVecClassifier(wvmodel=wvmodel,classdict=classdict_train,vecsize=300,n_gram=2,nb_filters=6000,maxlen=200)
CNN.train()
print(CNN.compute_accuracy())
