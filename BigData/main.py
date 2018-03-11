import numpy as np
import csv
import string
import time

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPooling1D, Embedding, CuDNNLSTM, LSTM, SimpleRNN, RNN, Dense, Input, GlobalMaxPooling1D, Dropout, Flatten, Layer
from keras.models import Model

import keras.backend as K

outputFolder = "output"
import time
ts = time.time()
outputFolder = outputFolder+"/"+str(ts).split(".")[0]
tbCallBack = TensorBoard(log_dir=outputFolder+'/log', histogram_freq=0,  write_graph=True, write_images=True)


class MinimalRNNCell(Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]


def readData():
	filename = "news_ds.csv"

	ids = []
	text = []
	labels = []

	with open(filename) as file:
		csvReader = csv.reader(file, delimiter=',', quotechar='"') 
		first = True
		for item in csvReader:
			if first:
				first = False
			elif len(item[1]) < 10:
				continue
			else:
				ids.append(int(item[0]))
				text.append(item[1])
				labels.append(int(item[2]))

	return ids, text, labels


#there are some empty articles in the data, need to remove these and their labels
def cleanData(text):
	for i in range(len(text)):
		text[i] = text[i].lower()
		text[i] = text[i].replace('\n', ' ')
		import re
		text[i] = re.sub(r'^https?:\/\/.*[\r\n]*', '', text[i])
		text[i] = re.sub( '\s+', ' ', text[i]).strip()

		text[i] = re.sub(r'(\S+)@(\S+)', '', text[i])
		text[i] = re.sub(r'(\A|\s)@(\w+)', '', text[i])
		text[i] = re.sub(r'(\A|\s)#(\w+)', '', text[i])
		text[i] = re.sub(r'([^\s\w]|_)+', '', text[i])
	return text

def evaluatePrediction(prediction,true):
	print(classification_report(true,prediction,digits=4, target_names=["Fake","Real"]))
	print("Accuracy: "+str(accuracy_score(true,prediction)))

def getWordVectors(text):
	nlp = spacy.load('en_core_web_lg')

	word2VecText = []

	for article in text:
		tokens = nlp(article)
		word2VecArticle = []

		for token in tokens:
			word2VecArticle.append(token.vector_norm)

		word2VecText.append(word2VecArticle)

	return word2VecText

def main():
	print("Reading Data")
	ids, text, labels = readData()

	print("Cleaning Data")
	text = cleanData(text)

	x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=26)

	print("Shallow Learning - tf-idf, 1-4 grams")

	start = time.time()

	print("Extracting Features")

	# print("Using tf")
	# vect = CountVectorizer(
	# 	ngram_range=(1,3),
	# 	min_df=10,
	# 	analyzer="word")

	print("Using tf-idf")
	vect = TfidfVectorizer(
		ngram_range=(1,4),
		min_df=10,
		analyzer="word"
		)

	print("Training MultinomialNaiveBayes")

	vect = vect.fit(x_train)
	x_train_vec = vect.transform(x_train)
	x_test_vec = vect.transform(x_test)

	classifier = MultinomialNB(alpha=0)
	classifier.fit(x_train_vec, y_train)
	prediction = classifier.predict(x_test_vec)
	true = y_test

	end = time.time()

	print("Evaluating")
	evaluatePrediction(prediction,true)
	print("Runtime: "+str(end-start))

	####################################################################################	

	print("Deep Learning - LSTM, 128 units")
	start = time.time()

	#Ref: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

	print("Tokenising text data")
	MAX_NUM_WORDS = 20000
	MAX_SEQUENCE_LENGTH = 1000
	EMBEDDING_DIM = 300
	VALIDATION_SPLIT = 0.2

	tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
	tokenizer.fit_on_texts(text)
	sequences = tokenizer.texts_to_sequences(x_train)
	sequences_test = tokenizer.texts_to_sequences(x_test)

	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))

	data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

	test_data = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

	labels = np.asarray(y_train)
	print('Shape of data tensor:', data.shape)
	print('Shape of label tensor:', labels.shape)

	# split the data into a training set and a validation set
	indices = np.arange(data.shape[0])
	np.random.shuffle(indices)
	data = data[indices]
	labels = labels[indices]
	nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

	x_train_deep = data[:-nb_validation_samples]
	y_train_deep = labels[:-nb_validation_samples]
	x_val = data[-nb_validation_samples:]
	y_val = labels[-nb_validation_samples:]

	print("Getting GloVe embeddings")
	embeddings_index = {}
	f = open("glove.6B.300d.txt")
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()

	print('Found %s word vectors.' % len(embeddings_index))

	print("Making embedding matrix")
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        # words not found in embedding index will be all-zeros.
	        embedding_matrix[i] = embedding_vector

	end = time.time()
	extractionTime = end-start

	layers = ["LSTM","RNN"]

	for i in layers:
		start = time.time()

		print("Using "+i)

		embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)

		sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
		embedded_sequences = embedding_layer(sequence_input)

		x = Conv1D(128, 5, activation='relu')(embedded_sequences)
		x = MaxPooling1D(5)(x)
		x = Conv1D(128, 5, activation='relu')(x)
		x = MaxPooling1D(5)(x)

		if i=="LSTM":
			x = LSTM(128)(x)
			#x = CuDNNLSTM(128)(x)
		else:
			cell = MinimalRNNCell(64)
			x = RNN(cell)(x)
		

		x = Dropout(0.25)(x)
		x = Dense(128)(x)
		preds = Dense(1, activation='sigmoid')(x)

		model = Model(sequence_input, preds)
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

		model.fit(x_train_deep, y_train_deep, validation_data=(x_val, y_val), epochs=20, batch_size=128, callbacks=[tbCallBack], verbose=2)

		prediction = K.eval(K.cast(K.greater(model.predict(test_data),0.5), "float32"))

		end = time.time()

		evaluatePrediction(prediction,y_test)

		print("Runtime: "+str((end-start)+extractionTime))
		

if __name__ == '__main__':
	main()