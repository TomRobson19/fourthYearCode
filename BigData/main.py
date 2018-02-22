import numpy as np
import csv
import string

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, SimpleRNN, RNN, Dense, Input, GlobalMaxPooling1D
from keras.models import Model

import spacy

def readData():
	filename = "news_ds.csv"

	ids = []
	text = []
	labels = []

	with open(filename) as file:
		csvReader = csv.reader(file, delimiter=',', quotechar='"') #“”
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
		text[i] = re.sub(r'([^\s\w]|_)+', '', text[i])
		text[i] = re.sub(r'^https?:\/\/.*[\r\n]*', '', text[i])
		text[i] = re.sub( '\s+', ' ', text[i]).strip()
	return text

def evaluatePrediction(prediction,true):
	print(classification_report(true,prediction))
	print(accuracy_score(true,prediction))

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

if __name__ == '__main__':
	print("Reading Data")
	ids, text, labels = readData()

	print("Cleaning Data")
	text = cleanData(text)
	#print(text)

	shallow = False

	if shallow:

		x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=26)

		print("Extracting Features")

		# grid search below parameters for both Count(tf) and tf-idf

		# vect = CountVectoriser(
		# 	ngram_range=(1,5),
		# 	min_df=10,
		# 	max_df=0.6,
		# 	analyzer="word")

		vect = TfidfVectorizer(
			ngram_range=(1,5),
			min_df=10,
			max_df=0.6,
			analyzer="word"
			)

		print("Training MultinomialNaiveBayes")

		vect = vect.fit(x_train)
		x_train = vect.transform(x_train)
		x_test = vect.transform(x_test)

		classifier = MultinomialNB(alpha=0)
		classifier.fit(x_train, y_train)
		prediction = classifier.predict(x_test)
		true = y_test

		print("Evaluating")

		evaluatePrediction(prediction,true)

	deep = True
	if deep:
		# print("Getting word2vec Vectors")
		# vectors = getWordVectors(text)
		# print(vectors)

		print("Tokenising text data")
		MAX_NUM_WORDS = 20000
		MAX_SEQUENCE_LENGTH = 1000
		EMBEDDING_DIM = 200
		VALIDATION_SPLIT = 0.2

		tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
		tokenizer.fit_on_texts(text)
		sequences = tokenizer.texts_to_sequences(text)

		word_index = tokenizer.word_index
		print('Found %s unique tokens.' % len(word_index))

		data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

		labels = to_categorical(np.asarray(labels))
		print('Shape of data tensor:', data.shape)
		print('Shape of label tensor:', labels.shape)

		# split the data into a training set and a validation set
		indices = np.arange(data.shape[0])
		np.random.shuffle(indices)
		data = data[indices]
		labels = labels[indices]
		nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

		x_train = data[:-nb_validation_samples]
		y_train = labels[:-nb_validation_samples]
		x_test = data[-nb_validation_samples:]
		y_test = labels[-nb_validation_samples:]

		print("Getting embeddings")
		embeddings_index = {}
		f = open("glove.twitter.27B.200d.txt")
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


		embedding_layer = Embedding(len(word_index) + 1,
		                            EMBEDDING_DIM,
		                            weights=[embedding_matrix],
		                            input_length=MAX_SEQUENCE_LENGTH,
		                            trainable=False)

		sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
		embedded_sequences = embedding_layer(sequence_input)

		lstm = True
		if lstm:
			print("Using LSTM")
			x = Conv1D(128, 5, activation='relu')(embedded_sequences)
			x = MaxPooling1D(5)(x)
			x = Conv1D(128, 5, activation='relu')(x)
			x = MaxPooling1D(5)(x)
			#x = Conv1D(128, 5, activation='relu')(x)
			#x = GlobalMaxPooling1D()(x)
			x = LSTM(5)(x)
			x = Dense(128, activation='relu')(x)
		else:
			print("Using RNN")
			x = Conv1D(128, 5, activation='relu')(embedded_sequences)
			x = MaxPooling1D(5)(x)
			x = Conv1D(128, 5, activation='relu')(x)
			x = MaxPooling1D(5)(x)
			#x = Conv1D(128, 5, activation='relu')(x)
			#x = GlobalMaxPooling1D()(x)
			# This should be the RNN base class not the Simple RNN
			x = SimpleRNN(5)(x)
			x = Dense(128, activation='relu')(x)



		preds = Dense(2, activation='softmax')(x)

		model = Model(sequence_input, preds)
		model.compile(loss='categorical_crossentropy',
		              optimizer='rmsprop',
		              metrics=['acc'])

		# happy learning!
		model.fit(x_train, y_train, validation_data=(x_test, y_test),
		          epochs=20, batch_size=128)

		prediction = model.predict(x_test)

		#evaluatePrediction(prediction,y_test)