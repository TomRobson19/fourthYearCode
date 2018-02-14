import numpy as np
import csv
import string

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer

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

	x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=26)

	shallow = False

	if shallow:
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

		classifier = MultinomialNB(alpha=0.1)
		classifier.fit(x_train, y_train)
		prediction = classifier.predict(x_test)
		true = y_test

		print("Evaluating")

		evaluatePrediction(prediction,true)

	deep = True
	if deep:
		print("Getting word2vec Vectors")
		vectors = getWordVectors(text)
		print(vectors)