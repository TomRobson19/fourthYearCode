import numpy as np
import csv
import string

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer

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
			else:
				ids.append(int(item[0]))
				text.append(item[1])
				labels.append(int(item[2]))
	return ids, text, labels

def cleanData(text):
	for article in text:
		article = article.lower()
		article = article.replace('\n', ' ')
		article = article.replace('  ', ' ')
		import re
		article = re.sub(r'([^\s\w]|_)+', '', article)
	return text

def evaluatePrediction(prediction,true):
	print(classification_report(prediction,true))

if __name__ == '__main__':
	print("Reading in Data")
	ids, text, labels = readData()

	print("Cleaning Data")
	text = cleanData(text)

	x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=26)

	print("Features")

	#CountVectoriser
	vect = TfidfVectorizer(
		ngram_range=(1,5),
		min_df=10,
		max_df=0.6,
		analyzer="word"
		)

	vect = vect.fit(x_train)
	x_train = vect.transform(x_train)
	x_test = vect.transform(x_test)

	classifier = MultinomialNB(alpha=0)
	classifier.fit(x_train, y_train)
	prediction = classifier.predict(x_test)
	true = y_test

	evaluatePrediction(prediction,true)