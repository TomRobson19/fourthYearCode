import numpy as np
import sklearn
import csv
import string

def readData():
	filename = "news_ds.csv"

	ids = []
	text = []
	labels = []

	with open(filename) as file:
		csvReader = csv.reader(file, delimiter=',', quotechar='"')
		for item in csvReader:
			ids.append(item[0])
			text.append(item[1])
			labels.append(item[2])
	return ids, text, labels

def cleanData(text):
	for article in text:
		article = article.lower()
		article = article.replace('\n', ' ')
		article = article.replace('  ', ' ')
		import re
		article = re.sub(r'([^\s\w]|_)+', '', article)
		print(article)

if __name__ == '__main__':
	print("Reading in Data")
	ids, text, labels = readData()

	print("Cleaning Data")
	text = cleanData(text)

	x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.1, random_state=69)