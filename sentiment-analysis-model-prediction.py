import re
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from nltk.tag import pos_tag
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import classify, NaiveBayesClassifier
import random
from nltk.tokenize import word_tokenize


class SentimentPredict:
	def __init__(self):
		self.stopwords = set(stopwords.words('english'))
		self.positive_tweets = twitter_samples.tokenized('positive_tweets.json')
		self.negative_tweets = twitter_samples.tokenized('negative_tweets.json')
	
	def tagging_position(self, tagging):
			if tagging.startswith('NN'): self.position = 'n'
			elif tagging.startswith('VB'): self.position = 'v'
			else: self.position = 'a'
			return self.position	

	def tweet_regularization(self, words, stopwords):
		self.standardized_tweet = []
		for word, tagging in pos_tag(words):
			self.word = re.sub(	r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', str(word))
			self.word = re.sub('(@[A-Za-z0-9_]+)','', self.word)

			self.tagging_value = self.tagging_position(tagging)
			self.wordLemmatizer = WordNetLemmatizer()
			self.word = self.wordLemmatizer.lemmatize(self.word, self.tagging_value)

			if len(self.word) > 0 and self.word not in string.punctuation and self.word.lower() not in self.stopwords:
				self.standardized_tweet.append(self.word.lower())

		return self.standardized_tweet

	def generate_dictionary_for_model(self, values):
		for value in values:
			yield dict([val, True] for val in value)

	def model(self):
		self.regularized_positive, self.regularized_negative  = [], []
		
		for positive_tweet in self.positive_tweets:
			self.regularized_positive.append(self.tweet_regularization(positive_tweet, self.stopwords))

		for negative_tweet in self.negative_tweets:
			self.regularized_negative.append(self.tweet_regularization(negative_tweet, self.stopwords))

		self.posetivity = self.generate_dictionary_for_model(self.regularized_positive)
		self.negativity = self.generate_dictionary_for_model(self.regularized_negative)

		self.positive_data = [(tweet, 'Positive') for tweet in self.posetivity]
		self.negative_data = [(tweet, 'Negative') for tweet in self.negativity]

		self.total = shuffle(self.positive_data + self.negative_data)

		self.X_train = self.total
		
		self.model = NaiveBayesClassifier.train(self.X_train)
		return self.model
	
	def predict(self, tweet):
		self.classifier = self.model()
		self.regularized_tweet = self.tweet_regularization(word_tokenize(tweet), self.stopwords)
		self.prediction = (tweet, self.classifier.classify(dict([tweet, True] for tweet in self.regularized_tweet)))

		return self.prediction


classifier = SentimentPredict()
prediction = classifier.predict(tweet = 'This could be the best day in my life')
print (prediction)
