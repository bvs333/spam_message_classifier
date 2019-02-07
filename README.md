# spam_message_classifier
This is a machine learning model which classifies the message into spam or not

The dataset had five fields , but the message was present in the column 'v2'. Fistly I have processed the data before classifying it.
At first I have converted the whole message into lower because a word present in captial or small means the same.Then using the PorterStemmer 
class in nltk library I have performed stemming on message if the word was not a stopword. Then I have given the whole message dataset to
CountVectorizer class with max features of 1500. Based on the dataset it has classified the messages into spam or ham.This model has an accuracy
of 82.10 .
