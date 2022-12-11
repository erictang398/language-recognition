import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer

# read in the csv, r means raw string, dont treat "\" as an escape character
data = pd.read_csv(r'C:\Users\Eric\Desktop\text-recognition\language-recognition\training_language_dataset.csv')

# break the dataframe columns into two arrays
text = np.array(data["Text"])
lang = np.array(data["Language"])

# count vectorizer changes each word to a number of occurances in a matrix
# for example, hello world may show up as [0 1 1] in an array with titles [hi hello world]
cv = CountVectorizer()

# converts the text to numbers in the matrix
TEXT = cv.fit_transform(text)

# frac is the fraction of the dataset to return, so 1 means all. data.sample(frac=0.2)
# drop=True prevents reset_index from creating a new column containing the old indices
# give 20% to testValidate, random_state will gurantee the same data split each time for reproduceable results
TEXT_train, TEXT_test, lang_train, lang_test = train_test_split(TEXT, lang, test_size=0.2, random_state=0)

# the multinomial naive bayes algorithm.
#model = MultinomialNB()
model = GaussianNB

# essentially feeding the model the data, the input and expected output 
model.fit(TEXT_train, lang_train)

# the accuracy on the test data 
# print(model.score(TEXT_test, lang_test))

input = input("Enter Something: ")

# convert the 1s in the matrix into a vector, .transform will produce the hits in array, .toarray converts to vector 
token = cv.transform([input]).toarray()
output = model.predict(token)
print(output)
