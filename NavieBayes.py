import os
import io
import numpy
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = pd.concat([data, dataFrameFromDirectory("../../MLCourse/emails/spam", "spam")]);
data = pd.concat([data, dataFrameFromDirectory("../../MLCourse/emails/ham", "ham")])

#For Pandas 1.3:
#data = data.append(dataFrameFromDirectory('emails/spam', 'spam'))
#data = data.append(dataFrameFromDirectory('emails/ham', 'ham'))

vectorizer = CountVectorizer()
#print(data['message'].values)
#vectorizer.fit(data['message'])
#print(vectorizer.vocabulary_)
counts = vectorizer.fit_transform(data['message'].values)
#print(counts)
classifier = MultinomialNB()
targets = data['class'].values
# Created a model naive based
classifier.fit(counts, targets)
# try predition
examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?", 'Free training for users']
example_counts = vectorizer.transform(examples)
print(example_counts)
predition = classifier.predict(example_counts)
print(predition)