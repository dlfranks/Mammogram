from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

 
document = ["One Geek helps Two Geeks",
            "Two Geeks help Four Geeks",
            "Each Geek helps many other Geeks at GeeksforGeeks"]
df = pd.DataFrame(document, columns=['message'])

# Create a Vectorizer Object
vectorizer = CountVectorizer()
 
vectorizer.fit(document)
 
# Printing the identified Unique words along with their indices
print("Vocabulary: ", vectorizer.vocabulary_)
 
# Encode the Document
vector = vectorizer.transform(document)
 
# Summarizing the Encoded Texts
print("Encoded Document is:")
print(vector.toarray())

tfidfTransformer = TfidfTransformer().fit(vector)
print(tfidfTransformer)
message_tfidf = tfidfTransformer.transform(vector)
print(message_tfidf)

model = MultinomialNB().fit(message_tfidf, )