import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import swifter

attributes=['positivity','sno','date','flag','account','tweet']
tweets_data=(pd.read_csv('training.1600000.processed.noemoticon.csv', names=attributes, encoding='ISO-8859-1'))
tweets_data.replace({'positivity': {4: 1}}, inplace=True)
port_stem=PorterStemmer()
stop_words=stopwords.words('english')
def clean_text(text):
    text = re.sub('[^a-zA-Z]',' ',text)
    words = text.lower().split()
    cleaned_words = []
    for word in words:
        if word not in stop_words:
            stemmed_word = port_stem.stem(word)
            cleaned_words.append(stemmed_word)
    cleanform=' '.join(cleaned_words)
    return cleanform
batches = np.array_split(tweets_data, 16)
all_text = []
all_labels = []
count=1
for  parts in batches:
    print(f"Processing Batch {count}")
    count+=1
    parts['stemmed_content']=parts['tweet'].swifter.apply(clean_text)
    all_text.extend(parts['stemmed_content'].tolist())
    all_labels.extend(parts['positivity'].tolist())
print("Preprocessing Complete")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(all_text)
Y = np.array(all_labels)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=42)
model = LogisticRegression(solver='saga', max_iter=1000)
model.fit(X_train, Y_train)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
accuracy_training=accuracy_score(Y_train, train_pred)
accuracy_testing=accuracy_score(Y_test, test_pred)
print("Training Accuracy:", accuracy_score(Y_train, train_pred))
print("Testing Accuracy:", accuracy_score(Y_test, test_pred))
if(accuracy_training>accuracy_testing):
    print("Analysed for any problem with difference in accuracy and result is positive")
else:
    print("Accuracy of Training Data must be better than Test Acuuracy,not a viable result")
import pickle
f=open("result.pkl",'wb')
pickle.dump(model,f)
f.close()
b=open("vectorizer.pkl",'wb')
pickle.dump(vectorizer,b)
b.close()