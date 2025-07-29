import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
with open("result.pkl", 'rb') as f:
    model = pickle.load(f)
with open("vectorizer.pkl", 'rb') as f:
    vectorizer = pickle.load(f)
print('''
Predict the nature of a tweet with the help of machine learning...

With over 1.5 million tweets analyzed through logistic regression,
we evaluate your sentence using stemmed and weighted word frequencies.
Achieves over 75% accuracy!
*__________________________________________________________________*


''')

tweet = input("Enter the tweet: ")

port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))
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
cleaned_tweet = clean_text(tweet)
vectorized_tweet = vectorizer.transform([cleaned_tweet])
prediction = model.predict(vectorized_tweet)
if prediction[0] == 0:
    print("❌ Negative tweet")
else:
    print("✅ Positive tweet")



