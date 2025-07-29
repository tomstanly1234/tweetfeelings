# tweetfeelings
An ML model which predicts the mood of a tweet by learning previous data using logistic regression
# Tweet Sentiment Classifier using Logistic Regression

This project is a simple yet powerful **Machine Learning-based sentiment analysis tool** that predicts whether a tweet expresses a **positive** or **negative** sentiment.

It uses the **Sentiment140 dataset**, which contains **1.6 million pre-labeled tweets**, and applies **Logistic Regression** after preprocessing the data using stemming, stopword removal, and TF-IDF vectorization.

---

## 📁 Dataset

The model was trained on the [**Sentiment140**](http://help.sentiment140.com/for-students) dataset — a popular open-source dataset containing:

- **1,600,000 tweets**
- **2 sentiment labels**: 0 = Negative, 4 = Positive (converted to 0/1)
- A file size of approximately **256 MB**
- Freely available with a quick Google search (`training.1600000.processed.noemoticon.csv`)

---

## ⚙️ Technologies Used

- **Python**
- **Scikit-learn**
- **NLTK**
- **Pandas, NumPy**
- **Swifter** (for fast dataframe operations)
- **TF-IDF Vectorization**
- **Logistic Regression** (via `sklearn.linear_model`)

---

## 💡 Features

- Preprocessing of raw tweet text: punctuation removal, lowercasing, stopword removal, stemming
- Vectorization using **TF-IDF**
- Classification using **Logistic Regression**
- Accuracy: Achieves around **75%** on test data
- Interactive terminal interface to test your own tweets

---

## 🧪 Inspiration

> This project takes inspiration from several online tutorials, including [GeeksforGeeks](https://www.geeksforgeeks.org/), to understand the fundamentals of text classification and logistic regression using scikit-learn.

---

## 🚀 How to Use

1. Download the Sentiment140 dataset (`training.1600000.processed.noemoticon.csv`)
2. Run the training script (`train_model.py`)
3. After training, run `predict_tweet.py` to classify your own tweet.

---

## 📦 Files

- `train_model.py` – Trains the model and saves the `result.pkl` and `vectorizer.pkl`
- `predict_tweet.py` – Loads the model and vectorizer, and predicts sentiment for user input
- `README.md` – Project overview and instructions

---

## 📌 Note

This is a simple baseline implementation for educational purposes. For production-grade sentiment analysis, consider:
- More advanced models (e.g., SVMs, Neural Networks)
- Pretrained transformers (e.g., BERT)
- Handling emojis, hashtags, slang, etc.

---

## 📄 License

This project is for educational use. The Sentiment140 dataset is publicly available and credited to its original creators.


