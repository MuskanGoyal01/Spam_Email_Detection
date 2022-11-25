# Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from texttable import Texttable
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Loading the Dataset
df = pd.read_csv("../Statics/spam.csv")
print(df.head(), '\n')

# Adding Columns
df['length'] = df['Message'].map(lambda text: len(text))
df['label'] = df['Category'].map({'ham': 0, 'spam': 1})
print(df.head())

print(df.info(), '\n')

# Let’s visualize the distribution of Ham and Spam data.
sns.countplot(x=df['Category'])
plt.show()

# Average number of words
avg_words_len = round(sum([len(i.split()) for i in df['Message']]) / len(df['Message']))
print("Average number of words :", avg_words_len, '\n')

# Total number of unique words
s = set()
for sent in df['Message']:
    for word in sent.split():
        s.add(word)
total_words_length = len(s)
print("Total number of unique words :", total_words_length, '\n')

# Splitting the data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.2, random_state=42)

# Shape of train and test data
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, '\n')


# For each word in the email text, get the base form of the word and return the list of base words
def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]


# Functions to apply the Count Vectorizer (BOW) and TF-IDF to a set of input features
def feature1(mail):
    # Get the bag of words for the mail text
    bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(X_train)
    # print(len(bow_transformer.vocabulary_))
    messages_bow = bow_transformer.transform(mail)

    # print sparsity value
    print('sparse matrix shape:', messages_bow.shape)
    print('number of non-zeros:', messages_bow.nnz)
    print('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))
    print('\n')

    return messages_bow


def feature2(mail):
    tfidf_transformer = TfidfVectorizer().fit(X_train)
    messages_tfidf = tfidf_transformer.transform(mail)
    # print sparsity value
    print('sparse matrix shape:', messages_tfidf.shape)
    print('number of non-zeros:', messages_tfidf.nnz)
    print('sparsity: %.2f%%' % (100.0 * messages_tfidf.nnz / (messages_tfidf.shape[0] * messages_tfidf.shape[1])))
    print('\n')

    # return result of transforms
    return messages_tfidf


# Transform training set features into a set of useful features to build models
print("Using Feature 1 for training data")
train_feature1 = feature1(X_train)
print("Using Feature 2 for training data")
train_feature2 = feature2(X_train)

# Transform test features to test the model performance
print("Using Feature 1 for testing data")
test_feature1 = feature1(X_test)
print("Using Feature 2 for testing data")
test_feature2 = feature2(X_test)


# Function which takes in y test value and y predicted value and prints the associated model performance metrics
def model_assessment(y_test, predicted_class, model, color):
    confusion_matrix_results = confusion_matrix(y_test, predicted_class)
    accuracy_results = accuracy_score(y_test, predicted_class)
    precision_results = precision_score(y_test, predicted_class, pos_label='spam')
    recall_result = recall_score(y_test, predicted_class, pos_label='spam')
    f_score_results = f1_score(y_test, predicted_class, pos_label='spam')
    auc_results = roc_auc_score(np.where(y_test == 'spam', 1, 0), np.where(predicted_class == 'spam', 1, 0))

    table = Texttable()
    table.add_rows(
        [
            ["Confusion Matrix", confusion_matrix_results],
            ["", ""],
            ["Accuracy", accuracy_results],
            ["Precision", precision_results],
            ["Recall", recall_result],
            ["F-Score", f_score_results],
            ["AUC", auc_results]
        ]
    )
    table.set_deco(Texttable.VLINES | Texttable.BORDER)
    print(model.upper())
    print(table.draw())
    print('\n')

    cm = confusion_matrix(y_test, predicted_class)
    sns.heatmap(cm, fmt='g', annot=True, cmap=color)
    plt.title(f'Confusion Matrix for {model}')
    plt.ylabel('Expected label')
    plt.xlabel('Predicted label')
    plt.show()

    return confusion_matrix_results, accuracy_results, precision_results, recall_result, f_score_results, auc_results


# Function for making Perfomance Metrics Table of All Models
def final_data_table(nbm, dtm, svm, rfm,num):
    headers = ["Model Name", "Naive Bayes Model", "Decision Tree Model", "SVM", "Random Forest Model"]
    rows = ["Confusion Matrix", "Accuracy", "Precision", "Recall", "F-Score", "AUC"]

    data = [[rows[i], nbm[i], dtm[i], svm[i], rfm[i]] for i in range(len(rows))]

    print(f'Comparison Table for Models with feature{num}')
    print(tabulate(data, headers=headers, tablefmt='pretty'))


# 1.Naive Bayes Model :

modelNB = MultinomialNB()  # Create and fit NB model

modelNB.fit(train_feature1, y_train)
predicted_class_NB1 = modelNB.predict(test_feature1)  # NB predictions
nbm_model1 = model_assessment(y_test, predicted_class_NB1, "Naive Bayes Model 1", "viridis")  # Assess NB

modelNB3 = modelNB.fit(train_feature2, y_train)
predicted_class_NB2 = modelNB.predict(test_feature2)  # NB predictions
nbm_model2 = model_assessment(y_test, predicted_class_NB2, "Naive Bayes Model 2", "viridis")  # Assess NB


# 2.Decision Tree Model:

model_tree = DecisionTreeClassifier()  # Create and fit tree model

model_tree.fit(train_feature1, y_train)
predicted_class_tree1 = model_tree.predict(test_feature1)  # Run model on test and print metrics
dtm_model1 = model_assessment(y_test, predicted_class_tree1, "Decision Tree Model 1", "Greens_r")

model_tree.fit(train_feature2, y_train)
predicted_class_tree2 = model_tree.predict(test_feature2)  # Run model on test and print metrics
dtm_model2= model_assessment(y_test, predicted_class_tree2, "Decision Tree Model 2", "Greens_r")


# 3.Support Vector Machine (SVM) Model:

model_svm = SVC()  # Create and fit SVM model

model_svm.fit(train_feature1, y_train)
predicted_class_svm1 = model_svm.predict(test_feature1)  # Run model on test and print metrics
svm_model1 = model_assessment(y_test, predicted_class_svm1, "SVM Model 1", "Reds_r")

model_svm.fit(train_feature2, y_train)
predicted_class_svm2 = model_svm.predict(test_feature2)  # Run model on test and print metrics
svm_model2 = model_assessment(y_test, predicted_class_svm2, "SVM Model 2", "Reds_r")


# 4. Random Forest Model

model_rf = RandomForestClassifier(n_estimators=20, criterion='entropy')  # Create and fit model

model_rf.fit(train_feature1, y_train)
predicted_class_rf1 = model_rf.predict(test_feature1)  # Run model on test and print metrics
rfm_model1 = model_assessment(y_test, predicted_class_rf1, "Random Forest Model 1", "Blues_r")

model_rf.fit(train_feature2, y_train)
predicted_class_rf2 = model_rf.predict(test_feature2)  # Run model on test and print metrics
rfm_model2 = model_assessment(y_test, predicted_class_rf2, "Random Forest Model 2", "Blues_r")


# Final Comparison Table for Models
final_data_table(nbm_model1, dtm_model1, svm_model1, rfm_model1, '1')
final_data_table(nbm_model2, dtm_model2, svm_model2, rfm_model2, '2')

