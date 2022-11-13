# Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

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

# Loading the Dataset
df = pd.read_csv("spam.csv")
print(df.head(),'\n')

# Adding Columns
df['length'] = df['Message'].map(lambda text: len(text))
df['label'] = df['Category'].map({'ham': 0, 'spam': 1})
print(df.head())

print(df.info(),'\n')

# Let’s visualize the distribution of Ham and Spam data.
sns.countplot(x=df['Category'])
plt.show()

# Average number of words
avg_words_len = round(sum([len(i.split()) for i in df['Message']])/len(df['Message']))
print("Average number of words :",avg_words_len,'\n')

# Total number of unique words
s = set()
for sent in df['Message']:
  for word in sent.split():
    s.add(word)
total_words_length = len(s)
print("Total number of unique words :",total_words_length,'\n')

# Splitting the data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.2, random_state=42)

# Shape of train and test data
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape,'\n')

# For each word in the email text, get the base form of the word and return the list of base words
def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]

# Function to apply the Count Vectorizer (BOW) and TF-IDF transforms to a set of input features
def features_transform(mail):
    # Get the bag of words for the mail text
    bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(X_train)
    # print(len(bow_transformer.vocabulary_))
    messages_bow = bow_transformer.transform(mail)

    # print sparsity value
    print('sparse matrix shape:', messages_bow.shape)
    print('number of non-zeros:', messages_bow.nnz)
    print('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))
    print()

    # Apply the TF-IDF transform to the output of BOW
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)
    # print(messages_tfidf.shape)

    # return result of transforms
    return messages_tfidf


# Transform training set features into a set of useful features to build models
train_features = features_transform(X_train)

# Transform test features to test the model performance
test_features = features_transform(X_test)


# Function which takes in y test value and y predicted value and prints the associated model performance metrics
def model_assessment(y_test , predicted_class):
    print('confusion matrix')
    print(confusion_matrix(y_test,predicted_class))
    print('accuracy')
    print(accuracy_score(y_test,predicted_class))
    print('precision')
    print(precision_score(y_test,predicted_class,pos_label='spam'))
    print('recall')
    print(recall_score(y_test,predicted_class,pos_label='spam'))
    print('f-Score')
    print(f1_score(y_test,predicted_class,pos_label='spam'))
    print('AUC')
    print(roc_auc_score(np.where(y_test=='spam',1,0),np.where(predicted_class == 'spam',1,0)))
    print()
    cm = confusion_matrix(y_test, predicted_class)
    sns.heatmap(cm, fmt='g', annot = True, cmap = "viridis" )
    plt.title('confusion matrix')
    plt.ylabel('expected label')
    plt.xlabel('predicted label')
    plt.show()

# 1.Naive Bayes Model :

# create and fit NB model
modelNB = MultinomialNB()
modelNB.fit(train_features, y_train)

# NB predictions
predicted_class_NB = modelNB.predict(test_features)

# Assess NB
model_assessment(y_test,predicted_class_NB)


# 2.Decision Tree Model:

# Create and fit tree model
model_tree = DecisionTreeClassifier()
model_tree.fit(train_features,y_train)

# Run model on test and print metrics
predicted_class_tree = model_tree.predict(test_features)
model_assessment(y_test,predicted_class_tree)


# 3.Support Vector Machine:

# Create and fit SVM model
model_svm = SVC()
model_svm.fit(train_features,y_train)

# Run model on test and print metrics
predicted_class_svm=model_svm.predict(test_features)
model_assessment(y_test,predicted_class_svm)


# 4. Random Forest

# Create and fit model
model_rf = RandomForestClassifier(n_estimators=20, criterion='entropy')
model_rf.fit(train_features,y_train)

# Run model on test and print metrics
predicted_class_rf = model_rf.predict(test_features)
model_assessment(y_test,predicted_class_rf)
