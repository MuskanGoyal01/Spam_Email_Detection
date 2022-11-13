# Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# Loading the Dataset
df = pd.read_csv("spam.csv")
print(df.head())

# Adding Columns
df['length'] = df['Message'].map(lambda text: len(text))
df['label'] = df['Category'].map({'ham': 0, 'spam': 1})
print(df.head())

print(df.info())

# Let’s visualize the distribution of Ham and Spam data.
sns.countplot(x=df['Category'])
plt.show()

# Average number of words
avg_words_len=round(sum([len(i.split()) for i in df['Message']])/len(df['Message']))
print("Average number of words :",avg_words_len)

# Total number of unique words
s = set()
for sent in df['Message']:
  for word in sent.split():
    s.add(word)
total_words_length=len(s)
print("Total number of unique words :",total_words_length)

# Splitting the data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.25, random_state=42)

# Shape of train and test data
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

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
    # print(len(bow_transformer.vocabulary))
    messages_bow = bow_transformer.transform(mail)
    # print sparsity value
    print('sparse matrix shape:', messages_bow.shape)
    print('number of non-zeros:', messages_bow.nnz)
    print('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))
    # Apply the TF-IDF transform to the output of BOW
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)
    # print(messages_tfidf.shape)
    # return result of transforms
    return messages_tfidf

# Transform training set features into a set of useful features to build models
train_features = features_transform(X_train)