# Import Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# Loading Dataset
df = pd.read_csv('spam.csv')

# Splitting the data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.2, random_state=42)

# Create an ensemble of 3 models
estimators = []
estimators.append(('Naive Bayes', MultinomialNB()))
estimators.append(('CART', DecisionTreeClassifier()))
estimators.append(('SVM', SVC()))
estimators.append(('RFC', RandomForestClassifier()))

# Create the Ensemble Model
ensemble = VotingClassifier(estimators)

# Make preprocess Pipeline
pipe = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('model', ensemble)
])

# Train the model
pipe.fit(X_train, y_train)

# Test Accuracy
print(f"Accuracy: {round(pipe.score(X_test, y_test), 3) * 100} %")
