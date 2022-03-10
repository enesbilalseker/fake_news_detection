from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import itertools
from sklearn import metrics
from nltk import tokenize
from nltk.corpus import stopwords
import nltk
import string
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# load the data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# flag to track fake and real
fake['target'] = 'fake'
true['target'] = 'true'

# concatenate the data frames
data = pd.concat([fake, true]).reset_index(drop=True)

# shuffle data to prevent bias
data = shuffle(data)
data = data.reset_index(drop=True)

### data cleansing ###
# remove data
data.drop(["date"], axis=1, inplace=True)
# remove title
data.drop(["title"], axis=1, inplace=True)
# convert text to lowercase
data['text'] = data['text'].apply(lambda x: x.lower())


# remove punctuation
def remove_punctuation(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str


# remove the stopwords a, the, is, are
data['text'] = data['text'].apply(remove_punctuation)
nltk.download('stopwords')
stop = stopwords.words('english')

data['text'] = data['text'].apply(lambda x: ' '.join(
    [word for word in x.split() if word not in (stop)]))

### data exploration ###

# how many fake articles
print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind="bar")
plt.show()


# most frequent words counter
token_space = tokenize.WhitespaceTokenizer()


def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame(
        {"Word": list(frequency.keys()), "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns="Frequency", n=quantity)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df_frequency, x="Word", y="Frequency", color='blue')
    ax.set(ylabel="Count")
    plt.xticks(rotation='vertical')
    plt.show()


counter(data[data["target"] == "fake"], "text", 20)
counter(data[data["target"] == "true"], "text", 20)

### Modeling ###

# The modeling process will consist of vectorizing the corpus stored in the “text” column,
# then applying TF-IDF, and finally a classification machine learning algorithm.
# Pretty standard in text analytics and NLP.
# For modeling, we have this function to plot the confusion matrix of the models:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# split the data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data.target, test_size=0.2, random_state=42)

############ Logistic Regression ############
# vectorizing and applying TF-IDF
pipe = Pipeline([('vect', CountVectorizer()), ('tfidf',
                TfidfTransformer()), ('model', LogisticRegression())])

# fiting the model
model = pipe.fit(X_train, y_train)

# Accuracy
prediction = model.predict(X_test)
print("Logistic Regression Accuracy: {}%".format(
    round(accuracy_score(y_test, prediction)*100, 2)))

# confusion matrix
cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])


############ Decision Tree Classifier ############
# Vectorizing and applying TF-IDF
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion='entropy',
                                                  max_depth=20,
                                                  splitter='best',
                                                  random_state=42))])
# Fitting the model
model = pipe.fit(X_train, y_train)

# Accuracy
prediction = model.predict(X_test)
print("Decision Tree Classifier accuracy: {}%".format(
    round(accuracy_score(y_test, prediction)*100, 2)))

cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])


############ Random Forest Classifier ############

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])

model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Random Forest Classifier Accuracy: {}%".format(
    round(accuracy_score(y_test, prediction)*100, 2)))

cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])
