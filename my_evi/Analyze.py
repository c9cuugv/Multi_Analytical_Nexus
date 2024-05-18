# import pickle
# import numpy as np
# import pandas as pd
# import re
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
from textblob import TextBlob

# import nltk
# nltk.download('stopwords')

# port_stem=PorterStemmer()
# def stemming(content):
#             stemmed_con = re.sub('[^a-zA-Z]',' ', content) # remove other than latters
#             stemmed_con = stemmed_con.lower()
#             stemmed_con = stemmed_con.split()  # spliting into the list
#             stemmed_con = [port_stem.stem(word) for word in stemmed_con if not word in stopwords.words('english')]
#             stemmed_con = ' '.join(stemmed_con)

#             return stemmed_con

# df = pd.read_csv('D:\aff\evi\my_evi\output.csv')

# column_names =['target', 'id', 'date', 'flag', 'user', 'text']
# df = pd.read_csv('D:\aff\evi\my_evi\output.csv', names=column_names, encoding='ISO-8859-1')

# df['stemmed_content'] = df['text'].apply(stemming)
# new_df = df.where((pd.notnull(df)),'')


# def analyze(text):
#     blob = TextBlob(text)
#     sentiment_polarity = round(blob.sentiment.polarity,2)
#     print(sentiment_polarity)   # polarity : positive, negative or neutral (range -1 to 1)
    
#     if sentiment_polarity >= 0.10:
#         return "Positive 🤗"
    
#     elif -0.10 < sentiment_polarity < 0.10:
#         return "Neutral 😐"
    
#     else:
#         return "Negative 😤😔"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
import re

port_stem=PorterStemmer()

def stemming(content):
  stemmed_con = re.sub('[^a-zA-Z]',' ', content) # remove other than latters
  stemmed_con = stemmed_con.lower()
  stemmed_con = stemmed_con.split()  # spliting into the list
  stemmed_con = [port_stem.stem(word) for word in stemmed_con if not word in stopwords.words('english')]
  stemmed_con = ' '.join(stemmed_con)

  return stemmed_con

df = pd.read_csv('D:/aff/evi/my_evi/output (2).csv')

column_names =['target', 'id', 'date', 'flag', 'user', 'text']
df = pd.read_csv('D:/aff/evi/my_evi/output (2).csv', names=column_names, encoding='ISO-8859-1')

df['stemmed_content'] = df['text'].apply(stemming)

new_df = df.where((pd.notnull(df)),'')

#using loc to locate the value is dataset
new_df.loc[new_df['stemmed_content']=='positive', 'Category',] = 0
new_df.loc[new_df['stemmed_content']=='negative', 'Category',] = 1

# saperating data for text and label
X = new_df['stemmed_content']
Y = new_df['target']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

feature_ext = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

x_train_feature = feature_ext.fit_transform(x_train)
x_test_feature = feature_ext.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

def analyze(text):
    loaded_model = pickle.load(open('D:/aff/evi/my_evi/trained_model.sav', 'rb'))

    input_data = [text]

    # Combine the training and input data
    combined_data = x_train + input_data

    # Fit the feature extractor on the combined data
    feature_ext = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    x_combined_feature = feature_ext.fit_transform(combined_data)

    # Retrain the model on the combined features
    loaded_model.fit(x_combined_feature, y_train)

    # Make predictions on the input data
    input_data_feature = feature_ext.transform(input_data)
    prediction = loaded_model.predict(input_data_feature)

    # convert text to feature vectors

    input_data_feature = feature_ext.transform(input_data)

    # mking prediction

    prediction = loaded_model.predict_proba(input_data_feature)
    print(prediction[0][0])
    print(prediction)

    if prediction[0][1] >= prediction[0][0]:
        return "Negative 😤😔"
    elif prediction[0][0]==prediction[0][1]:
        return "Neutral 😐"
    else:
        return "Positive 🤗"