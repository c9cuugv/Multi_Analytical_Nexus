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


def analyze(text):
    blob = TextBlob(text)
    sentiment_polarity = round(blob.sentiment.polarity,2)
    print(sentiment_polarity)   # polarity : positive, negative or neutral (range -1 to 1)
    
    if sentiment_polarity >= 0.10:
        return "Positive 🤗"
    
    elif -0.10 < sentiment_polarity < 0.10:
        return "Neutral 😐"
    
    else:
        return "Negative 😤😔"

#     df = pd.read_csv('D:\aff\evi\my_evi\output.csv')

#     column_names =['target', 'id', 'date', 'flag', 'user', 'text']
#     df = pd.read_csv('D:\aff\evi\my_evi\output.csv', names=column_names, encoding='ISO-8859-1')

#     df['stemmed_content'] = df['text'].apply(stemming)
#     new_df = df.where((pd.notnull(df)),'')

#     #using loc to locate the value is dataset
#     new_df.loc[new_df['stemmed_content']=='positive', 'target',] = 0
#     new_df.loc[new_df['stemmed_content']=='negative', 'target',] = 1

#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

#     feature_ext = TfidfVectorizer(min_df =1, stop_words='english', lowercase = True)

#     x_train_feature = feature_ext.fit_transform(x_train)

#     x_test_feature = feature_ext.transform(x_test)

    
#     X = new_df['Message']
#     Y = new_df['Category']

#     y_train = y_train.astype('int')
#     y_test = y_test.astype('int')

#     def non(text):
#         input_data = [text]

#         # convert text to feature vectors

#         input_data_feature = feature_ext.transform(input_data)

#         # mking prediction

#         loaded_model = pickle.load(open('D:\aff\evi\my_evi\trained_model.sav', 'rb'))
#         decision_values = loaded_model.decision_function(input_data_feature)
#         probabilities = round(np.sig(decision_values)[0],3)
#         prediction = loaded_model.predict(input_data_feature)
#         print(prediction)

#         if probabilities<=0.25:
#             return 'Tweet is positive'
#         elif (probabilities > 0.3 and probabilities < 0.32):
#             return 'Tweet is Nutaral'
#         else:
#             return 'Tweet is negative'
        

#     return non(text1)


#     # loaded_model = pickle.load(open('D:\aff\evi\my_evi\trained_model.sav', 'rb'))
#     # vectorizer = TfidfVectorizer()
#     # Ana = stemming(text)
#     # Ana = str(Ana)
#     # Done = [Ana]
#     # Done = np.array(Done)

#     # Done = vectorizer.fit_transform(Done)
#     # Done.resize([1,461488])
#     # decision_values = loaded_model.decision_function(Done)
#     # probabilities = round(sigmoid(decision_values)[0],3)


#     # if sentiment_polarity >= 0.10:
#     #     return "Positive 🤗"
    
#     # elif -0.10 < sentiment_polarity < 0.10:
#     #     return "Neutral 😐"
    
#     # else:
#     #     return "Negative 😤😔"


# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.metrics import accuracy_score
# # from scipy.special import expit as sigmoid

#     # loaded_model = pickle.load(open('C:/Users/raval/OneDrive/Desktop/new1/project1/project UI/evi/my_evi/trained_model.sav', 'rb'))
#     # vectorizer = TfidfVectorizer()
#     # Ana = stemming(text)
#     # Ana = str(Ana)
#     # Done = [Ana]
#     # Done = np.array(Done)

#     # Done = vectorizer.fit_transform(Done)
#     # Done.resize([1,461488])
#     # decision_values = loaded_model.decision_function(Done)
#     # probabilities = round(sigmoid(decision_values)[0],3)

#     # if probabilities<=0.25:
#     #     return 'Tweet is positive'
#     # elif (probabilities > 0.3 and probabilities < 0.32):
#     #     return 'Tweet is Nutaral'
#     # else:
#     #     return 'Tweet is negative'


# name = "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer. You shoulda got David Carr of Third Day to do it. ;D"
# print(analyze(name))
