# from flask import Flask, render_template, request
# from Analyze1 import summarizer

# app = Flask(__name__, template_folder='templates')

# def process_text(text):
#     # Placeholder for your processing logic
#     return f"Processed: {text}"

# @app.route('/', methods=['GET', 'POST'])
# def process_input():
#     text_input = None
#     result = None

#     if request.method == 'POST':
#         try:
#             # Get text input from the form
#             text_input = request.form['textInput']

#             # Process the text using your machine learning model or function
#             result = summarizer(text_input)

#         except Exception as e:
#             # Handle any errors that might occur during processing
#             result = f"Error: {str(e)}"

#     return render_template('index1.html', text_input=text_input, result=result)

# if __name__ == '__main__':
#     app.run(debug=True, port=5002)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd

df = pd.read_csv('D:/aff/evi/my_evi/mail_data.csv')

new_df = df.where((pd.notnull(df)),'')

#using loc to locate the value is dataset
new_df.loc[new_df['Category']=='spam', 'Category',] = 0
new_df.loc[new_df['Category']=='ham', 'Category',] = 1

# saperating data for text and label
X = new_df['Message']
Y = new_df['Category']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# transform text data to feature vectorethat can be used as input in logistic regration model
# min_df is for to priority fo wrod should be greater then one here.
feature_ext = TfidfVectorizer(min_df =1, stop_words='english', lowercase = True)

x_train_feature = feature_ext.fit_transform(x_train)

x_test_feature = feature_ext.transform(x_test)


# convert y_trainf and y_test values as int

y_train = y_train.astype('int')
y_test = y_test.astype('int')

def non(text):
    input_data = [text]

    # convert text to feature vectors

    input_data_feature = feature_ext.transform(input_data)

    # mking prediction

    loaded_model = pickle.load(open('D:/aff/evi/my_evi/2_spam_checker_model.sav', 'rb'))

    prediction = loaded_model.predict(input_data_feature)
    print(prediction)

    if prediction[0] == 1:
      return "Not spam"
    else:
      return "Spam"

# i = input('enter value : ')
# print(non(i))