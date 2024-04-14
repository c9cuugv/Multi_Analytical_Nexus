# from flask import Flask, render_template

# app = Flask(__name__, template_folder='templates', static_folder='static')

# @app.route('/')
# def index():
#     return render_template('new.html')

# if __name__ == "__main__":
#     app.run(debug=True)

import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from Analyze1 import read_article
from Analyze1 import build_similarity_matrix

def generate_summary(file_name):
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    # print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(0, 2):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize text
    # print("Summarize Text: \n", ". ".join(summarize_text))
    a = ". ".join(summarize_text)
    return a

# # let's begin
# print(generate_summary('msft.txt'))