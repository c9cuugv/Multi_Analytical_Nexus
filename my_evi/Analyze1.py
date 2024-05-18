# # from transformers import PegasusTokenizer
# # from transformers import PegasusForConditionalGeneration
# # from transformers import pipeline
# import pickle
# import torch

# # model_name = "google/pegasus-xsum"
# # token = "<your_access_token>"
# def summarizer(article):
#     with open('D:/aff/evi/my_evi/my_model.sav', 'rb') as f:
#         loaded_model = pickle.load(f)

#     # Set the model to evaluation mode
#     loaded_model.eval()

#     # Perform inference using the loaded model
#     input_tensor = torch.tensor(article)  # Replace with your actual input data
#     output = loaded_model(input_tensor)

#     return output


# import spacy
# import pytextrank

# def summarizer(article):
#     nlp = spacy.load("en_core_web_lg")
#     nlp.add_pipe('textrank')
#     doc= nlp(article)
#     for sent in doc._.textrank.summary(limit_phrases=2, limit_sentences=5):
#         return sent


# ______________________________________
# # import spacy
# import pytextrank

# nlp = spacy.load("en_core_web_lg")
# nlp.add_pipe('textrank')

# def summarizer(article):
#     doc = nlp(article)
#     return [sent for sent in doc._.textrank.summary(limit_phrases=2, limit_sentences=5)]


# print(summarizer('''The economy of India has transitioned from a mixed planned economy to a mixed middle-income developing social market economy with notable public sector in strategic sectors.[48] It is the world's fifth-largest economy by nominal GDP and the third-largest by purchasing power parity (PPP); on a per capita income basis, India ranked 139th by GDP (nominal) and 127th by GDP (PPP).[49] From independence in 1947 until 1991, successive governments followed Soviet model and promoted protectionist economic policies, with extensive Sovietization, state intervention, demand-side economics, natural resources, bureaucrat driven enterprises and economic regulation. This is characterised as dirigism, in the form of the Licence Raj.[50][51] The end of the Cold War and an acute balance of payments crisis in 1991 led to the adoption of a broad economic liberalisation in India and indicative planning.[52][53] Since the start of the 21st century, annual average GDP growth has been 6% to 7%.[48] The economy of the Indian subcontinent was the largest in the world for most of recorded history up until the onset of colonialism in early 19th century.[54][55][56]

# Nearly 70% of India's GDP is driven by domestic consumption;[57] country remains the world's sixth-largest consumer market.[58] Apart from private consumption, India's GDP is also fueled by government spending, investments, and exports.[59] In 2022, India was the world's 6th-largest importer and the 9th-largest exporter.[60] India has been a member of the World Trade Organization since 1 January 1995.[61] It ranks 63rd on the Ease of doing business index and 40th on the Global Competitiveness Index.[62] With 476 million workers, the Indian labour force is the world's second-largest.[21] India has one of the world's highest number of billionaires and extreme income inequality.[63][64]'''))

import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
 
def read_article(file_name):
    # file = open(file_name, "r")
    filedata = [file_name]  
    article = filedata[0].split(". ")
    sentences = []

    for sentence in article:
        # print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix



