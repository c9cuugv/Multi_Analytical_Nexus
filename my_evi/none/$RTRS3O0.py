# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow import datasets, layers, models

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# train_images,test_images = train_images / 255, test_images / 255

# class_name =['Plane','Car','Bird','Cat','Deer','Dog','Frog', 'Horse', 'Ship', 'Truck']

# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_images[i],camp=plt.cm.binary)
#     plt.xlabel(class_name[train_labels[i][0]])

# plt.show()

# train_images = train_images[:20000]
# train_labels = train_labels[:20000]
# train_images = train_images[:4000]
# train_images = test_labels[:4000]

# model = models.Sequential()
# model.add(layers.Conv2D(32,(3,3),activation = 'relu', input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(train_images,train_labels,epochs=10,validation_data=(test_images,test_labels))

# import tkinter as tk
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
# from textblob import TextBlob
# from newspaper import Article

# stopWords = set(stopwords.words("english")) 

# string = str(input('Enter content or url to summarize : '))
# words = word_tokenize(string) 
# x = string.split()
# for i in x:
#     if i.find("https:")==0 or  i.find("http:") == 0:
#         ana = Article(string)
#         ana.download()
#         ana.parse()
#         ana.nlp()
#         print(f'title : {ana.title}\n')
#         # print(f'title : {ana.authors}\n')
#         print(f'summary : {ana.summary}\n')
#         # print(f'title : {ana.publish_date}')
#     else:
#         freqTable = dict() 
# for word in words: 
#     word = word.lower() 
#     if word in stopWords: 
#         continue
#     if word in freqTable: 
#         freqTable[word] += 1
#     else: 
#         freqTable[word] = 1
   
#         # Creating a dictionary to keep the score 
#         # of each sentence 
#         sentences = sent_tokenize(string) 
#         sentenceValue = dict() 
        
#         for sentence in sentences: 
#             for word, freq in freqTable.items(): 
#                 if word in sentence.lower(): 
#                     if sentence in sentenceValue: 
#                         sentenceValue[sentence] += freq 
#                     else: 
#                         sentenceValue[sentence] = freq 
        
        
        
#         sumValues = 0
#         for sentence in sentenceValue: 
#             sumValues += sentenceValue[sentence] 
        
#         # Average value of a sentence from the original text 
        
#         average = int(sumValues / len(sentenceValue)) 
        
#         # Storing sentences into our summary. 
#         summary = '' 
#         for sentence in sentences: 
#             if sentence in sentenceValue: 
#                 summary += " " + sentence 

# print(f'summary :{summary}') 



import spacy
import pytextrank
nlp = spacy.load("en_core_web_lg")
def summarizer(article):
    nlp.add_pipe('textrank')
    doc = nlp(article)
    for sent in doc._.textrank.summary(limit_phrases=2, limit_sentences=1):
        sents = sent




