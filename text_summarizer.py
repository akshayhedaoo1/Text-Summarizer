import nltk
from nltk.corpus import stopwords
import numpy as np
from nltk.cluster.util import cosine_distance
import networkx
import streamlit as st

st.header("Text Summarizer")

with st.form(key='my_form'):
    text_input = st.text_input(label='Add Paragraph here')
    submit_button = st.form_submit_button(label='Submit')

uploaded_file = st.file_uploader("Choose File here")

sw = open(r"C:\Users\welcome\Data Science\Deep Learning\NLP\Text Summarizer\stopwords.txt", 'r')
stopwords =  sw.read().split('\n')

def read_para(para):
    article = para.split('.')
    sentences = []
    for sentence in article:
        if len(sentence) == 0:
            pass
        else:
            sentences.append(sentence.replace("[^a-zA-Z]", ' '))
    return sentences

def read_data(file_name):
  string = file_name.read().decode('utf-8')
  article = string.split('.')
  sentences = []

  for sentence in article:
    if len(sentence) == 0:
        pass
    else:
        sentences.append(sentence.replace("[^a-zA-Z]", ' '))
  return sentences

def sentence_similarity(sent1, sent2, stopwords = None):
  if stopwords == None:
    stopwords = []
  
  sent1 = [w.lower() for w in sent1]
  sent2 = [w.lower() for w in sent2]

  all_words = list(set(sent1 + sent2))

  vector1 = [0] * len(all_words)
  vector2 = [0] * len(all_words)

  # constructing vector for 1st sentence
  for w in sent1:
    if w not in stopwords:
      vector1[all_words.index(w)] += 1
  # constructing vector for 2nd sentence
  for w in sent2:
    if w not in stopwords:
      vector2[all_words.index(w)] += 1

  return 1 -cosine_distance(vector1, vector2)

def create_similarity_matrix(sentences, stopwords):
  similarity_matrix = np.zeros([len(sentences), len(sentences)])
  for i in range(len(sentences)):
    for j in range(len(sentences)):
      if i == j:
        continue
      similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stopwords)
  return similarity_matrix

def generate_summary(file):
  summarize_text = []

  # Step 1 : Read File and convert to list of sentences
  sentences = read_data(file)

  # Step 2 : Generate Similarity Matrix
  similar_matrix = create_similarity_matrix(sentences, stopwords)

  # Step 3 : Rank the sentences in similarity matrix
  sm_graph = networkx.from_numpy_array(similar_matrix)
  scores = networkx.pagerank(sm_graph)
  
  # Step 4 : Sort the rank and pick top sentences
  rank = sorted(((scores[i], s) for i, s in enumerate (sentences)), reverse=True)
  
  top = len(sentences)// 3
  for idx in range(top):
    summarize_text.append(rank[idx][1])

  summarize_text = ". ".join(summarize_text)
  return summarize_text

def generate_summary2(para):
  summarize_text = []

  # Step 1 : Read File and convert to list of sentences
  sentences = read_para(para)

  # Step 2 : Generate Similarity Matrix
  similar_matrix = create_similarity_matrix(sentences, stopwords)

  # Step 3 : Rank the sentences in similarity matrix
  sm_graph = networkx.from_numpy_array(similar_matrix)
  scores = networkx.pagerank(sm_graph)
  
  # Step 4 : Sort the rank and pick top sentences
  rank = sorted(((scores[i], s) for i, s in enumerate (sentences)), reverse=True)
  
  top = len(sentences)// 3
  for idx in range(top):
    summarize_text.append(rank[idx][1])

  summarize_text = ". ".join(summarize_text)
  return summarize_text

try:
    if uploaded_file is not None:
         st.success(generate_summary(uploaded_file))
except FileNotFoundError:
    st.error('File not found.')    

if submit_button:
    st.success(generate_summary2(text_input))
