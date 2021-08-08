from flask import Flask, render_template, current_app, request
from pickle import load 
import string
import os
import torch
import numpy as np
from operator import itemgetter
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from model import Model
from config import config

app = Flask(__name__)

def load_features(filename):
  # load all features
  all_features = load(open(filename, 'rb'))
  # filter features
  images = list(all_features.keys())
  features = {k: all_features[k] for k in images}
  return features

def query_cleaner(cap):
  # prepare translation table for removing punctuation
  table = str.maketrans('', '', string.punctuation)
  # tokenize
  cap = cap.split()
  # convert to lower case
  cap = [word.lower() for word in cap]
  # remove punctuation from each token
  cap = [w.translate(table) for w in cap]
  # remove hanging 's' and 'a'
  cap = [word for word in cap if len(word)>1]
  # remove tokens with numbers in them
  cap = [word for word in cap if word.isalpha()]
  # remove stopwords
  cap = [word for word in cap if word not in current_app.stopwords]
  # store as string
  cap =  ' '.join(cap)
  return cap
 

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
      query = request.form['search']
    else:
      query = 'cars on the roads'
    caption_tokenized, caption_array, failed_array = {}, {}, {}
    tokens, token_arrays = [], []

    tokens = query_cleaner(query).split(' ')
    query2vec = torch.FloatTensor([np.sum([current_app.word2vec[i] for i in tokens], axis=0)])
    query_embedding = current_app.model.forward_caption(query2vec)

    dist_data = {}
    for k, v in current_app.image_vectors.items():
      dist = torch.cdist(query_embedding, torch.FloatTensor(v), p=2)
      dist_data[k] = dist

    #knn_1 = list(dict(sorted(dist_data.items(), key=itemgetter(1))[:1]).keys())
    #knn_5 = list(dict(sorted(dist_data.items(), key=itemgetter(1))[:5]).keys())
    knn_10 = list(dict(sorted(dist_data.items(), key=itemgetter(1))[:10]).keys())
    return render_template('index.html', query=query, images=knn_10)


if __name__ == '__main__':
   dir_path = os.getcwd()

   with app.app_context():
    current_app.model = Model()
    current_app.model.load_state_dict(torch.load(dir_path+'/data/model.pth'))
    current_app.model.eval()
    current_app.image_vectors = load_features(dir_path+'/data/flask_embeddings.pkl')
    current_app.word2vec = KeyedVectors.load(dir_path+'/data/glove.6B.300d.gs', mmap='r')
    word_file = open(dir_path+"/data/stopwords.txt", "r")
    stopwords = word_file.readlines()
    current_app.stopwords = [s.rstrip() for s in stopwords]
    word_file.close()

   app.run(debug = True)

   