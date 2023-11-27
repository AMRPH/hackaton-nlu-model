from joblib import load
import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel
import pymorphy3
from razdel import tokenize
import nltk
from nltk.corpus import stopwords

class GModel:
    def __init__(self):
        nltk.download("stopwords")
        self.russian_stopwords = stopwords.words("russian")

        self.morph = pymorphy3.MorphAnalyzer()

        self.stopTags = ['PNCT', 'NUMB', 'UNKN', 'LATN', 'ROMN']

        self.model = load('group_model.joblib')

        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
        self.emb_model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
    
    def get_normalize_text(self, s_list: []):
        return self.__get_embeddings(s_list)
        
    
    def __get_embeddings(self, s_list: []):
        res = []
        for s in s_list:
            s = self.__preprocess_text(s)
            encoded_input = self.tokenizer(s, padding=True, truncation=True, max_length=256, return_tensors='pt')
            with torch.no_grad():
                model_output = self.emb_model(**encoded_input)
            emb = model_output.pooler_output
            res.append((emb)[0].numpy())

        return np.asarray(res)
    
    def __preprocess_text(self, text):
        tokens = tokenize(text.lower())
        tokens = [self.morph.parse(token.text)[0].normal_form for token in tokens 
                  if self.morph.parse(token.text)[0].normal_form not in self.russian_stopwords and 
                  not any(tag in self.morph.parse(token.text)[0].tag for tag in self.stopTags)]
        text = " ".join(tokens) 
        return text
    
    def predict(self, x):
        return self.model.predict(x)
    
    def test_embeddings(self):
        return np.load('embeddings_test.npy')

class TModel:
    def __init__(self):
        nltk.download("stopwords")
        self.russian_stopwords = stopwords.words("russian")

        self.morph = pymorphy3.MorphAnalyzer()

        self.stopTags = ['PNCT', 'NUMB', 'UNKN', 'LATN', 'ROMN']

        self.model = load('theme_model.joblib')

        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
        self.emb_model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
    
    def get_normalize_text(self, s_list: []):
        return self.__get_embeddings(s_list)
        
    
    def __get_embeddings(self, s_list: []):
        res = []
        for s in s_list:
            s = self.__preprocess_text(s)
            encoded_input = self.tokenizer(s, padding=True, truncation=True, max_length=256, return_tensors='pt')
            with torch.no_grad():
                model_output = self.emb_model(**encoded_input)
            emb = model_output.pooler_output
            res.append((emb)[0].numpy())

        return np.asarray(res)
    
    def __preprocess_text(self, text):
        tokens = tokenize(text.lower())
        tokens = [self.morph.parse(token.text)[0].normal_form for token in tokens 
                  if self.morph.parse(token.text)[0].normal_form not in self.russian_stopwords and 
                  not any(tag in self.morph.parse(token.text)[0].tag for tag in self.stopTags)]
        text = " ".join(tokens) 
        return text
    
    def predict(self, x):
        return self.model.predict(x)
    
    def test_embeddings(self):
        return np.load('embeddings_test.npy')
