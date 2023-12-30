import re
from pyarabic.araby import strip_tashkeel
import nltk 
from nltk import word_tokenize
from diacritization_evaluation import util
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from gensim.models import Word2Vec
#import tensorflow as tf

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Bidirectional, LSTM, Dense,Embedding

import gensim
from gensim.models import Word2Vec
import numpy as np

import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import random as rnd
import copy

from gensim.models import FastText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.optim as optim



nltk.download('punkt')