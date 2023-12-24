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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

import gensim
from gensim.models import Word2Vec


nltk.download('punkt')