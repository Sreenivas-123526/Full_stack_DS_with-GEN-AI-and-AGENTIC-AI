import os
os.chdir(r'C:\Users\hp\Desktop')

import xml.etree.ElementTree as ET

tree=ET.parse("xml version.xml")

root=tree.getroot()

root=ET.tostring(root,encoding='utf8').decode('utf8')

import re,string,unicodedata
import nltk

from bs4 import BeautifulSoup
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer,WordNetLemmatizer

def strip_xml(text):
    soup=BeautifulSoup(text,'xml')
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def noise_text(text):
    text=strip_xml(text)
    text=remove_between_square_brackets(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

final=noise_text(root)
