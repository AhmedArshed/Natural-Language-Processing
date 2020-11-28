import pandas as pd, numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string
df_train = pd.read_csv("Tweets.csv")
df_train["negativereason"].fillna("unknown", inplace=True)
df_train["tweet_location"].fillna("unknown", inplace=True)
df_train["text"].fillna("unknown", inplace=True)
text_train = df_train.iloc[: , 6].values
labels_train = df_train.iloc[: , [1,4]].values
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
vect = TfidfVectorizer(use_idf = False ,tokenizer=tokenize,smooth_idf = True )
matrix_x = vect.fit_transform(text_train)
DT = DecisionTreeClassifier()
weight = DT.fit(matrix_x, labels_train)

text_input = input("Enter the Text: ")
text = []
text.append(text_input)
pre_x = vect.transform(text)
labels = weight.predict(pre_x)
print(labels)