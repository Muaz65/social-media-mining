
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords

tweets_data_path = 'Tweets.csv'
tweets = pd.read_csv(tweets_data_path, header=0)
df = tweets.copy()[['airline_sentiment', 'text']]
sentence=df['text'] 
y_train=df['airline_sentiment']



max=0.00
n='negative'
p='positive'
u='neutral'
prediction = []
for sentences in sentence:
    #print (sentences)
    analyzer = SentimentIntensityAnalyzer()
    result= analyzer.polarity_scores(sentences)
    max=result['pos']
    neg=result['neg']
    neu=result['neu']
    if(max>neg):
      if(max>neu):
        prediction.append(p)
      else:
        prediction.append(u)
    else:
      if(neg>neu):
        prediction.append(n)
      else:
        prediction.append(u)   
    
   # print ("\n\t" + str(result))



print(classification_report(y_train, prediction))
