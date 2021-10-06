import nltk
import gensim.corpora as corpora
import gensim
import re
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import dictionary as dict

def Predict(word,numword):

    word          = re.sub(r'[^\w\s]', '', word).lower()

    stop_words    = set(stopwords.words('english'))
    word_tokens   = word_tokenize(word)

    filtered_word = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_word.append(w)

    lm = nltk.WordNetLemmatizer()

    def lemmatize(data):
       text = [lm.lemmatize(word) for word in data]
       return text

    word3 = lemmatize(filtered_word)
    
    for word in list(word3):  # iterating on a copy since removing will mess things up
        if word in dict.phrase:
            word3.remove(word)

    word3 = [word3]

    id2word = corpora.Dictionary(word3)

    corpus = [id2word.doc2bow(text) for text in word3]

    num_topics = 1

    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)

    opt = lda_model.print_topic(topicno=0, topn=numword).split("+")
    rtn = "["
    for x in opt:
        tags = x.split("*")
        tags[1] = tags[1].replace('\"', '')
        tags[1] = tags[1].replace(' ', '')
        rtn = rtn + "{tag:'"+tags[1]+"'},"
        # rtn = rtn + "{tag:'"+tags[1]+"',val:"+tags[0]+"},"
    rtn = rtn + "]"
    return json.dumps(rtn)

