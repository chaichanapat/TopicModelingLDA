import nltk
import gensim.corpora as corpora
import gensim
import re
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def Predict(word,numword):

    word          = re.sub(r'[^\w\s]', '', word).lower()

    stop_words    = set(stopwords.words('english'))
    word_tokens   = word_tokenize(word)

    filtered_word = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_word.append(w)

    phrase   = ['made','make']

    for word in list(filtered_word):  # iterating on a copy since removing will mess things up
        if word in phrase:
            filtered_word.remove(word)

    lm = nltk.WordNetLemmatizer()

    def lemmatize(data):
       text = [[lm.lemmatize(word) for word in data]]
       return text

    word = lemmatize(filtered_word)

    id2word = corpora.Dictionary(word)

    corpus = [id2word.doc2bow(text) for text in word]

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

