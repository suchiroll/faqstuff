import gensim
import nltk
import glob
import os
nltk.download('punkt')

class Node:
    def _init_(self,simval, docnum1, docnum2):
        self.left = None
        self.right = None
        self.simval = simval
        self.docnum1 = docnum1
        self.docnum2 = docnum2
    def PrintTree(self):
        print(self.simval)

class Stack:
    def _init_(self):
        self.items = []
    def isEmpty(self):
        return self.items == []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def peek(self):
        return self.items[len(self.items)-1]
    def size(self):
        return len(self.items)

print(dir(gensim))

"""raw_documents = ["I'm taking the show on the road.",
                 "My socks are a force multiplier.",
             "I am the barber who cuts everyone's hair who doesn't cut their own.",
             "Legend has it that the mind is a mad monkey.",
            "I make my own fun."]
print("Number of documents:",len(raw_documents))"""

raw_documents = ["" for x in range(213)]
i = 1
f = open("revised/revised0.txt", "w")
while i < 214:
    #print("i is first ", i)
    try:
        doc_title = "revisedfaq/revised" + str(i) + ".txt"
        if os.path.exists(doc_title):
            f = open(doc_title,"r")
            print("i is and f is ", i, doc_title)
            message = f.read()
    #    print("message:")
     #   print(message)
            raw_documents[i-1] = message
      #  print("raw docs: " + raw_documents[i-1])
        i+=1
    finally:
            f.close()

from nltk.tokenize import word_tokenize
gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents]
print(gen_docs)

dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary[5])

print("Number of words in dictionary:",len(dictionary))
for i in range(len(dictionary)):
    print(i, dictionary[i])

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
print(corpus)

tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)
s = 0
for i in corpus:
    s += len(i)
print(s)

sims = gensim.similarities.Similarity('/work',tf_idf[corpus],
                                      num_features=len(dictionary))
print(sims)
print("whjswh IKLJFDKLSJAFKLDS ")
print(type(sims))

query_doc = [w.lower() for w in word_tokenize(raw_documents[0])]
#query_doc = [w.lower() for w in word_tokenize("Socks are a force for good.")]
print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]

print(query_doc_tf_idf)
print("array of similarities:")
print(sims[query_doc_tf_idf])

