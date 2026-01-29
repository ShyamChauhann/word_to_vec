import os, sys
print(os.getcwd())
print(sys.path)

from Word2Vec_1WordContext import Word2Vec_1WordContext

sentences = [["i", "love", "nlp"], ["word2vec", "works"]]
model1 = Word2Vec_1WordContext(sentences, learning_rate=1.2, nodes_HL=3)