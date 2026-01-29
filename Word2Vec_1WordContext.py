import numpy as np
from collections import OrderedDict, defaultdict
from plotly import graph_objs as py
from plotly.graph_objs import *

class Word2Vec_1WordContext(object):
    def __init__(self, sentences, learning_rate = 1.0, nodes_HL = 3):
        self.sentences = sentences
        self.N = nodes_HL # number of nodes in Hidden Layer
        self.V = None # Vocabulary size
        self.WI = None # input weight matrix
        self.WO = None # output weight matrix
        self.vocabulary = None
        self.learning_rate = learning_rate
    
    def Vocabulary(self):
        """ Instantiates a default dictionary with its 
        length as default factory """
        dictionary = defaultdict()
        # len of dictionary gives a unique integer to each new word
        dictionary.default_factory = lambda: len(dictionary) 
        return dictionary

    def docs2bow(self, docs, dictionary):
        """Transforms a list of strings into a list of lists where 
        each unique item is converted into a unique integer."""
        for doc in docs:
            yield [dictionary[word] for word in doc.split()] # returns a generator
    
    def sentences2bow(self):
        """ Creates the dictionary of the text's vocabulary 
        and returns the text with each words replaced by their unique integer"""
        self.vocabulary = self.Vocabulary()
        bow = list(self.docs2bow(self.sentences, self.vocabulary))
        return bow
    
    def random_init(self):
        """ initializes  weight matrices for neural network """
        self.V = len(self.vocabulary)
        
        # random initialization of weights between [-0.5 , 0.5] normalized by number of nodes mapping to.
        self.WI =(np.random.random((self.V, self.N)) - 0.5) / self.N # input weights
        self.WO =(np.random.random((self.N, self.V)) - 0.5) / self.V # output weights
    
    def softmax_regression(self, word, h):
        """ returns posterior probability P(word | context) """
        return (np.exp(h.dot(self.WO.T[self.vocabulary[word]])) / 
                sum(np.exp(h.dot(self.WO.T[self.vocabulary[w]])) for w in self.vocabulary))
    
    def backprop(self, context, target):
        """ Computes backpropagation of errors to weight matrices,
        using stochastic gradient descent """

        for word in self.vocabulary:

            h = self.WI[self.vocabulary[context]] # context word weight vector
            P_word_context = self.softmax_regression(word, h) # posterior probability P(word | context)
            
            if word == target:
                t = 1
                #print "P(target|context)", P_word_context
            else:
                t = 0

            err = t - P_word_context # error

            # weight update using stochastic gradient descent
            self.WO.T[self.vocabulary[word]] -= self.learning_rate * err * h
            # update brings word vector closer in the feature space if word == target, and push them apart otherwise.

        # update only weights for input word
        self.WI[self.vocabulary[context]] -= self.learning_rate * self.WO.sum(axis = 1) 
    
    def train(self):
        """ trains text and returns trained word matrix
        and ordered dictionary of vocabulary"""

        bow = self.sentences2bow()
        # visualize bag-of-word sentence conversion
        # print bow
        self.random_init()
        
        # train text by predicting next word in the sentence
        for sentence in self.sentences:
            prev_word = None
            for word in sentence.split():
                if prev_word != None:
                    target = word
                    context = prev_word
                    self.backprop(context, target)
                prev_word = word

        return self.WI, OrderedDict(sorted(self.vocabulary.items(), key = lambda t: t[1]))

    def graph_vector_space(self):
        """ 3D Scatter plot of first 3 word features with Plotly"""
        vocab = OrderedDict(sorted(self.vocabulary.items(), key = lambda t: t[1]))

        trace1 = Scatter3d(
            x = self.WI.T[0],
            y = self.WI.T[1],
            z = self.WI.T[2],
            mode ='markers+text',
            text = vocab.keys(),
            marker = Marker(
                size = 8,
                line = Line(
                    color = 'rgba(217, 217, 217, 0.14)',
                    width = 0.5
                ),
                opacity = 0.8
            )
        )
        data = Data([trace1])
        layout = Layout(
            margin = Margin(
                l = 0,
                r = 0,
                b = 0,
                t = 0
            )
        )
        fig = Figure(data = data, layout = layout)
        return py.iplot(fig)

if __name__ == '__main__':
    sentences = ['<s> the prince loves skateboarding in the park </s>', 
                 '<s> the princess loves the prince but the princess hates skateboarding </s>',
                 '<s> skateboarding in the park is popular </s>',
                 '<s> the prince is popular but the prince hates attention </s>',
                 '<s> the princess loves attention but the princess hates the park </s>']

    model = Word2Vec_1WordContext(sentences, learning_rate = 1.0)
    WI, vocab = model.train()
    print( WI, vocab)




