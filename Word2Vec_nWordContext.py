import numpy as np
from collections import OrderedDict, defaultdict
# import plotly.plotly as py
from plotly import graph_objs as py
from plotly.graph_objs import *
from functools import reduce


class Word2Vec_nWordContext(object):
    def __init__(self, sentences, learning_rate = 1.0, context_size = 3, nodes_HL = 3):
        self.sentences = sentences
        self.N = nodes_HL # number of nodes in Hidden Layer
        self.V = None # Vocabulary size
        self.WI = None # input weight matrix
        self.WO = None # output weight matrix
        self.vocabulary = None
        self.learning_rate = learning_rate
        self.context_size = context_size # number of words in context vector
    
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
        
    def average_context_vec(self, context):
        """ Takes the average of the word vectors and returns a new vector for the context"""
        c = len(context)
        context_weights = map(lambda word: self.WI[self.vocabulary[word]], context)
        return reduce(lambda a, b: a + b, context_weights ) / float(c)
    
    def softmax_regression(self, word, h):
        """ returns posterior probability P(word | context) """
        return (np.exp(h.dot(self.WO.T[self.vocabulary[word]])) / 
                sum(np.exp(h.dot(self.WO.T[self.vocabulary[w]])) for w in self.vocabulary))
    
    def backprop(self, context, target):
        """ Computes backpropagation of errors to weight matrices,
        using stochastic gradient descent """

        for word in self.vocabulary:
            
            h = self.average_context_vec(context) # context word weight vector
            P_word_context = self.softmax_regression(word, h)  # posterior probability P(word | context)

            if word == target:
                t = 1
                #print "P(target|context)", P_word_context
            else:
                t = 0

            err = t - P_word_context # error

            # weight update using stochastic gradient descent
            self.WO.T[self.vocabulary[word]] -= self.learning_rate * err * h
            # update brings word vector closer in the feature space if word = target, and push them apart otherwise.

        EH = self.WO.sum(axis=1) 
        for input_word in context:
            # update only weights for context words
            self.WI[self.vocabulary[input_word]] -= (1. / len(context)) * self.learning_rate * EH
    
    def train(self):
        """ trains text and returns trained word matrix
        and ordered dictionary of vocabulary"""

        bow = self.sentences2bow()
        self.random_init()
        
        # runs context window across sentence
        # applies window expansion and reduction 
        # at the begin and end of sentence respectively
        for sentence in self.sentences:
            word_tuple =  tuple(sentence.split())
            count = 1
            context = []
            for i, word in enumerate(word_tuple):
                if word != '<s>':
                    target = word
                    if count > self.context_size:
                        context = context[1:]
                    context.append(word_tuple[i-1])
                    self.backprop(context, target)
                    if word == '</s>':
                        for n in range(len(context) - 1, 0, -1):
                            context = context[-n:]
                            self.backprop(context, target)
                    count += 1
                    
        return self.WI, OrderedDict(sorted(self.vocabulary.items(), key=lambda t: t[1]))

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

    model = Word2Vec_nWordContext(sentences, learning_rate = 1.0, context_size = 3)
    WI, vocab = model.train()
    print (WI, vocab)