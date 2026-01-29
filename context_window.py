
def context_window_search(context_size = 3): 
    """ Illustrates how the context window changes across a sentence
    for a given target word """
    
    word_tuple = tuple(['<s>', 'a', 'b', 'c', 'd', 'e', '</s>'])
    print "sentence example: ", word_tuple
    context_size = context_size
    count = 1
    context = []

    for i, word in enumerate(word_tuple):
        if word != '<s>':
            target = word
            if count > context_size:
                context = context[1:]
            context.append(word_tuple[i-1])         
            print "context: {}, target: {}".format(context, target)
            if word == '</s>':
                for n in range(len(context)-1, 0, -1):
                    context = context[-n:]
                    print "context: {}, target: {}".format(context, target)
            count += 1