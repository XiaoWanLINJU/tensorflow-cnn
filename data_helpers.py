#-*-encoding:utf-8-*-
import numpy as np
import re
import itertools
from collections import Counter
import parameter
import codecs


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def getEmbeddings(wordlist, index):
    global embeddingDic
    embeddings = []
    # set the max length of a sentence as 100, add zero padding
    for i in range(parameter.max_sen_len):
        if i < len(wordlist[index]) and wordlist[index][i] in embeddingDic.keys():
            embeddings += embeddingDic[wordlist[index][i]]
        else:
            embeddings += [0 for x in range(len(wordlist[index][0]))]# zeropadding and unk
    return [str(x) for x in embeddings]
def load_data2(file_name):
    instances = list(open(file_name).readlines())
    inputx = []
    inputy = []
    for instance in instances:
        inputy.append([float(x) for x in str(instance.strip().split(':')[0]).split()])
        inputx.append([float(x) for x in str(instance.strip().split(':')[1]).split()])
    return inputx, inputy        

def ori2svm(file_name):
    global embeddingDic
    embeddingDic = load_embedding() 
    print "len(embeddingDic)", len(embeddingDic)
    fr = codecs.open(file_name, 'r')
    fw = codecs.open(file_name + '.best_worst_embedding_svm', 'w')
    line = fr.readline()
    currentId = ''
    x = []
    y = []
    while line:
        if line.__contains__('Sent Id'):
            currentId = str(line.split(':')[0]).split()[2]
            line = fr.readline()
            scorelist = []
            featureslist = []
            wordlist = []
            while not line.__contains__('==='):
                tokens = line.split()
                scorelist.append(tokens[-1])
                featureslist.append(tokens[-10:-1])
                wordlist.append(tokens[1:-10])
                # fw.write(score + ' ' + 'sid:' + currentId + ' ' + ' '.join(features) + '\n')
                line = fr.readline()
            maxindex = scorelist.index(max(scorelist))
            minindex = scorelist.index(min(scorelist))
            x.append(getEmbeddings(wordlist, maxindex) + getEmbeddings(wordlist,  minindex))
            y.append([0, 1])
            x.append(getEmbeddings(wordlist, minindex) + getEmbeddings(wordlist,  maxindex))
            y.append([1, 0])
            fw.write('0 1: ' + ' '.join(featureslist[maxindex]) + ' ' + ' '.join(getEmbeddings(wordlist, maxindex)) + '\n')
            fw.write('1 0: ' + ' '.join(featureslist[minindex]) + ' ' + ' '.join(getEmbeddings(wordlist, minindex)) + '\n')

        line = fr.readline()
    fr.close()
    fw.close()
    return x, y

def load_embedding():
    """
    Load pre_trained embedding
    """
    global enbeddingDic#存成matrix, index 对应的是词
    embeddingDic = {}
    fr = codecs.open(parameter.dada_path + '/giga-50.bin')
    line = fr.readline()
    while line:
        tokens = line.strip().split()
        embeddingDic[tokens[0]]  = tokens[1:]
        line = fr.readline()
    return embeddingDic

def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) - 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_cntkfile(file_name):
    examples = list(open(file_name, 'r').readlines())
    x = []
    y = []
    for ex in examples:
        tokens = ex.strip().split()
        x.append(np.array([float(t) for t in tokens[:-1]]))#make the imput height as 2
        if tokens[-1] == '1.0':
            y.append([0, 1])
        else:
            y.append([1, 0])
    return x, y
