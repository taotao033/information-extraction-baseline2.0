import tensorflow as tf
import numpy as np
from tqdm import tqdm
##The relationship between subject and object is irreversible.


def en_class2id():
    class2id = {}
    with open("../../data/target_labels/relationship_zh2en", 'r') as f:
        for i, line in enumerate(f.readlines()):
            p_zh, p_en = line.strip().split('\t')
            p_en_ = p_en.replace("\n", '')
            class2id[p_en_] = i
    f.close()
    return class2id


def en_id2class():
    class2id = en_class2id()
    id2class_en = {}
    for k, v in class2id.items():
        id2class_en[v] = k
    return id2class_en


def initializer():
    return tf.keras.initializers.glorot_normal()


def load_word2vec(word2vec_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) * np.sqrt(2.0 / len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load word2vec file {0}".format(word2vec_path))
    with open(word2vec_path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            idx = vocab.vocabulary_.get(word)
            if idx != 0:
                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return initW


def load_baidubaike(word2vec_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) * np.sqrt(2.0 / len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load baidubaike file {0}".format(word2vec_path))
    f = open(word2vec_path, 'r', encoding='utf8')
    for line in tqdm(f.readlines()):
        line = line.strip()
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        idx = vocab.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = embedding
    return initW


def load_glove(word2vec_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) * np.sqrt(2.0 / len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load glove file {0}".format(word2vec_path))
    f = open(word2vec_path, 'r', encoding='utf8')
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        idx = vocab.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = embedding
    return initW


def load_bert(bert_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) * np.sqrt(2.0 / len(vocab.vocabulary_))
    # load any vectors from the bert word vector
    print("Load bert vector file {0}".format(bert_path))
    f = open(bert_path, 'r', encoding='utf8')
    for line in tqdm(f.readlines()):
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        idx = vocab.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = embedding
    return initW

# en_class2id()
# en_id2class()
