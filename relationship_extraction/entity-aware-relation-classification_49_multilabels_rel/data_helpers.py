import numpy as np
import pandas as pd
import nltk
import re
import os
import json
from tqdm import tqdm
import subprocess
import utils
from configure import FLAGS
from bert_serving.client import BertClient
import multiprocessing


def clean_str(text):
    text = text.lower()
    # Clean the text
    # text = re.sub(r"_", " ", text)
    # text = re.sub(r"what's", "what is ", text)
    # text = re.sub(r"that's", "that is ", text)
    # text = re.sub(r"there's", "there is ", text)
    # text = re.sub(r"it's", "it is ", text)
    # text = re.sub(r"\'s", " ", text)
    # text = re.sub(r"\'ve", " have ", text)
    # text = re.sub(r"can't", "can not ", text)
    # text = re.sub(r"n't", " not ", text)
    # text = re.sub(r"i'm", "i am ", text)
    # text = re.sub(r"\'re", " are ", text)
    # text = re.sub(r"\'d", " would ", text)
    # text = re.sub(r"\'ll", " will ", text)
    # text = re.sub(r",", " ", text)
    # text = re.sub(r"\.", " ", text)
    # text = re.sub(r"!", " ! ", text)
    # text = re.sub(r"\/", " ", text)
    # text = re.sub(r"\^", " ^ ", text)
    # text = re.sub(r"\+", " + ", text)
    # text = re.sub(r"\-", " - ", text)
    # text = re.sub(r"\=", " = ", text)
    # text = re.sub(r"'", " ", text)
    # text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    # text = re.sub(r":", " : ", text)
    # text = re.sub(r" e g ", " eg ", text)
    # text = re.sub(r" b g ", " bg ", text)
    # text = re.sub(r" u s ", " american ", text)
    # text = re.sub(r"\0s", "0", text)
    # text = re.sub(r" 9 11 ", "911", text)
    # text = re.sub(r"e - mail", "email", text)
    # text = re.sub(r"j k", "jk", text)
    # text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def _is_valid_input_data(input_line):
    """is the input data valid"""
    try:
        dic = input_line.strip()
        dic = json.loads(dic)
    except:
        return False
    if "text" not in dic or "postag" not in dic or \
            type(dic["postag"]) is not list:
        return False
    for item in dic['postag']:
        if "word" not in item or "pos" not in item:
            return False
    return True


def load_data_and_labels(path, features_saved_path, type):
    if not os.path.exists(features_saved_path + type + "_features.csv"):
        data = []
        relation_all_true = []
        lines = [line.strip() for line in open(path)]
        max_sentence_length = 0
        for idx in tqdm(range(0, len(lines), 4)):
            id = lines[idx].split("\t")[0]
            relation = lines[idx + 2]
            relation_all_true.append(relation)
            sentence = lines[idx].split("\t")[1][1:-1]
            sentence = sentence.replace('<s>', ' s1 ')
            sentence = sentence.replace('</s>', ' s2 ')
            sentence = sentence.replace('<o>', ' o1 ')
            sentence = sentence.replace('</o>', ' o2 ')
            #sentence = clean_str(sentence)
            subject = 0
            object = 0
            try:
                tokens = nltk.word_tokenize(sentence)
                if max_sentence_length < len(tokens):
                    max_sentence_length = len(tokens)
                subject = tokens.index("s2") - 1
                object = tokens.index("o2") - 1
                sentence = " ".join(tokens)
            except:
                print(sentence)

            data.append([id, sentence, subject, object, relation])
        # if not os.path.exists("./resource/dev_target.txt") and type == "dev":
        #     with open("./resource/dev_target.txt", "w") as dev_relation_truely_output:
        #         for rel in relation_all_true:
        #             dev_relation_truely_output.write(rel + "\n")
        #         dev_relation_truely_output.close()

        print(path)
        print("max sentence length = {}\n".format(max_sentence_length))

        df = pd.DataFrame(data=data, columns=["id", "sentence", "subject", "object", "relation"])

        pos1, pos2 = get_relative_position(df, FLAGS.max_sentence_length)
        df["pos1"] = pos1
        df["pos2"] = pos2
        labels = []
        for rel in df['relation']:
            label_temp = ''
            rels_list = rel.split("\t")
            for r in rels_list:
                label_temp += str(utils.en_class2id()[r]) + ' '
            label_temp_ = label_temp.strip()
            labels.append(label_temp_)
        df['label'] = labels
        df.to_csv(features_saved_path + type + "_features.csv", sep="\t", index=False)
    else:
        df = pd.read_csv(features_saved_path + type + "_features.csv", sep="\t")
        print(df.columns)
    pos1 = df["pos1"].tolist()
    pos2 = df["pos2"].tolist()

    # Text Data
    x_text = df['sentence'].tolist()
    subject = df['subject'].tolist()
    object = df['object'].tolist()

    # Label Data
    y = df['label'].tolist()
    #labels_flat = y.values.ravel()
    # labels_count = np.unique(labels_flat).shape[0]
    labels_count = 49
    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # ...
    # 49 => [0 0 0 0 0 ... 0 0 0 0 1]

    def dense_to_one_hot(labels_y, num_classes):
        num_labels = len(labels_y)
        # index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        for i, la in enumerate(labels_y):
            one_hot_temp = [0] * num_classes
            y_list = str(la).split(" ")
            for rel in y_list:
                one_hot_temp[int(rel)] = 1
            labels_one_hot[i] = one_hot_temp
        return labels_one_hot

    labels = dense_to_one_hot(y, labels_count)
    labels = labels.astype(np.uint8)

    return x_text, labels, subject, object, pos1, pos2


def get_relative_position(df, max_sentence_length):
    # Position data
    pos1 = []
    pos2 = []
    for df_idx in tqdm(range(len(df))):
        sentence = df.iloc[df_idx]['sentence']
        tokens = nltk.word_tokenize(sentence)
        subject = df.iloc[df_idx]['subject']
        object = df.iloc[df_idx]['object']

        p1 = ""
        p2 = ""
        for word_idx in range(len(tokens)):
            p1 += str((max_sentence_length - 1) + word_idx - subject) + " "
            p2 += str((max_sentence_length - 1) + word_idx - object) + " "
        pos1.append(p1)
        pos2.append(p2)

    return pos1, pos2


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
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


class BuildingBertVector(object):
    # bert-serving-start -model_dir ./chinese_L-12_H-768_A-12 -num_worker=2

    def __init__(self, vocabulary_path="./vocab_word2id.dict",
                 single_file_lines_num=5000,
                 # start_threads_num=20,
                 bert_vec_output_path="./bert_vector/bert_vector.d768-"):
        self.vocabulary_path = vocabulary_path
        self.single_file_lines_num = single_file_lines_num  # Set an appropriate value based on the vocabulary size.
        # self.start_threads_num = start_threads_num   # Number of threads turned on, if this value is too high,
        # the bert service will die.
        self.bert_vec_output_path = bert_vec_output_path
        self.bc = BertClient()

    def generate_bert_vec(self, words_list_, num_):
        print("current job number: " + str(num_))
        with open(self.bert_vec_output_path + str(num_), "w") as output:
            vector_list = self.bc.encode(words_list_).tolist()
            for idx, v in enumerate(vector_list):
                str_ = ''
                for v_ in v:
                    str_ += str(v_) + ' '
                str_line = str_.strip(' ')
                output.write(words_list_[idx] + " " + str_line + '\n')
        output.close()

    def get_bert_embed(self,):

        line_num = self.single_file_lines_num
        words_list = []
        print("Building bert word vector...")
        with open(self.vocabulary_path, "r") as vocab:
            vocab_dict = json.load(vocab)
            for key in tqdm(vocab_dict.keys()):
                words_list.append(key)
            vocab.close()
            split_num = int(len(words_list)/line_num)
            print("start job...")
            for num in tqdm(range(0, split_num)):
                words_temp = words_list[num*line_num:(num+1)*line_num]
                self.generate_bert_vec(words_temp, num)
                # p = multiprocessing.Process(target=self.generate_bert_vec, args=(words_temp, num))
                # jobs_num.append(p)
                # p.start()
            # print("The number of working threads is: " + str(len(jobs_num)) + "\nPlease wait for a moment.")
            if len(words_list) > split_num*line_num:
                words_leftover = words_list[split_num*line_num:]
                self.generate_bert_vec(words_leftover, split_num)
                split_num = split_num + 1

        self.cat_bert_vec(self.bert_vec_output_path, split_num)

    def cat_bert_vec(self, path, file_num):
        home_name = path
        file_name = ""
        for i in range(file_num):
            file_name += home_name + str(i) + " "
        file_name = file_name.strip()
        subprocess.call("cat " + file_name +" > lic2019_bert_vector.768d.txt", shell=True)
        subprocess.call("rm -rf ./bert_vector/bert_vector.d768-*", shell=True)
        print("finished! and are be saved at : " + "\"" + "./lic2019_bert_vector.768d.txt" + "\"")

    def build_word_embedding_dict(self, path="./lic2019_bert_vector.768d.txt"):

        f = open(path, "r", encoding="utf-8")
        word_embedding_dict = {}
        for line in f.readlines():
            values = line.split()
            wid_key = values[0]
            wid_values = np.asarray(values[1:], dtype='float32')
            word_embedding_dict[wid_key] = wid_values

        return word_embedding_dict


if __name__ == "__main__":
    # trainFile = './entity_aware_dataset_lic2019_information_extraction/entity_aware_train'
    # devFile = './data/dev_mini'
    #
    # load_data_and_labels(trainFile)
    # load_data_and_labels(devFile, "./", type="dev")
    # with open("./vocab_word2id.dict", "r", encoding="utf-8") as f:
    #     dict = json.load(f)
    #     for k, v in dict.items():
    #         print(str(k) + " " + str(v)) ##何庆成 89946
    build_bert_vec = BuildingBertVector(vocabulary_path="./vocab_word2id.dict",
                                        single_file_lines_num=3000,
                                        bert_vec_output_path="./bert_vector/bert_vector.d768-")
    build_bert_vec.get_bert_embed()
    # df = pd.read_csv("./dev_features.csv", sep="\t")
    # print(df["pos1"])
