from tqdm import tqdm
import json
import re
from data_helper import DataReader
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from thundersvm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

clf = SVC(gamma=0.5, C=100, verbose=True)


class SvmConfirmP(object):
    def __init__(self, train_path, dev_path, test_path):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

    def _is_valid_input_data(self, input_line):
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

    def list2str(self, list_):
        str_ = ''
        for li in list_:
            str_ += li + ' '
        return str_.strip()

    def _add_item_offset(self, token, sentence): #'喜剧之王'
        """Get the start and end offset of a token in a sentence"""
        s_pattern = re.compile(re.escape(token), re.I) #re.compile('\\喜\\剧\\之\\王', re.IGNORECASE)
        token_offset_list = []
        for m in s_pattern.finditer(sentence): #m={SRE_MATCH} <_sre.SRE_Match object; span=(21, 25), match='喜剧之王'>
            token_offset_list.append((m.group(), m.start(), m.end()))
        return token_offset_list # <class 'list'>: [('喜剧之王', 21, 25)]

    def _cal_item_pos(self, target_offset, idx_list):
        """Get the index list where the token is located"""
        target_idx = []
        for target in target_offset:
            start, end = target[1], target[2]
            cur_idx = []
            for i, idx in enumerate(idx_list):
                if idx >= start and idx < end:
                    cur_idx.append(i)
            if len(cur_idx) > 0:
                target_idx.append(cur_idx)
        return target_idx

    def _get_token_idx(self, sentence_term_list, sentence):
        """Get the start offset of every token"""
        token_idx_list = []
        start_idx = 0
        for sent_term in sentence_term_list:
            if start_idx >= len(sentence):
                break
            token_idx_list.append(start_idx)
            start_idx += len(sent_term)
        return token_idx_list

    def data_load(self, train_output_path, dev_output_path, test_output_path):
        if not os.path.exists(train_output_path) or not \
         os.path.exists(dev_output_path) or not os.path.exists(test_output_path):
            nullkey = 'N'
            data_out_path = [train_output_path, dev_output_path, test_output_path]
            data_type = ['training', 'dev', 'test']
            for path_i, file in enumerate([self.train_path, self.dev_path, self.test_path]):
                logger.info("Processing " + data_type[path_i] + " set, please wait for a moment...")
                with open(file, 'r', encoding='utf-8') as f:
                    data = []
                    for row_id, line in enumerate(tqdm(f.readlines()), 1):
                        # verify that the input format of each line meets the format
                        if not self._is_valid_input_data(line):
                            print('Format is error')
                            return None
                        dic = line.strip()
                        dic = json.loads(dic)
                        if "spo_list" not in dic:
                            continue
                        sentence = dic['text']
                        sentence = str(sentence).replace(" ", '').lower()
                        sentence_term_list = [str(item['word']).replace(" ", '').lower() for item in dic['postag']
                                              if item["word"] != " "]
                        token_idx_list = self._get_token_idx(sentence_term_list, sentence)

                        sub_temp = []
                        obj_temp = []
                        entities = set()
                        for spo in dic["spo_list"]:
                            subject = str(spo['subject']).replace(" ", '').replace("《", '').replace("》", '').lower()
                            sub_temp.append(subject)
                            object = str(spo['object']).replace(" ", '').replace("《", '').replace("》", '').lower()
                            obj_temp.append(object)
                            entities.add(subject)
                            entities.add(object)
                        if len(sub_temp) == 0 or len(obj_temp) == 0:
                            continue
                        assert len(sub_temp) == len(obj_temp), "Error, because len(sub_temp) != len(obj_temp)"
                        sub_obj_zip = list(zip(sub_temp, obj_temp))
                        for e1 in entities:
                            for e2 in entities:
                                tuple_temp = (e1, e2)
                                if tuple_temp not in sub_obj_zip:
                                    sub_obj_zip.append(tuple_temp)
                        add_len = len(sub_obj_zip) - len(sub_temp)
                        sub_obj_p_zip_1 = []
                        sub_obj_p_zip_0 = []
                        try:
                            if add_len > 0:
                                sub_obj_zip_1 = sub_obj_zip[0:len(sub_temp)]
                                sub_obj_zip_0 = sub_obj_zip[len(sub_temp):]
                                sub_obj_p_zip_1 = zip(list(zip(*sub_obj_zip_1))[0], list(zip(*sub_obj_zip_1))[1],
                                                      [1] * len(sub_temp))
                                sub_obj_p_zip_0 = zip(list(zip(*sub_obj_zip_0))[0], list(zip(*sub_obj_zip_0))[1],
                                                      [0] * add_len)
                            else:
                                sub_obj_p_zip_1 = zip(list(zip(*sub_obj_zip))[0], list(zip(*sub_obj_zip))[1],
                                                      [1] * len(sub_temp))
                                sub_obj_p_zip_0 = []
                        except IndexError:
                            print(sub_obj_zip)

                        for sub_obj_p in [sub_obj_p_zip_1, sub_obj_p_zip_0]:
                            for spo in sub_obj_p:
                                m1_char = ''
                                m2_char = ''
                                m1_bichar = ''
                                m2_bichar = ''
                                mention2_left_temp = []  # Words or bigrams in particular positions left and right of M1/M2.
                                mention2_right_temp = []
                                words_between_s_o = ''  # Bag of words or bigrams between the two entities.
                                s_idx_list = self._cal_item_pos(self._add_item_offset(spo[0], sentence),
                                                                token_idx_list)  ##[[0,1,2]]
                                o_idx_list = self._cal_item_pos(self._add_item_offset(spo[1], sentence),
                                                                token_idx_list)
                                if len(s_idx_list) == 0 or len(o_idx_list) == 0:
                                    continue
                                try:
                                    m1_char = self.list2str([c1 for c1 in spo[0]])
                                    m1_char_zip = zip(m1_char.replace(' ', ''), m1_char.replace(' ', '')[1:] + nullkey)
                                    m1_bichar = self.list2str([c[0] + c[1] for c in m1_char_zip])

                                    m2_char = self.list2str([c2 for c2 in spo[1]])
                                    m2_char_zip = zip(m2_char.replace(" ", ''), m2_char.replace(" ", '')[1:] + nullkey)
                                    m2_bichar = self.list2str([c[0] + c[1] for c in m2_char_zip])

                                    for o_idx in o_idx_list:
                                        pos_left_index = o_idx[0] - 1
                                        pos_right_index = o_idx[-1] + 1
                                        if pos_left_index < 0 or pos_right_index > len(sentence_term_list) - 1:
                                            continue
                                        mention2_left_temp.append(sentence_term_list[pos_left_index])
                                        mention2_right_temp.append(sentence_term_list[pos_right_index])

                                    if s_idx_list[0][0] > o_idx_list[0][0]:
                                        words_between_s_o = self.list2str(sentence_term_list[o_idx_list[0][0] + 1:
                                                                                             s_idx_list[0][0]])
                                    elif s_idx_list[0][0] < o_idx_list[0][0]:
                                        words_between_s_o = self.list2str(sentence_term_list[s_idx_list[0][0] + 1:
                                                                                             o_idx_list[0][0]])
                                    elif s_idx_list[0][0] == o_idx_list[0][0]:
                                        words_between_s_o = ''

                                except IndexError:
                                    data_reader_ = DataReader()
                                    sentence, sentence_term_list = \
                                        data_reader_.deal_with_participle_less_cutting_problem(sentence, spo[0], spo[1])
                                    token_idx_list_update = self._get_token_idx(sentence_term_list, sentence)

                                    s_idx_list = self._cal_item_pos(self._add_item_offset(spo[0], sentence),
                                                                    token_idx_list_update)  ##[[0,1,2]]
                                    o_idx_list = self._cal_item_pos(self._add_item_offset(spo[1], sentence),
                                                                    token_idx_list_update)
                                    if len(s_idx_list) == 0 or len(o_idx_list) == 0:
                                        continue
                                    try:
                                        m1_char = self.list2str([c1 for c1 in spo[0]])
                                        m1_char_zip = zip(m1_char, m1_char[1:] + nullkey)
                                        m1_bichar = self.list2str([c[0] + c[1] for c in m1_char_zip])

                                        m2_char = self.list2str([c2 for c2 in spo[1]])
                                        m2_char_zip = zip(m2_char, m2_char[1:] + nullkey)
                                        m2_bichar = self.list2str([c[0] + c[1] for c in m2_char_zip])

                                        mention2_left_temp = []
                                        mention2_right_temp = []
                                        for o_idx in o_idx_list:
                                            pos_left_index = o_idx[0] - 1
                                            pos_right_index = o_idx[-1] + 1
                                            if pos_left_index < 0 or pos_right_index > len(sentence_term_list) - 1:
                                                continue
                                            mention2_left_temp.append(sentence_term_list[pos_left_index])
                                            mention2_right_temp.append(sentence_term_list[pos_right_index])

                                        if s_idx_list[0][0] > o_idx_list[0][0]:
                                            words_between_s_o = self.list2str(sentence_term_list[o_idx_list[0][0] + 1:
                                                                                                 s_idx_list[0][0]])
                                        elif s_idx_list[0][0] < o_idx_list[0][0]:
                                            words_between_s_o = self.list2str(sentence_term_list[s_idx_list[0][0] + 1:
                                                                                                 o_idx_list[0][0]])
                                        elif s_idx_list[0][0] == o_idx_list[0][0]:
                                            words_between_s_o = ''

                                    except IndexError:
                                        print("Error, unsolved")

                                sent_seg = self.list2str(sentence_term_list)
                                m1_m2_unigram_bigram = m1_char + " " + m2_char + " " + m1_bichar + " " + m2_bichar
                                label = spo[2]
                                features_cat = str(spo[0]) + " " + str(spo[1]) + " " + \
                                               self.list2str(mention2_left_temp) + " " + \
                                               self.list2str(mention2_right_temp) + " " + \
                                               words_between_s_o
                                # data.append([label,
                                #              row_id,
                                #              sent_seg,
                                #              spo[0],
                                #              spo[1],
                                #              m1_m2_unigram_bigram,
                                #              self.list2str(mention2_left_temp),
                                #              self.list2str(mention2_right_temp),
                                #              words_between_s_o])

                    # data_out = pd.DataFrame(data=data,
                    #                         columns=["label", "row_id_in_raw_data", "sentence_seg",
                    #                                  "subject", "object", "m1_m2_unigram_bigram",
                    #                                  "mention2_left_temp", "mention2_right_temp",
                    #                                  "words_between_s_o"])
                                data.append([label,
                                             row_id,
                                             sent_seg,
                                             features_cat])
                    data_out = pd.DataFrame(data=data,
                                            columns=["label", "row_id_in_raw_data", "sentence_seg",
                                                     "features_cat"])
                    data_out.drop_duplicates(inplace=True)
                    if data_type[path_i] == 'training':
                        train_df = data_out
                    elif data_type[path_i] == 'dev':
                        dev_df = data_out
                    elif data_type[path_i] == 'test':
                        test_df = data_out
                    logger.info("Saving file " + data_type[path_i] + " , please wait...")
                    data_out.to_csv(data_out_path[path_i], sep='\t', index=False)
        else:
            logger.info("Loading files...")
            train_df = pd.read_csv(train_output_path, sep='\t')
            dev_df = pd.read_csv(dev_output_path, sep='\t')
            test_df = pd.read_csv(test_output_path, sep='\t')

        return train_df, dev_df, test_df

    def train(self, x, y):

        clf.fit(x, y)
        print("saving model")
        clf.save_to_file("./svm_dataset/model")

    def predict(self, x, model_path):
        clf.load_from_file(model_path)
        pre = clf.predict(x)
        return pre


if __name__ == '__main__':
    stop_words = []
    MAX_NB_WORDS = 700000
    MAX_LENGTH = 50
    data_reader = SvmConfirmP(train_path="./data/train_data.json", dev_path="./data/dev_data.json",
                              test_path="./entities_extraction/bert-chinese-ner/output/test1_data_postag_sub_obj.json")
    train_df, dev_df, test_df = data_reader.data_load(train_output_path="./svm_dataset/train.csv",
                                                      dev_output_path="./svm_dataset/dev.csv",
                                                      test_output_path="./svm_dataset/test.csv")
    train_X = train_df["features_cat"].values.tolist()
    train_len = [len(sent) for sent in train_X]
    print("In training set, max length: " + str(max(train_len)))
    train_Y = np.array(train_df["label"].tolist())

    dev_X = dev_df["features_cat"].values.tolist()
    dev_len = [len(sent) for sent in dev_X]
    print("In dev set, max length: " + str(max(dev_len)))
    dev_Y = np.array(dev_df["label"].tolist())

    test_X = test_df["features_cat"].values.tolist()
    test_len = [len(sent) for sent in test_X]
    print("In test set, max length: " + str(max(test_len)))
    test_Y = np.array(test_df["label"].tolist())

    vectorizer_tfidf = TfidfVectorizer(analyzer='word', stop_words=stop_words, lowercase=True,
                                       ngram_range=(1, 5), min_df=5, norm='l2')
    vectorizer_tfidf.fit_transform(train_X + dev_X + test_X)
    train_X = vectorizer_tfidf.transform(train_X)
    dev_X = vectorizer_tfidf.transform(dev_X)
    test_X = vectorizer_tfidf.transform(test_X)
    # # token
    # # filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。'
    # tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False, split=" ", oov_token="UNK")
    # if not os.path.exists("./svm_dataset/vocab.json"):
    #     logger.info('Building vocabulary...')
    #     tokenizer.fit_on_texts(train_X + dev_X + test_X)
    #     word_index = tokenizer.word_index
    #     # save word_vocabulary_dict
    #     with open("./svm_dataset/vocab.json", mode="w", encoding="utf-8") as f:
    #         json.dump(word_index, f)
    #     f.close()
    # else:
    #     logger.info("Loading vocabulary")
    #     with open("./svm_dataset/vocab.json", mode="r", encoding="utf-8") as vocab:
    #         word_index = json.load(vocab)
    #         tokenizer.word_index = word_index
    # print('Found %s unique tokens' % len(word_index))
    # logger.info("Convert text into sequences...")
    # sequences_train = tokenizer.texts_to_sequences(train_X)
    # sequences_dev = tokenizer.texts_to_sequences(dev_X)
    # sequences_test = tokenizer.texts_to_sequences(test_X)
    # logger.info("Padding sequences...")
    # train_X = pad_sequences(sequences_train, maxlen=MAX_LENGTH)
    # dev_X = pad_sequences(sequences_dev, maxlen=MAX_LENGTH)
    # test_X = pad_sequences(sequences_test, maxlen=MAX_LENGTH)
    # ss = StandardScaler()
    # ss.fit(np.vstack((train_X, dev_X, test_X)))
    # train_X = ss.transform(train_X)
    # dev_X = ss.transform(dev_X)
    # test_X = ss.transform(test_X)
    # # train
    # logger.info("train_X shape: " + str(train_X.shape))
    # logger.info("train_Y shape: " + str(train_Y.shape))
    # logger.info("train starting")

    data_reader.train(train_X, train_Y)

    # evalution
    logger.info("evalution starting")
    pre_results = data_reader.predict(dev_X, "./svm_dataset/model").astype(dtype='int64')

    precision = precision_score(dev_Y, pre_results, average='binary')
    recall = recall_score(dev_Y, pre_results, average='binary')
    f1 = f1_score(dev_Y, pre_results, average='binary')
    print("macro-averaged Precision = {:g}%\n".format(precision) + \
          "macro-averaged Recall = {:g}%\n".format(recall) + \
          "macro-averaged F1-score = {:g}%\n\n".format(f1))

    results_out = open('./results_output/svm_output/dev_prediction.txt', 'w')

    for pre in pre_results.tolist():
        results_out.write(str(pre) + '\n')
    results_out.close()
