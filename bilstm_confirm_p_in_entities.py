from tqdm import tqdm
import json
import re
from data_helper import DataReader
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Embedding
from keras.layers import Bidirectional, GlobalMaxPool1D, Conv1D, Convolution1D
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import multi_gpu_model, plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import subprocess
from bert_serving.client import BertClient
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class BilstmConfirmP(object):
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
                                             spo[0],
                                             spo[1],
                                             features_cat])
                    data_out = pd.DataFrame(data=data,
                                            columns=["label", "row_id_in_raw_data", "sentence_seg",
                                                     "subject", "object", "features_cat"])
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


if __name__ == '__main__':
    use_bert_words_vector = False
    train = False
    evalution = False
    prediction = True
    # EMBEDDING_DIM = 768
    # EMBEDDING_FILE = ''
    MAX_NB_WORDS = 700000
    MAX_LENGTH = 100
    train_batch_size = 1000
    evalution_batch_size = 300
    prediction_batch_size = 300
    data_reader = BilstmConfirmP(train_path="./data/train_data.json", dev_path="./data/dev_data.json",
                                 test_path="./entities_extraction/bert-chinese-ner/output/test1_data_postag_sub_obj.json")
    train_df, dev_df, test_df = data_reader.data_load(train_output_path="./bilstm_dataset/train.csv",
                                                      dev_output_path="./bilstm_dataset/dev.csv",
                                                      test_output_path="./bilstm_dataset/test.csv")
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
    # test_Y = np.array(test_df["label"].tolist())

    # token
    # filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。'
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False, split=" ", oov_token="UNK")
    if not os.path.exists("./bilstm_dataset/vocab.json"):
        logger.info('Building vocabulary...')
        tokenizer.fit_on_texts(train_X + dev_X + test_X)
        word_index = tokenizer.word_index
        # save word_vocabulary_dict
        with open("./bilstm_dataset/vocab.json", mode="w", encoding="utf-8") as f:
            json.dump(word_index, f)
        f.close()
    else:
        logger.info("Loading vocabulary")
        with open("./bilstm_dataset/vocab.json", mode="r", encoding="utf-8") as vocab:
            word_index = json.load(vocab)
            tokenizer.word_index = word_index
    print('Found %s unique tokens' % len(word_index))
    logger.info("Convert text into sequences...")
    sequences_train = tokenizer.texts_to_sequences(train_X)
    sequences_dev = tokenizer.texts_to_sequences(dev_X)
    sequences_test = tokenizer.texts_to_sequences(test_X)
    logger.info("Padding sequences...")
    train_X = pad_sequences(sequences_train, maxlen=MAX_LENGTH)
    dev_X = pad_sequences(sequences_dev, maxlen=MAX_LENGTH)
    test_X = pad_sequences(sequences_test, maxlen=MAX_LENGTH)
    logger.info("train_X shape: " + str(train_X.shape))
    logger.info("dev_X shape: " + str(dev_X.shape))
    logger.info("train_Y shape: " + str(train_Y.shape))
    logger.info("dev_Y shape: " + str(dev_Y.shape))

    # print('Preparing word embedding matrix')
    if use_bert_words_vector:
        if not os.path.exists("./bilstm_dataset/bert_vector.768d.txt"):
            print("Please make sure the Bert service is turned on.")
            build_vector = BuildingBertVector(vocabulary_path="./bilstm_dataset/vocab.json",
                                              single_file_lines_num=5000,
                                              bert_vec_output_path="./bilstm_dataset/bert_vector.768d.txt")
            build_vector.get_bert_embed()

    # embeddings_index = data_reader.build_word_embedding_dict(path=EMBEDDING_FILE)
    # nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
    # print('nbwords:', nb_words)
    # if not os.path.exists("embedding_matrix.npy"):
    #     embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    #     for word, i in word_index.items():
    #         embedding_vector = embeddings_index.get(word)
    #         if embedding_vector is not None:
    #             embedding_matrix[i] = embedding_vector
    #     np.save("embedding_matrix.npy", embedding_matrix)
    #     print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    # else:
    #     embedding_matrix = np.load("embedding_matrix.npy")
    #     print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    # define model networks
    main_input = Input(shape=(MAX_LENGTH,), dtype='float64')
    # embedding_layer = Embedding(nb_words,
    #                             embedding_matrix.shape[1],
    #                             weights=[embedding_matrix],
    #                             input_length=MAX_LENGTH,
    #                             trainable=False)
    embedding_layer = Embedding(len(word_index) + 1, 256, input_length=MAX_LENGTH)
    embedded_squences = embedding_layer(main_input)
    rnn = Bidirectional(LSTM(128, return_sequences=True))(embedded_squences)
    cnn = Convolution1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(rnn)
    cnn = GlobalMaxPool1D()(cnn)
    cnn = Dropout(0.1)(cnn)
    cnn = Dense(50, activation='relu')(cnn)
    cnn = Dropout(0.1)(cnn)
    main_output = Dense(1, activation='sigmoid')(cnn)
    model = Model(inputs=main_input, outputs=main_output)
    plot_model(model, to_file='models_save/bilstm_model.png',
               show_shapes=True, show_layer_names=True)  # Save a graphical representation of the model
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    parallel_model.summary()

    best_model_path = 'models_save/bilstm_model.h5'
    # Set up callbacks
    tensorboard = TensorBoard(log_dir='./bilstm_logs')
    early_stopping = EarlyStopping(monitor='val_acc', patience=3)
    checkpoint = ModelCheckpoint(best_model_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    logger.info("train starting")
    if train:
        history = parallel_model.fit(train_X, train_Y, batch_size=train_batch_size, epochs=100,
                                     validation_data=(dev_X, dev_Y),
                                     callbacks=[checkpoint, early_stopping, tensorboard])
        bst_val_score = min(history.history['val_loss'])

    if evalution:
        model = load_model(best_model_path)
        preds_dev = model.predict(dev_X, batch_size=evalution_batch_size, verbose=1)
        preds_dev_labels = []
        for i in preds_dev:
            if i >= 0.5:
                preds_dev_labels.append(1)
            else:
                preds_dev_labels.append(0)
        preds_dev_labels = np.array(preds_dev_labels)
        # evalution
        logger.info("evalution starting")
        # pre_results = data_reader.predict(dev_X, "./bilstm_dataset/model").astype(dtype='int64')
        #
        precision = precision_score(dev_Y, preds_dev_labels, average='binary')
        recall = recall_score(dev_Y, preds_dev_labels, average='binary')
        f1 = f1_score(dev_Y, preds_dev_labels, average='binary')
        report = classification_report(dev_Y, preds_dev_labels)
        result_logs = "macro-averaged val_Precision = {:g}%\n".format(precision) + \
                      "macro-averaged val_Recall = {:g}%\n".format(recall) + \
                      "macro-averaged val_F1-score = {:g}%\n\n".format(f1) + report
        report_f = open("./results_output/bilstm_output/val_binary_classification_report.txt", 'w')
        report_f.write(result_logs)
        print(report)
    if prediction:
    # prediction
        model = load_model(best_model_path)
        logger.info("predication starting")
        preds_test = model.predict(test_X, batch_size=prediction_batch_size, verbose=1)
        preds_test_out = open("./results_output/bilstm_output/test_predication.txt", 'w')
        preds_test_labels = []
        for i in preds_test.ravel():
            preds_test_out.write(str(i) + '\n')
            if i >= 0.5:
                preds_test_labels.append(1)
            else:
                preds_test_labels.append(0)
        test_df.drop("features_cat", axis=1)
        test_df["label_prediction"] = preds_test_labels
        test_df.to_csv("./results_output/bilstm_output/test_predication.csv",
                       columns=["row_id_in_raw_data", "subject", "object", "label_prediction", "sentence_seg"],
                       sep='\t', index=False)

