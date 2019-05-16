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
from keras.layers.merge import concatenate
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

    def data_load(self,):

        train_df_ = pd.read_csv(self.train_path, sep='\t')
        rel_train = []
        for i in train_df_["relation"].tolist():
            if i == 'OTHER':
                rel_train.append(0)
            else:
                rel_train.append(1)
        train_df_["relation"] = rel_train

        dev_df_ = pd.read_csv(self.dev_path, sep='\t')
        rel_dev = []
        for i in dev_df_["relation"].tolist():
            if i == 'OTHER':
                rel_dev.append(0)
            else:
                rel_dev.append(1)
        dev_df_["relation"] = rel_dev

        test_df_ = pd.read_csv(self.test_path, sep='\t')
        rel_test = [0] * len(test_df_["relation"].tolist())
        test_df_["relation"] = rel_test

        return train_df_, dev_df_, test_df_


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
        subprocess.call("cat " + file_name +" > ./bilstm_dataset/bilstm_v2/bert_vector.768d.txt", shell=True)
        subprocess.call("rm -rf ./bert_vector_temp/bert_vector.d768_*", shell=True)
        print("finished! and are be saved at : " + "\"" + "./bilstm_dataset/bilstm_v2/bert_vector.768d.txt" + "\"")

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
    train = True
    evalution = False
    prediction = False
    EMBEDDING_DIM = 768
    EMBEDDING_FILE = "./bilstm_dataset/bilstm_v2/bert_vector.768d.txt"
    MAX_NB_WORDS = 700000
    MAX_LENGTH = 100
    train_batch_size = 64
    evalution_batch_size = 32
    prediction_batch_size = 32
    data_reader = BilstmConfirmP(train_path="./relationship_extraction/"
                                            "entity-aware-relation-classification_49_multilabels_rel/"
                                            "data/20190509_rel_50/train_features.csv",
                                 dev_path="./relationship_extraction/"
                                          "entity-aware-relation-classification_49_multilabels_rel/"
                                          "data/20190509_rel_50/dev_features.csv",
                                 test_path="./relationship_extraction/"
                                           "entity-aware-relation-classification_49_multilabels_rel/"
                                           "data/20190509_rel_50/test_features.csv")
    train_df, dev_df, test_df = data_reader.data_load()
    train_X = train_df["sentence"].values.tolist()
    train_len = [len(str(sent).split(" ")) for sent in train_X]
    print("In training set, max length: " + str(max(train_len)))
    train_Y = np.array(train_df["relation"].tolist())

    dev_X = dev_df["sentence"].values.tolist()
    dev_len = [len(str(sent).split(" ")) for sent in dev_X]
    print("In dev set, max length: " + str(max(dev_len)))
    dev_Y = np.array(dev_df["relation"].tolist())

    test_X = test_df["sentence"].values.tolist()
    test_len = [len(str(sent).split(" ")) for sent in test_X]
    print("In test set, max length: " + str(max(test_len)))
    # test_Y = np.array(test_df["label"].tolist())

    # features
    train_features1 = train_df[["subject", "object"]]
    train_features2 = train_df[["pos1", "pos2"]]
    dev_features = dev_df[["subject", "object", "pos1", "pos2"]]
    test_features = test_df[["subject", "object", "pos1", "pos2"]]

    # ss = StandardScaler()
    # ss.fit(np.vstack((train_features, dev_features, test_features)))
    # fea_train = ss.transform(train_features)
    # fea_dev = ss.transform(dev_features)
    # fea_test = ss.transform(test_features)

    # token
    # filters = '!?"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n\r！@#￥%…&*（）：“”’‘；《》？，。'
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False, split=" ", oov_token="UNK")
    if not os.path.exists("./bilstm_dataset/bilstm_v2/vocab.json"):
        logger.info('Building vocabulary...')
        tokenizer.fit_on_texts(train_X + dev_X + test_X)
        word_index = tokenizer.word_index
        # save word_vocabulary_dict
        with open("./bilstm_dataset/bilstm_v2/vocab.json", mode="w", encoding="utf-8") as f:
            json.dump(word_index, f)
        f.close()
    else:
        logger.info("Loading vocabulary")
        with open("./bilstm_dataset/bilstm_v2/vocab.json", mode="r", encoding="utf-8") as vocab:
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

    print('Preparing bert word embedding vector')
    build_vector = BuildingBertVector(vocabulary_path="./bilstm_dataset/bilstm_v2/vocab.json",
                                      single_file_lines_num=3000,
                                      bert_vec_output_path="./bert_vector_temp/bert_vector.d768_")
    if not os.path.exists(EMBEDDING_FILE):
        print("Please make sure the Bert service is turned on.")
        build_vector.get_bert_embed()

    # embeddings_index = build_vector.build_word_embedding_dict(path=EMBEDDING_FILE)
    # nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
    # print('nbwords:', nb_words)
    # if not os.path.exists("./bilstm_dataset/bilstm_v2/embedding_matrix.npy"):
    #     embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    #     for word, i in word_index.items():
    #         embedding_vector = embeddings_index.get(word)
    #         if embedding_vector is not None:
    #             embedding_matrix[i] = embedding_vector
    #     np.save("./bilstm_dataset/bilstm_v2/embedding_matrix.npy", embedding_matrix)
    #     print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    # else:
    #     embedding_matrix = np.load("./bilstm_dataset/bilstm_v2/embedding_matrix.npy")
    #     print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    #
    # # define model networks
    # def model_network(MAX_LENGTH, embedding_matrix, nb_words, features):
    #
    #     main_input = Input(shape=(MAX_LENGTH,), dtype='float64')
    #     features_input = Input(shape=(features.shape[1],))
    #
    #     embedding_layer = Embedding(nb_words,
    #                                 embedding_matrix.shape[1],
    #                                 weights=[embedding_matrix],
    #                                 input_length=MAX_LENGTH,
    #                                 trainable=False)
    #     # embedding_layer = Embedding(len(word_index) + 1, 256, input_length=MAX_LENGTH)
    #     embedded_squences = embedding_layer(main_input)
    #     rnn = Bidirectional(LSTM(128, return_sequences=True))(embedded_squences)
    #     cnn = Convolution1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(rnn)
    #     features_dense = Dense(50, activation='relu')(features_input)
    #     merged = concatenate([cnn, features_dense])
    #     cnn = GlobalMaxPool1D()(merged)
    #     cnn = Dropout(0.1)(cnn)
    #     cnn = Dense(50, activation='relu')(cnn)
    #     cnn = Dropout(0.1)(cnn)
    #     main_output = Dense(1, activation='sigmoid')(cnn)
    #     model = Model(inputs=[main_input, features_input], outputs=main_output)
    #
    #     return model
    #
    # model = model_network(MAX_LENGTH, embedding_matrix, nb_words)
    # plot_model(model, to_file='models_save/bilstm_model.png',
    #            show_shapes=True, show_layer_names=True)  # Save a graphical representation of the model
    # parallel_model = multi_gpu_model(model, gpus=2)
    # parallel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # parallel_model.summary()
    #
    # best_model_path = 'models_save/bilstm_model.h5'
    # # Set up callbacks
    # tensorboard = TensorBoard(log_dir='./bilstm_logs')
    # early_stopping = EarlyStopping(monitor='val_acc', patience=3)
    # checkpoint = ModelCheckpoint(best_model_path,
    #                              monitor='val_acc',
    #                              verbose=1,
    #                              save_best_only=True,
    #                              mode='max')
    # logger.info("train starting")
    # if train:
    #     history = parallel_model.fit([train_X, fea_train], train_Y, batch_size=train_batch_size, epochs=100,
    #                                  validation_data=([dev_X, fea_dev], dev_Y),
    #                                  callbacks=[checkpoint, early_stopping, tensorboard])
    #     bst_val_score = min(history.history['val_loss'])
    #
    # if evalution:
    #     model = load_model(best_model_path)
    #     preds_dev = model.predict([dev_X, fea_dev], batch_size=evalution_batch_size, verbose=1)
    #     preds_dev_labels = []
    #     for i in preds_dev:
    #         if i >= 0.5:
    #             preds_dev_labels.append(1)
    #         else:
    #             preds_dev_labels.append(0)
    #     preds_dev_labels = np.array(preds_dev_labels)
    #     # evalution
    #     logger.info("evalution starting")
    #     # pre_results = data_reader.predict(dev_X, "./bilstm_dataset/model").astype(dtype='int64')
    #     #
    #     precision = precision_score(dev_Y, preds_dev_labels, average='binary')
    #     recall = recall_score(dev_Y, preds_dev_labels, average='binary')
    #     f1 = f1_score(dev_Y, preds_dev_labels, average='binary')
    #     report = classification_report(dev_Y, preds_dev_labels)
    #     result_logs = "macro-averaged val_Precision = {:g}%\n".format(precision) + \
    #                   "macro-averaged val_Recall = {:g}%\n".format(recall) + \
    #                   "macro-averaged val_F1-score = {:g}%\n\n".format(f1) + report
    #     report_f = open("./results_output/bilstm_output/val_binary_classification_report.txt", 'w')
    #     report_f.write(result_logs)
    #     print(report)
    # if prediction:
    # # prediction
    #     model = load_model(best_model_path)
    #     logger.info("predication starting")
    #     preds_test = model.predict([test_X, fea_test], batch_size=prediction_batch_size, verbose=1)
    #     preds_test_out = open("./results_output/bilstm_output/test_predication.txt", 'w')
    #     preds_test_labels = []
    #     for i in preds_test.ravel():
    #         preds_test_out.write(str(i) + '\n')
    #         if i >= 0.5:
    #             preds_test_labels.append(1)
    #         else:
    #             preds_test_labels.append(0)
    #     test_df.drop("features_cat", axis=1)
    #     test_df["label_prediction"] = preds_test_labels
    #     test_df.to_csv("./results_output/bilstm_output/test_predication.csv",
    #                    columns=["row_id_in_raw_data", "subject", "object", "label_prediction", "sentence_seg"],
    #                    sep='\t', index=False)

