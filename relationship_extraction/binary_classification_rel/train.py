import os
import time
import numpy as np
import tensorflow as tf

import data_helpers
from configure import FLAGS
from logger import Logger
from model.entity_att_lstm import EntityAttentionLSTM
import utils
import json
from tqdm import tqdm
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train():
    with tf.device('/cpu:0'):
        train_text, train_y, train_e1, train_e2, train_pos1, train_pos2 = \
            data_helpers.load_data_and_labels(FLAGS.train_path, FLAGS.train_features_path, type="train")
    with tf.device('/cpu:0'):
        dev_text, dev_y, dev_e1, dev_e2, dev_pos1, dev_pos2 = \
            data_helpers.load_data_and_labels(FLAGS.dev_path, FLAGS.dev_features_path, type="dev")
    with tf.device('/cpu:0'):
        test_text, test_y, test_e1, test_e2, test_pos1, test_pos2 = \
            data_helpers.load_data_and_labels(FLAGS.test_path, FLAGS.test_features_path, type="test")

    # Build vocabulary
    # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # dimension = MAX_SENTENCE_LENGTH
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    vocab_processor.fit(train_text + dev_text + test_text)
    # Saving vocabulary
    vocab_dict = vocab_processor.vocabulary_._mapping
    if not os.path.exists("./vocab_word2id.dict"):
        with open("./vocab_word2id.dict", encoding="utf-8", mode="w") as vocab_f:
            json.dump(vocab_dict, vocab_f)

    train_x = np.array(list(vocab_processor.transform(train_text)))
    dev_x = np.array(list(vocab_processor.transform(dev_text)))
    test_x = np.array(list(vocab_processor.transform(test_text)))
    train_text = np.array(train_text)
    dev_text = np.array(dev_text)
    test_text = np.array(test_text)

    print("\nText Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("train_x = {0}".format(train_x.shape))
    print("train_y = {0}".format(train_y.shape))
    print("dev_x = {0}".format(dev_x.shape))
    print("dev_y = {0}".format(dev_y.shape))
    print("test_x = {0}".format(test_x.shape))
    print("test_y = {0}".format(test_y.shape))

    # Example: pos1[3] = [-2 -1  0  1  2   3   4 999 999 999 ... 999]
    # [95 96 97 98 99 100 101 999 999 999 ... 999]
    # =>
    # [11 12 13 14 15  16  21  17  17  17 ...  17]
    # dimension = MAX_SENTENCE_LENGTH
    pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    pos_vocab_processor.fit(train_pos1 + train_pos2 + dev_pos1 + dev_pos2)
    train_p1 = np.array(list(pos_vocab_processor.transform(train_pos1)))
    train_p2 = np.array(list(pos_vocab_processor.transform(train_pos2)))
    dev_p1 = np.array(list(pos_vocab_processor.transform(dev_pos1)))
    dev_p2 = np.array(list(pos_vocab_processor.transform(dev_pos2)))
    test_p1 = np.array(list(pos_vocab_processor.transform(test_pos1)))
    test_p2 = np.array(list(pos_vocab_processor.transform(test_pos2)))
    print("\nPosition Vocabulary Size: {:d}".format(len(pos_vocab_processor.vocabulary_)))
    print("train_p1 = {0}".format(train_p1.shape))
    print("dev_p1 = {0}".format(dev_p1.shape))
    print("test_p1 = {0}".format(test_p1.shape))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = EntityAttentionLSTM(
                sequence_length=train_x.shape[1],
                num_classes=1,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_size,
                pos_vocab_size=len(pos_vocab_processor.vocabulary_),
                pos_embedding_size=FLAGS.pos_embedding_size,
                hidden_size=FLAGS.hidden_size,
                num_heads=FLAGS.num_heads,
                attention_size=FLAGS.attention_size,
                use_elmo=(FLAGS.embeddings == 'elmo'),
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
            gvs = optimizer.compute_gradients(model.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            # timestamp = str(int(time.time()))
            localtime = time.localtime(time.time())
            timestamp = str(localtime.tm_year) + str(localtime.tm_mon) + str(localtime.tm_mday)
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_logs", timestamp))
            print("\nWriting to {}\n".format(out_dir))

            # Logger

            logger = Logger(out_dir)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            pos_vocab_processor.save(os.path.join(out_dir, "pos_vocab"))

            ##Initialize all variables
            sess.run(tf.global_variables_initializer())

            if FLAGS.embeddings == "word2vec":
                pretrain_W = utils.load_word2vec('resource/GoogleNews-vectors-negative300.bin',
                                                 FLAGS.embedding_size, vocab_processor)
                sess.run(model.W_text.assign(pretrain_W))
                print("Success to load pre-trained word2vec model!\n")
            elif FLAGS.embeddings == "baidubaike":
                pretrain_W = utils.load_baidubaike('resource/baidubaike_word_vectors.dim300',
                                                   FLAGS.embedding_size, vocab_processor)
                sess.run(model.W_text.assign(pretrain_W))
                print("Success to load pre-trained baidubaike model!\n")
            elif FLAGS.embeddings == "glove100":
                pretrain_W = utils.load_glove('resource/glove.6B.100d.txt', FLAGS.embedding_size, vocab_processor)
                sess.run(model.W_text.assign(pretrain_W))
                print("Success to load pre-trained glove100 model!\n")
            elif FLAGS.embeddings == "glove300":
                pretrain_W = utils.load_glove('resource/glove.840B.300d.txt', FLAGS.embedding_size, vocab_processor)
                sess.run(model.W_text.assign(pretrain_W))
                print("Success to load pre-trained glove300 model!\n")
            elif FLAGS.embeddings == "bert768":
                pretrain_W = utils.load_bert('resource/lic2019_bert_vector.768d.txt', FLAGS.embedding_size,
                                             vocab_processor)
                sess.run(model.W_text.assign(pretrain_W))
                print("Success to load pre-trained bert model!\n")

            # Generate batches
            train_batches = data_helpers.batch_iter(list(zip(train_x, train_y, train_text,
                                                             train_e1, train_e2, train_p1, train_p2)),
                                                    FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            best_f1 = 0.0  # For save checkpoint(model)
            for train_batch in train_batches:
                train_bx, train_by, train_btxt, train_be1, train_be2, train_bp1, train_bp2 = zip(*train_batch)
                feed_dict = {
                    model.input_x: train_bx,
                    model.input_y: train_by,
                    model.input_text: train_btxt,
                    model.input_e1: train_be1,
                    model.input_e2: train_be2,
                    model.input_p1: train_bp1,
                    model.input_p2: train_bp2,
                    model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
                    model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    logger.logging_train(step, loss, accuracy)

                # Evaluation

                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    # Generate batches
                    dev_batches = data_helpers.batch_iter(list(zip(dev_x, dev_y, dev_text,
                                                                   dev_e1, dev_e2, dev_p1, dev_p2)),
                                                          FLAGS.batch_size, 1, shuffle=False)
                    # Training loop. For each batch...
                    losses = 0.0
                    accuracy = 0.0
                    predictions = []
                    iter_cnt = 0
                    for dev_batch in tqdm(dev_batches):
                        dev_bx, dev_by, dev_btxt, dev_be1, dev_be2, dev_bp1, dev_bp2 = zip(*dev_batch)
                        feed_dict = {
                            model.input_x: dev_bx,
                            model.input_y: dev_by,
                            model.input_text: dev_btxt,
                            model.input_e1: dev_be1,
                            model.input_e2: dev_be2,
                            model.input_p1: dev_bp1,
                            model.input_p2: dev_bp2,
                            model.emb_dropout_keep_prob: 1.0,
                            model.rnn_dropout_keep_prob: 1.0,
                            model.dropout_keep_prob: 1.0
                        }
                        loss, acc, pred = sess.run(
                            [model.loss, model.accuracy, model.predictions], feed_dict)
                        losses += loss
                        accuracy += acc
                        predictions += pred.tolist()
                        iter_cnt += 1
                    losses /= iter_cnt
                    accuracy /= iter_cnt
                    predictions = np.array(predictions, dtype='int')

                    # dev_true_label = []
                    # for true_label in dev_relation_truely:
                    #     dev_true_label.append(utils.en_class2id()[true_label])
                    # dev_relation_truely_ = np.array(dev_true_label, dtype='int')

                    logger.logging_eval(step, loss, accuracy, dev_y, predictions)

                    # Model checkpoint
                    if best_f1 < logger.best_f1:
                        best_f1 = logger.best_f1
                        # predict
                        print("\nPredict:")
                        test_batches = data_helpers.batch_iter(list(zip(test_x, test_y, test_text, test_e1,
                                                                        test_e2, test_p1, test_p2)),
                                                               FLAGS.test_batch_size, 1, shuffle=False)
                        predictions_test = []
                        for test_batch in tqdm(test_batches):
                            test_bx, test_by, test_btxt, test_be1, test_be2, test_bp1, test_bp2 = zip(*test_batch)
                            feed_dict = {
                                model.input_x: test_bx,
                                model.input_y: test_by,
                                model.input_text: test_btxt,
                                model.input_e1: test_be1,
                                model.input_e2: test_be2,
                                model.input_p1: test_bp1,
                                model.input_p2: test_bp2,
                                model.emb_dropout_keep_prob: 1.0,
                                model.rnn_dropout_keep_prob: 1.0,
                                model.dropout_keep_prob: 1.0
                            }
                            loss, acc, pred_t = sess.run(
                                [model.loss, model.accuracy, model.predictions], feed_dict)
                            predictions_test += pred_t.tolist()
                        predictions_test = np.array(predictions_test, dtype='int')
                        # predictions_test_list = predictions_test.tolist()

                        with open("./predict_results/test1_binary_classification_rel_predict_labels.txt", 'w') as test_pre:
                            for pre_label in predictions_test.reshape(len(predictions_test),):
                                test_pre.write(str(pre_label) + "\n")
                        test_pre.close()
                        path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(best_f1), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == "__main__":

    tf.app.run()
