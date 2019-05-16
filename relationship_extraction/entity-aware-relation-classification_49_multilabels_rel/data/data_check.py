from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_p_eng_dict(dict_path):
    """load label dict from file"""
    p_eng2zh_dict_ = {}
    with open(dict_path, 'r') as fr:
        for li in fr:
            p_zh, p_eng = li.strip().split('\t')
            p_eng2zh_dict_[p_eng] = p_zh
    return p_eng2zh_dict_


def rel_check(path, type="train"):
    p_eng2zh_dict = load_p_eng_dict("../../../data/target_labels/relationship_zh2en")

    rules = [
            ['丈夫', '妻子'],
            ['父亲', '母亲']
        ]

    count = 0
    lines = [line.strip() for line in open(path)]
    for idx in tqdm(range(0, len(lines), 4)):
        row_id = lines[idx].split("\t")[0]
        relation = lines[idx + 2]
        relation = relation.split("\t")
        rel = [p_eng2zh_dict[r] for r in relation]
        rel_reverse = rel.reverse()
        if rel in rules or rel_reverse in rules or (len(relation) > 1 and "OTHER" in relation) or len(relation) == 0:
            print("The" + " " + type + " " + "set in" + " " + '\"' + "./" + type + '\"' + " "
                  + "line: " + str(idx+1) + "\t" + "and in " + '\"' + "~/information-extraction-baseline2.0/data/"
                  + type + "_data.json" + '\"' + " " + "line: " + str(row_id) + '\t' + str(rel) + "\n")
            count += 1
    print("In the" + " " + type + " " + "set, Relationship labeling error number: " + str(count))


def statistics_after_segment_sentence_length(train_path='train_features.csv', dev_path='dev_features.csv'):
    train_df = pd.read_csv(train_path, sep='\t')
    train_sentence_length_list = [len(sent.split(' ')) for sent in train_df["sentence"].tolist()]

    text_length_train = pd.Series(train_sentence_length_list)
    cnt_srs_train = text_length_train.value_counts()
    print("The maximum words of the text in the training set is: " + str(text_length_train.max()))  # 202
    print("The average words of the text in the training set is: " + str(text_length_train.mean()))  # 32.17122053745684
    print("The std text words of the text in the training set is: " + str(text_length_train.std()))  # 17.86407963791644

    plt.figure(figsize=(20, 8))
    plt.bar(cnt_srs_train.index, cnt_srs_train.values, alpha=0.9, width=0.8, facecolor='lightskyblue',
            edgecolor='white', label='train', lw=1)
    # for a, b in zip(cnt_srs.index, cnt_srs.values):  # Display corresponding number
    #     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)
    # sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('The number of words of the text', fontsize=12)
    plt.title("The number of words of the text in training set")
    plt.xticks(rotation='horizontal')
    plt.legend(loc="upper right")
    plt.show()

    dev_df = pd.read_csv(dev_path, sep='\t')
    dev_sentence_length_list = [len(sent.split(' ')) for sent in dev_df["sentence"].tolist()]

    text_length_dev = pd.Series(dev_sentence_length_list)
    cnt_srs_dev = text_length_dev.value_counts()
    print("The maximum words of the text in the dev set is: " + str(text_length_dev.max()))  # 180
    print("The average words of the text in the dev set is: " + str(text_length_dev.mean()))  # 32.521032989310726
    print("The std text words of the text in the dev set is: " + str(text_length_dev.std()))  # 18.294552311938837

    plt.figure(figsize=(20, 8))
    plt.bar(cnt_srs_dev.index, cnt_srs_dev.values, alpha=0.9, width=0.8, facecolor='limegreen',
            edgecolor='white', label='dev', lw=1)
    # for a, b in zip(cnt_srs.index, cnt_srs.values):  # Display corresponding number
    #     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)
    # sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('The number of words of the text', fontsize=12)
    plt.title("The number of words of the text in dev set")
    plt.xticks(rotation='horizontal')
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    # rel_check(path="./train", type="train")
    # rel_check(path="./dev", type="dev")
    statistics_after_segment_sentence_length()
