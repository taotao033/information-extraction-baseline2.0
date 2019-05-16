import json
from tqdm import tqdm
import pandas as pd

def load_p_eng_dict(dict_path):
    """load label dict from file"""
    p_eng2zh_dict_ = {}
    with open(dict_path, 'r') as fr:
        for li in fr:
            p_zh, p_eng = li.strip().split('\t')
            p_eng2zh_dict_[p_eng] = p_zh
    return p_eng2zh_dict_


print("Generating the final submit result, please wait for a moment.")
p_eng2zh_dict = load_p_eng_dict("../data/target_labels/relationship_zh2en")
test1_data_postag_sub_obj_path = "../entities_extraction/bert-chinese-ner/output/test1_data_postag_sub_obj.json"
rel_test_path = "../relationship_extraction/entity-aware-relation-classification_49_multilabels_rel/data/test_features.csv"

test1_50_predict_labels_path = '../relationship_extraction/entity-aware-relation-classification_49_multilabels_rel/' \
                               'predict_results/test1_49_multilabels_rel_predict_labels.txt'

with open(test1_data_postag_sub_obj_path) as test_f2:
    dic_all_49p = {}
    for row_id, line in enumerate(test_f2.readlines(), 1):
        dic_temp = {}
        line = line.strip()
        dic = json.loads(line)
        dic_temp["postag"] = dic["postag"]
        dic_temp["text"] = dic["text"]
        dic_temp["spo_list"] = []
        dic_all_49p[row_id] = dic_temp
    test_f2.close()

    lines_50_df = pd.read_csv(rel_test_path, sep='\t') # columns=["id", "sentence", "subject", "object","sub_name", "obj_name", "relation"]
    lines_50_predict_labels = [line.strip().replace('\n', "") for line in open(test1_50_predict_labels_path, 'r')]
    lines_50_df["predict_labels"] = lines_50_predict_labels

    count = 0
    count_empty_spo = 0
    rel_more_than1 = open("./rel_more_than1.txt", 'w')
    empty_spo_f = open("./empty_spo.txt", 'w')
    rules = [
        ['丈夫', '妻子'],
    ]

    sentence_list = lines_50_df["sentence"]
    sub_list = lines_50_df["sub_name"].tolist()
    obj_list = lines_50_df["obj_name"].tolist()
    p_list = lines_50_df["predict_labels"].tolist()
    row_id_list = lines_50_df["id"].tolist()
    for idx in range(len(lines_50_df)):
        if not p_list[idx]:
            count_empty_spo += 1
            # print(sentence_list[idx] + '\n')
            empty_spo_f.write(sentence_list[idx] + '\n')
            continue
        p = p_list[idx].split(' ')

        rel = [p_eng2zh_dict[p_] for p_ in p]
        # if rel in rules:
        #     label_id += 1
        #     print(rel)
        #     continue

        if len(rel) > 1:
            count += 1
            # print('\"' + "In" + rel_test_path + '\"' + " " + "line: " + str(idx+1) + "\t" + str(rel))
            rel_more_than1.write("The test set in" + " " + '\"' + rel_test_path + '\"' + " "
                                 + "line: " + str(idx+1) + '\t' + str(rel) + "\n" +
                                 "If model predictions are checked for errors, "
                                 "manually correct them in " + '\"' + test1_50_predict_labels_path
                                 + '\"' + " " + "line: " + str(idx+1) + '\n')
        for r in rel:
            spo_dict = {}
            sub = sub_list[idx]
            obj = obj_list[idx]
            spo_dict["predicate"] = r
            spo_dict["object_type"] = ''
            spo_dict["subject_type"] = ''
            spo_dict["object"] = obj
            spo_dict["subject"] = sub

            spo_list = dic_all_49p[row_id_list[idx]]["spo_list"]
            spo_list.append(spo_dict)
            dic_all_49p[row_id_list[idx]]["spo_list"] = spo_list
    print("The number of empty of spo_list is : " + str(count_empty_spo) + ", saved path: " + "\"" + "./empty_spo.txt" + "\"")
    print("In the test set, the number of the same subject-object relationship is greater than 1 is : "
          + str(count) + ", saved path: " + "\"" + "./rel_more_than1.txt" + "\"")
    rel_more_than1.write("In the test set, the number of the same subject-object relationship is greater than 1 is : "
                         + str(count))
    rel_more_than1.close()
    empty_spo_f.close()
    with open("./results/result.json", 'w') as result_49p:
        for dic in dic_all_49p.values():
            result_49p.write(json.dumps(dic, ensure_ascii=False))
            result_49p.write("\n")
    result_49p.close()







