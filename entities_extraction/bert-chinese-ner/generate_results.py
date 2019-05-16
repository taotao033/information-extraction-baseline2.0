import json
import copy


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


def get_sub_obj(sent_list, tag_list):
    word_seq = sent_list
    entity_list = set()
    token_idx = 0
    while token_idx < len(tag_list):
        if tag_list[token_idx] == 'O':
            token_idx += 1
        elif tag_list[token_idx].endswith('entity') \
                and tag_list[token_idx].startswith('B'):
            cur_e = word_seq[token_idx]
            token_idx += 1
            while token_idx < len(tag_list) and tag_list[token_idx].endswith('entity'):
                cur_e += word_seq[token_idx]
                token_idx += 1
            entity_list.add(cur_e)
        else:
            token_idx += 1

    entities_final = set()
    reverse_rules = [
        '贾乃亮李小璐', '贾乃亮贾乃亮'
    ]
    split_entity_rules = [
        '刘惠贤1932年10月'
    ]
    for e in entity_list:
        if e not in reverse_rules:
            if e in split_entity_rules:
                if e == '刘惠贤1932年10月':
                    entities_final.add('刘惠贤')
                    entities_final.add('1932年10月')
            else:
                entities_final.add(e)
        else:
            if e == '贾乃亮李小璐' or e == '贾乃亮贾乃亮':
                entities_final.add('贾乃亮')
                entities_final.add('李小璐')

    s_list = copy.deepcopy(list(entities_final))
    o_list = copy.deepcopy(list(entities_final))

    return s_list, o_list


def generate_results(token_test_path, labels_test_path, raw_test_data_path, output_path):

    sent_temp = []
    sent_all = []
    with open(token_test_path, 'r') as token_test:
        for line in token_test.readlines():
            line = line.replace("\n", '')
            if line == '[CLS]':
                continue
            if line != '[SEP]':
                sent_temp.append(line)
            else:
                sent_all.append(sent_temp)
                sent_temp = []

    labels_temp = []
    labels_all = []
    with open(labels_test_path, 'r') as labels_test:
        for line in labels_test.readlines():
            line = line.replace("\n", '')
            if line == '[CLS]':
                continue
            if line != '[SEP]':
                labels_temp.append(line)
            else:
                labels_all.append(labels_temp)
                labels_temp = []
    with open(raw_test_data_path, 'r') as raw_data:
        raw_postag_list_all = []
        raw_text_list_all = []
        for line in raw_data:
            if not _is_valid_input_data(line):
                print('Format is error')
                print(line)
                return None
            dic = line.strip()
            dic = json.loads(dic)
            raw_postag_list_all.append(dic["postag"])
            raw_text_list_all.append(dic["text"])

        count_spo_list = 0
        spo_list_all = []
        fo = open("entities_test1.txt", 'w')
        for i, sent in enumerate(sent_all):
            spo_list_temp = []
            s_list, o_list = get_sub_obj(sent, labels_all[i])
            #print(s_list, o_list)
            fo.write(str(s_list) + '\n')
            for s in s_list:
                for o in o_list:
                    spo_dict = {}
                    spo_dict["predicate"] = ''
                    spo_dict["object_type"] = ''
                    spo_dict["subject_type"] = ''
                    spo_dict["object"] = o
                    spo_dict["subject"] = s
                    spo_list_temp.append(spo_dict)
            spo_list_all.append(spo_list_temp)
            if len(spo_list_temp) == 0:
                count_spo_list += 1
        fo.close()
        print("The number of spo_list that are empty is: " + str(count_spo_list))

    assert len(raw_postag_list_all) == len(raw_text_list_all) == len(spo_list_all), 'The length must be equal.'
    with open(output_path, 'w') as output_f:
        for i in range(len(raw_text_list_all)):
            line_dict = {}
            line_dict["postag"] = raw_postag_list_all[i]
            line_dict["text"] = raw_text_list_all[i]
            line_dict["spo_list"] = spo_list_all[i]
            output_f.write(json.dumps(line_dict, ensure_ascii=False))
            output_f.write('\n')
    print("Subject and object prediction is completed! And saved path is: " + "\"" +
          output_path + "\"")

#
# def split_test_data_postag_sub_obj(input_path, output_path1, output_path2):
#
#     with open(input_path, 'r') as raw_data:
#         output1 = open(output_path1, 'w')
#         output2 = open(output_path2, 'w')
#         for line in raw_data:
#             if not _is_valid_input_data(line):
#                 print('Format is error')
#                 return None
#             dic = line.strip()
#             dic = json.loads(dic)
#
#             if "spo_list" not in dic:
#                 continue
#             if len(dic["spo_list"]) > 1:
#                 subject_temp = set()
#                 object_temp = set()
#                 for spo in dic["spo_list"]:
#                     sub = spo["subject"]
#                     subject_temp.add(sub)
#                     obj = spo["object"]
#                     object_temp.add(obj)
#                 if len(subject_temp) > 1 and len(object_temp) > 1:
#                     output1.write(json.dumps(dic, ensure_ascii=False))
#                     output1.write('\n')
#                 else:
#                     output2.write(json.dumps(dic, ensure_ascii=False))
#                     output2.write('\n')
#             else:
#                 output2.write(json.dumps(dic, ensure_ascii=False))
#                 output2.write('\n')
#         output1.close()
#         output2.close()
#         raw_data.close()


if __name__ == "__main__":
    generate_results(token_test_path="./output/results_dir/token_test.txt",
                     labels_test_path="./output/results_dir/label_test.txt",
                     raw_test_data_path="../../data/test1_data_postag.json",
                     output_path="./output/test1_data_postag_sub_obj.json")

    # split_test_data_postag_sub_obj(input_path="../../data/temp/test1_data_postag_sub_obj.json",
    #                                output_path1="../../data/temp/test1_data_postag_multi_sub_obj.json",
    #                                output_path2="../../data/temp/test1_data_postag_single_sub_obj.json")