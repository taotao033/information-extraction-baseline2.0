import json
from tqdm import tqdm


spo_list_all_r1 = {}
with open('result_f1_0.81.json', "r") as r1:
    for row_id, li in enumerate(tqdm(r1.readlines()), 1):
        dic = li.strip()
        dic = json.loads(dic)
        spo_list_all_r1[row_id] = dic["spo_list"]
r1.close()

with open('result.json', "r") as r2:
    increase_spo_num_count = 0
    result_optimize = open("result_optimize.json", 'w')
    for row_id, li in enumerate(tqdm(r2.readlines()), 1):
        line = li.strip()
        dic = json.loads(line)
        list_temp = list(spo_list_all_r1[row_id])
        for spo in dic["spo_list"]:
            if spo not in list_temp:
                increase_spo_num_count += 1
                list_temp.append(spo)
        dic["spo_list"] = list_temp
        result_optimize.write(json.dumps(dic, ensure_ascii=False))
        result_optimize.write("\n")
    result_optimize.close()
    print("increase spo number: " + str(increase_spo_num_count))
r2.close()