import pandas as pd
pred_result = './predict_results/test1_binary_classification_rel_predict_labels.txt'
pred_list = []
with open(pred_result, 'r') as pred_f:
    for line in pred_f.readlines():
        pred_list.append(int(line))
pred_f.close()

test_df = pd.read_csv('./data/test_features.csv', sep='\t')
test_df["binary_rel"] = pred_list
test_df = test_df[test_df["binary_rel"] == 1]
test_df.to_csv("../entity-aware-relation-classification_49_multilabels_rel/data/test_features.csv",
               index=False, sep='\t')
