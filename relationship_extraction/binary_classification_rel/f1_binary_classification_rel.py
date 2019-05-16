from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np

gold_file = './runs_logs/2019514/dev_gold.txt'
prediction_file = './runs_logs/2019514/logs/predictions.txt'

ture = []
with open(gold_file, 'r') as gold_f:
    for line in gold_f.readlines():
        ture.append(int(line))
gold_f.close()

pred = []
with open(prediction_file, 'r') as pred_f:
    for line in pred_f.readlines():
        pred.append(int(line.split('\t')[1]))
pred_f.close()

ture = np.array(ture)
pred = np.array(pred)
binary_classification_report = classification_report(ture, pred)
print(binary_classification_report)

with open('./runs_logs/2019514/binary_classification_report.txt', 'w') as report:
    report.write(binary_classification_report)
report.close()