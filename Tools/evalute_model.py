from collections import defaultdict
import numpy as np

def precision_at_k(preds, trues, k):
    correct = sum([1 for label in preds[:k] if label in trues])
    return correct / k

def dcg_at_k(preds, trues, k):
    return sum([1 / np.log2(i + 2) for i, label in enumerate(preds[:k]) if label in trues])

def idcg_at_k(trues, k):
    return sum([1 / np.log2(i + 2) for i in range(min(len(trues), k))])

def ndcg_at_k(preds, trues, k):
    idcg = idcg_at_k(trues, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(preds, trues, k) / idcg

def evaluate_metrics(true_path, pred_path, max_k=15):
    with open(true_path, 'r') as f:
        true_lines = f.readlines()[1:]  # skip first line

    with open(pred_path, 'r') as f:
        pred_lines = f.readlines()[1:]  # skip first line

    precision_scores = defaultdict(list)
    ndcg_scores = defaultdict(list)

    for true_line, pred_line in zip(true_lines, pred_lines):
        true_labels = {int(item.split(':')[0]) for item in true_line.strip().split()}
        pred_dict = {int(k): float(v) for k, v in (item.split(":") for item in pred_line.strip().split())}
        sorted_preds = [k for k, v in sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)]

        for k in range(1, max_k + 1):
            precision_scores[k].append(precision_at_k(sorted_preds, true_labels, k))
            ndcg_scores[k].append(ndcg_at_k(sorted_preds, true_labels, k))

    avg_precision = {k: np.mean(precision_scores[k]) for k in range(1, max_k + 1)}
    avg_ndcg = {k: np.mean(ndcg_scores[k]) for k in range(1, max_k + 1)}

    return avg_precision, avg_ndcg
