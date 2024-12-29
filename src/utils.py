import numpy as np
import torch
import random


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy_max(scores, labels):
    accuracies = []
    for th in np.linspace(min(scores), max(scores), 100000):
        accuracies.append(np.sum((np.array(scores) > th) == np.array(labels)) / (len(labels)))
    greater_than_threshold = max(accuracies)

    accuracies = []
    for th in np.linspace(min(scores), max(scores), 100000):
        accuracies.append(np.sum((np.array(scores) < th) == np.array(labels)) / (len(labels)))
    less_than_threshold = max(accuracies)

    return (greater_than_threshold, True) if greater_than_threshold > less_than_threshold else (less_than_threshold, False)

def threshold_scanning(scores, labels):
    acc_max, is_greater_than_threshold = accuracy_max(scores, labels)
    print(f"Max Accuracy: {acc_max}")
    good_thresholds = []
    for th in np.linspace(min(scores), max(scores), 100000):
        if is_greater_than_threshold:
            acc = np.sum((np.array(scores) > th) == np.array(labels)) / len(labels)
        else:
            acc = np.sum((np.array(scores) < th) == np.array(labels)) / len(labels)
        if acc == acc_max:
            good_thresholds.append(float(th))
    return good_thresholds
