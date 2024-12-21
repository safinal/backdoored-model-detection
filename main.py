import torch
from glob import glob
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

from preact_resnet import PreActResNet18
from backdoor_detection import backdoor_model_detector
from utils import threshold_scanning, accuracy_max


zeros = []
ones = []
labels = []
scores = []
for sample_path in tqdm(sorted(glob('/mnt/old/home/nafisi/temp/TRODO/eval_dataset/*'))):
    metadata = torch.load(os.path.join(sample_path, "metadata.pt"))
    model = PreActResNet18(num_classes=metadata['num_classes'])
    model.load_state_dict(torch.load(os.path.join(sample_path, "model.pt"), weights_only=True))
    pred = backdoor_model_detector(
        model=model, 
        num_classes=metadata['num_classes'], 
        test_images_folder_address=os.path.join(sample_path, metadata['test_images_folder_address']), 
        transformation=metadata['transformation']
    )
    labels.append(int(metadata['ground_truth']))
    scores.append(pred)
    if metadata['ground_truth']:
        ones.append(pred)
    else:
        zeros.append(pred)
    print(f"zero mean: {np.mean(zeros)}")
    print(f"zero std: {np.std(zeros)}")
    print(f"one mean: {np.mean(ones)}")
    print(f"one std: {np.std(ones)}")
    if len(set(labels)) > 1:
        print(f"\nROC {roc_auc_score(labels, scores)}")
    acc_max = accuracy_max(scores, labels)[0]
    print(f"Max Accuracy: {acc_max}")

np.save('ones.npy', np.array(ones))
np.save('zeros.npy', np.array(zeros))
np.save('scores.npy', np.array(scores))
np.save('labels.npy', np.array(labels))

good_thresholds = threshold_scanning(scores, labels)
print(good_thresholds)