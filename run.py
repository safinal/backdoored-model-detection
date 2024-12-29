import torch
from glob import glob
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import json
import argparse

from src.config import ConfigManager
from src.preact_resnet import PreActResNet18
from src.backdoor_detection import backdoor_model_detector
from src.utils import threshold_scanning, accuracy_max


def parse_args():
    parser = argparse.ArgumentParser(description="Rayan International AI Contest: Backdoored Model Detection")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to the config file")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config_path = args.config
    config = ConfigManager(config_path)  # Initialize the singleton with the config file


    backdoored_models_scores = []
    clean_models_scores = []
    labels = []
    scores = []
    for sample_path in tqdm(sorted(glob(config.get("dataset_path").rstrip('/') + '/*'))):
        metadata = torch.load(os.path.join(sample_path, "metadata.pt"))
        model = PreActResNet18(num_classes=metadata['num_classes'])
        model.load_state_dict(torch.load(os.path.join(sample_path, "model.pt"), weights_only=True))
        pred = backdoor_model_detector(
            model=model, 
            num_classes=metadata['num_classes'], 
            test_images_folder_address=os.path.join(sample_path, metadata['test_images_folder_address']), 
            transformation=metadata['transformation']
        )
        labels.append(int(not metadata['ground_truth']))
        scores.append(pred)
        if metadata['ground_truth']:
            clean_models_scores.append(pred)
        else:
            backdoored_models_scores.append(pred)

        results = {
            "num_samples": len(scores),
            "clean_models_scores": clean_models_scores,
            "backdoored_models_scores": backdoored_models_scores,
            "clean_models_scores_mean": np.mean(clean_models_scores),
            "clean_models_scores_std": np.std(clean_models_scores),
            "backdoored_models_scores_mean": np.mean(backdoored_models_scores),
            "backdoored_models_scores_std": np.std(backdoored_models_scores),
            "scores": scores,
            "labels": labels,
            "roc_auc": roc_auc_score(labels, scores) if len(set(labels)) > 1 else None,
            "acc_max": accuracy_max(scores, labels)[0],
            "good_thresholds": threshold_scanning(scores, labels)
        }

        with open(config.get("results_path"), 'w') as f:
            json.dump(results, f, indent=4)

        if config.get("verbose"):
            print('-'*100)
            for key, val in results.items():
                if key != "good_thresholds":
                    print(f"{key}: {val}")
            print('-'*100)

if __name__ == "__main__":
    main()