#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from argparse import ArgumentParser
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier

def run_multi_label_classification(train_embeds, train_labels, test_embeds, test_labels):
    """
    Train a multi-label classifier (OneVsRest) on embeddings,
    then evaluate using micro-averaged F1.
    train_labels and test_labels can be NxK arrays (one-hot or multi-label).
    """
    # Baseline that predicts the most frequent combination
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(train_embeds, train_labels)
    dummy_pred_test = dummy.predict(test_embeds)
    dummy_f1 = f1_score(test_labels, dummy_pred_test, average="micro")

    # Multi-label logistic regression
    log = OneVsRestClassifier(SGDClassifier(loss="log", random_state=42))
    log.fit(train_embeds, train_labels)

    pred_test  = log.predict(test_embeds)
    pred_train = log.predict(train_embeds)

    test_f1  = f1_score(test_labels, pred_test, average="micro")
    train_f1 = f1_score(train_labels, pred_train, average="micro")

    print("\n==== Multi-label Classification Results ====")
    print(f"Train F1 (micro):  {train_f1:.4f}")
    print(f"Test F1 (micro):   {test_f1:.4f}")
    print(f"Dummy baseline:    {dummy_f1:.4f}")

def main():
    parser = ArgumentParser(description="Evaluate multi-label embeddings for the supervised GraphSAGE version.")
    parser.add_argument("--dataset_dir", required=True,
                        help="Folder containing *-class_map.json, etc.")
    parser.add_argument("--prefix", required=True,
                        help="Prefix for your dataset (e.g. 100129275726588145876).")
    parser.add_argument("--embeds_dir", required=True,
                        help="Folder containing train.npy, train.txt, test.npy, test.txt, etc.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.rstrip("/")
    prefix      = args.prefix
    embeds_dir  = args.embeds_dir.rstrip("/")

    # 1) Load the class map (multi-label)
    class_map_path = f"{dataset_dir}/{prefix}-class_map.json"
    print(f"Loading class map from: {class_map_path}")
    with open(class_map_path) as f:
        labels_dict = json.load(f)  # e.g. { "node_id": [0,1], ... }

    # 2) Load train embeddings
    train_npy_path = f"{embeds_dir}/train.npy"
    train_txt_path = f"{embeds_dir}/train.txt"
    print(f"Loading train embeddings from: {train_npy_path}")

    train_embeds = np.load(train_npy_path)
    with open(train_txt_path, "r") as f:
        train_nodes = [line.strip() for line in f]

    # Convert labels to a NumPy array
    train_labels = np.array([labels_dict[str(n)] for n in train_nodes])

    print(f"Train embeddings shape: {train_embeds.shape}")
    print(f"Train labels shape:     {train_labels.shape}")

    # 3) Load test embeddings
    test_npy_path = f"{embeds_dir}/test.npy"
    test_txt_path = f"{embeds_dir}/test.txt"
    print(f"Loading test embeddings from: {test_npy_path}")

    test_embeds = np.load(test_npy_path)
    with open(test_txt_path, "r") as f:
        test_nodes = [line.strip() for line in f]

    test_labels = np.array([labels_dict[str(n)] for n in test_nodes])

    print(f"Test embeddings shape:  {test_embeds.shape}")
    print(f"Test labels shape:      {test_labels.shape}")

    # 4) Run multi-label classification
    run_multi_label_classification(train_embeds, train_labels, test_embeds, test_labels)

if __name__ == "__main__":
    main()
