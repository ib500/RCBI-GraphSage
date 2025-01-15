#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
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
    # Baseline (dummy) that always predicts the most frequent combination
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(train_embeds, train_labels)
    dummy_pred_test = dummy.predict(test_embeds)
    dummy_f1 = f1_score(test_labels, dummy_pred_test, average="micro")

    # Multi-label logistic regression
    log = OneVsRestClassifier(SGDClassifier(loss="log", random_state=42))
    log.fit(train_embeds, train_labels)

    pred_test = log.predict(test_embeds)
    pred_train = log.predict(train_embeds)

    test_f1  = f1_score(test_labels, pred_test, average="micro")
    train_f1 = f1_score(train_labels, pred_train, average="micro")

    print("\n==== Multi-label Classification Results ====")
    print(f"Train F1 (micro):  {train_f1:.4f}")
    print(f"Test F1 (micro):   {test_f1:.4f}")
    print(f"Dummy baseline:    {dummy_f1:.4f}")

def main():
    parser = ArgumentParser(description="Evaluate multi-label embeddings on a single Google+ ego-network.")
    parser.add_argument("--dataset_dir", required=True,
                        help="Folder with prefix-G.json, prefix-class_map.json, etc.")
    parser.add_argument("--embed_dir", required=True,
                        help="Folder with val.npy and val.txt from GraphSAGE output.")
    parser.add_argument("--ego_prefix", required=True,
                        help="Ego prefix ID (e.g. 100129275726588145876).")
    parser.add_argument("--split_key", default="test",
                        help="Which key to use for evaluation split: 'test' or 'val'. Default is 'test'.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.rstrip("/")
    embed_dir   = args.embed_dir.rstrip("/")
    prefix      = args.ego_prefix
    split_key   = args.split_key

    # 1) Load the graph
    g_json_path = f"{dataset_dir}/{prefix}-G.json"
    print(f"Loading graph from {g_json_path} ...")
    with open(g_json_path) as f:
        G = json_graph.node_link_graph(json.load(f))

    # 2) Load class_map (which likely has multi-label vectors like [0,1], [1,0], etc.)
    c_map_path = f"{dataset_dir}/{prefix}-class_map.json"
    print(f"Loading class map from {c_map_path} ...")
    with open(c_map_path) as f:
        labels_dict = json.load(f)  # { "node_id": [0,1], ... }

    # 3) Gather train/test IDs
    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids  = [n for n in G.nodes() if G.node[n][split_key]]

    print(f"Found {len(train_ids)} train nodes, {len(test_ids)} {split_key} nodes.")

    # Convert multi-label arrays to np.array
    train_labels = np.array([labels_dict[str(n)] for n in train_ids])
    test_labels  = np.array([labels_dict[str(n)] for n in test_ids])

    # 4) Load embeddings from the unsupervised GraphSAGE output
    val_npy_path = f"{embed_dir}/val.npy"
    val_txt_path = f"{embed_dir}/val.txt"
    print(f"Loading embeddings from {val_npy_path} ...")
    embeds = np.load(val_npy_path)
    with open(val_txt_path) as fp:
        lines = [line.strip() for line in fp]
    id_map = {node_id: i for i, node_id in enumerate(lines)}

    # 5) Align embeddings
    # Note: ensure IDs match the format in val.txt (often strings)
    def get_embed(nid):
        return embeds[id_map[str(nid)]]

    train_embeds = np.array([get_embed(n) for n in train_ids])
    test_embeds  = np.array([get_embed(n) for n in test_ids])

    # 6) Run multi-label classification (no GraphSAGE retraining needed!)
    run_multi_label_classification(train_embeds, train_labels, test_embeds, test_labels)

if __name__ == "__main__":
    main()
