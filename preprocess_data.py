#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import json
import random
import shutil

# ==============================================================================
# PARAMETERS
# ==============================================================================
input_folder = "./gplus"            # Folder with *.edges, *.circles, *.feat, *.egofeat
output_folder = "./gplus_pdata"     # Where we'll store each ego's output
val_ratio = 0.1                     # Fraction of nodes to mark as validation
test_ratio = 0.1                    # Fraction of nodes to mark as test
num_walks = 10                      # Number of random walks per node
walk_length = 40                    # Length of each random walk
# We ALWAYS use (root, context) pairs for the random walks:
use_pairs_instead_of_full_walk = True

# ==============================================================================
# 1) CLEAR OUTPUT FOLDER
# ==============================================================================
if os.path.exists(output_folder):
    print(f"Removing previous contents of '{output_folder}'...")
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# ==============================================================================
# 2) DISCOVER ALL EGO-IDs
# ==============================================================================
node_ids = {
    file.split(".")[0]
    for file in os.listdir(input_folder)
    if file.endswith(".edges")
}
node_ids = sorted(node_ids)
print(f"Found {len(node_ids)} ego-nets in '{input_folder}': {node_ids}")

# ==============================================================================
# 3) PROCESS EACH EGO-NET INDIVIDUALLY
# ==============================================================================
for ego_id in node_ids:
    print("\n========================================================")
    print(f"Processing ego-node: {ego_id}")

    # Paths for input files
    edges_file   = os.path.join(input_folder, f"{ego_id}.edges")
    circles_file = os.path.join(input_folder, f"{ego_id}.circles")
    feat_file    = os.path.join(input_folder, f"{ego_id}.feat")
    egofeat_file = os.path.join(input_folder, f"{ego_id}.egofeat")

    # Check required files
    missing = [f for f in [edges_file, circles_file, feat_file, egofeat_file] if not os.path.exists(f)]
    if missing:
        print(f"  Missing required files for ego {ego_id}: {missing}")
        print("  Skipping this ego.")
        continue

    # Prefix for output
    prefix = os.path.join(output_folder, ego_id)
    graph_file     = f"{prefix}-G.json"
    id_map_file    = f"{prefix}-id_map.json"
    class_map_file = f"{prefix}-class_map.json"
    feats_file     = f"{prefix}-feats.npy"
    walks_file     = f"{prefix}-walks.txt"

    # -------------------------------------------------------------------------
    # 3.1) LOAD THE EGO-NET GRAPH
    # -------------------------------------------------------------------------
    print("  Loading the graph...")
    G_tmp = nx.read_edgelist(edges_file, nodetype=int, data=False, create_using=nx.Graph())

    # Create a new graph from G_tmp
    G = nx.Graph()
    G.add_nodes_from(G_tmp.nodes(data=True))
    G.add_edges_from(G_tmp.edges())
    del G_tmp

    # Add the ego node itself and connect it to all nodes in G (optional, but typical)
    ego_node = int(ego_id)
    G.add_node(ego_node)
    for n in G.nodes():
        if n != ego_node:
            G.add_edge(ego_node, n)

    # -------------------------------------------------------------------------
    # 3.2) SPLIT NODES INTO TRAIN/VAL/TEST
    # -------------------------------------------------------------------------
    nodes = list(G.nodes())
    val_size  = int(len(nodes) * val_ratio)
    test_size = int(len(nodes) * test_ratio)

    val_nodes = set(random.sample(nodes, val_size))
    remaining_for_test = [n for n in nodes if n not in val_nodes]
    test_nodes = set(random.sample(remaining_for_test, test_size))

    for node in nodes:
        G.node[node]['val']  = (node in val_nodes)
        G.node[node]['test'] = (node in test_nodes)

    # -------------------------------------------------------------------------
    # 3.3) SAVE THE GRAPH TO JSON
    # -------------------------------------------------------------------------
    print("  Saving graph to JSON...")
    with open(graph_file, "w") as f:
        json.dump(json_graph.node_link_data(G), f)

    # -------------------------------------------------------------------------
    # 3.4) CREATE ID MAP
    # -------------------------------------------------------------------------
    print("  Creating ID map...")
    id_map = {node: idx for idx, node in enumerate(G.nodes())}
    with open(id_map_file, "w") as f:
        json.dump(id_map, f)

    # -------------------------------------------------------------------------
    # 3.5) LOAD FEATURES & EGO-FEATURES
    # -------------------------------------------------------------------------
    print("  Loading and saving features...")

    # Force at least 2D shape to avoid "IndexError: too many indices"
    feat_data = np.loadtxt(feat_file, ndmin=2)
    ego_data  = np.loadtxt(egofeat_file, ndmin=2)

    print("    feat_data shape:", feat_data.shape)
    print("    ego_data shape: ", ego_data.shape)

    # If there's only 1 column in feat_data, shape might be (N,) or (N,1).
    if feat_data.ndim < 2 or feat_data.shape[1] < 2:
        print(f"  The .feat file for ego {ego_id} doesn't have multiple columns (ID + features). Skipping.")
        continue

    # Here we handle two scenarios:
    # 1) .egofeat DOES include the ID in the first column (same number of columns as .feat)
    # 2) .egofeat does NOT include the ID, so it has (feat_data.shape[1] - 1) columns

    expected_with_id    = feat_data.shape[1]  # If .egofeat includes ID
    expected_without_id = feat_data.shape[1] - 1  # If .egofeat excludes ID

    if ego_data.shape[1] == expected_with_id:
        # CASE 1: .egofeat includes ID
        # The first column is the ID, the rest are features
        node_ids_in_feat = feat_data[:, 0].astype(int)     # shape (N,)
        node_feats_only  = feat_data[:, 1:]               # shape (N, K-1)
        ego_feats_only   = ego_data[0, 1:]                # skip the first col for ego as well
        print("  Detected .egofeat has an ID column. Using ego_data[0,1:].")

    elif ego_data.shape[1] == expected_without_id:
        # CASE 2: .egofeat does NOT include ID
        # So all columns in ego_data are already features
        node_ids_in_feat = feat_data[:, 0].astype(int)     # shape (N,)
        node_feats_only  = feat_data[:, 1:]               # shape (N, K-1)
        ego_feats_only   = ego_data[0, :]                 # take all columns as features
        print("  Detected .egofeat has NO ID column. Using ego_data[0,:].")

    else:
        # If it doesn't match either, we skip
        print(f"  Mismatch: feat has {feat_data.shape[1]} columns, "
            f"but egofeat has {ego_data.shape[1]}. Skipping.")
        continue

    # Stack them: (non-ego) + (ego)
    features = np.vstack([node_feats_only, ego_feats_only.reshape(1, -1)])
    np.save(feats_file, features)


    # -------------------------------------------------------------------------
    # 3.6) PROCESS CIRCLES => CLASS_MAP
    # -------------------------------------------------------------------------
    print("  Processing circles (class_map)...")
    class_map = {}
    with open(circles_file, "r") as fcir:
        for line in fcir:
            parts = line.strip().split()
            circle_label = parts[0]
            members = list(map(int, parts[1:]))

            for mem in members:
                if mem not in class_map:
                    class_map[mem] = []
                if circle_label not in class_map[mem]:
                    class_map[mem].append(circle_label)

    # Convert circle labels to one-hot
    all_labels = set()
    for mem, labels in class_map.items():
        all_labels.update(labels)
    all_labels = sorted(all_labels)
    label_index = {lb: i for i, lb in enumerate(all_labels)}

    one_hot_class_map = {}
    for n in G.nodes():
        label_vec = [0]*len(all_labels)
        if n in class_map:
            for lb in class_map[n]:
                idx = label_index[lb]
                label_vec[idx] = 1
        one_hot_class_map[n] = label_vec

    with open(class_map_file, "w") as f:
        json.dump(one_hot_class_map, f)

    # -------------------------------------------------------------------------
    # 3.7) GENERATE RANDOM WALKS AS (ROOT, CONTEXT) PAIRS
    # -------------------------------------------------------------------------
    print("  Generating random walks => (root, context) pairs...")
    with open(walks_file, "w") as f_out:
        for _ in range(num_walks):
            for node in G.nodes():
                walk = [node]
                for _ in range(walk_length - 1):
                    neighbors = list(G.neighbors(walk[-1]))
                    if not neighbors:
                        break
                    walk.append(random.choice(neighbors))

                # Write each step as (root, context)
                root = walk[0]
                for cnode in walk[1:]:
                    f_out.write(f"{root}\t{cnode}\n")

    print(f"  --> Ego {ego_id} processed successfully!")

# END LOOP
print("\nAll done! Processed data stored in:", output_folder)
print("You can now run GraphSAGE on each prefix individually, e.g.:")
print(f"  unsupervised_train.py --train_prefix {output_folder}/1234")
