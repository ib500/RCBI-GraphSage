import os
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import json
import random
import shutil

# Parameters
input_folder = "./facebook"  # Replace with the folder containing the ego network files
output_folder = "./input_data"  # Replace with the folder where output files will be saved
val_ratio = 0.1  # Proportion of nodes for validation
test_ratio = 0.1  # Proportion of nodes for testing

# Step 1: Clear output folder
if os.path.exists(output_folder):
    print("Clearing output folder...")
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Step 2: Get all ego node IDs from the input folder
node_ids = set()
for file in os.listdir(input_folder):
    if file.endswith(".edges"):
        node_ids.add(file.split(".")[0])  # Extract nodeId from filename
        
print(f"There are {node_ids}")

# Step 3: Process each ego node
for node_id in node_ids:
    print(f"Processing node {node_id}...")

    # Paths to input files
    edges_file = os.path.join(input_folder, f"{node_id}.edges")
    circles_file = os.path.join(input_folder, f"{node_id}.circles")
    feat_file = os.path.join(input_folder, f"{node_id}.feat")
    egofeat_file = os.path.join(input_folder, f"{node_id}.egofeat")

    # Paths to output files
    output_prefix = os.path.join(output_folder, node_id)
    graph_file = f"{output_prefix}-G.json"
    id_map_file = f"{output_prefix}-id_map.json"
    class_map_file = f"{output_prefix}-class_map.json"
    feats_file = f"{output_prefix}-feats.npy"

    # Step 3.1: Load the graph
    print("  Loading graph...")
    G = nx.read_edgelist(edges_file, nodetype=int)

    # Add ego node to the graph
    ego_node = int(node_id)
    G.add_node(ego_node)
    for nodeID in G.nodes():
        if nodeID != ego_node:
            G.add_edge(ego_node, nodeID)

    # Annotate nodes with validation and test attributes
    for nodeID in G.nodes():
        G.node[nodeID]['val'] = random.random() < val_ratio
        G.node[nodeID]['test'] = random.random() < test_ratio

    # Ensure no overlap between validation and test sets
    for nodeID in G.nodes():
        if G.node[nodeID]['val'] and G.node[nodeID]['test']:
            G.node[nodeID]['test'] = False

    # Save graph to JSON
    print("  Saving graph...")
    with open(graph_file, "w") as f:
        json.dump(json_graph.node_link_data(G), f)

    # Step 3.2: Create ID map
    print("  Creating ID map...")
    id_map = {node: idx for idx, node in enumerate(G.nodes())}
    with open(id_map_file, "w") as f:
        json.dump(id_map, f)

    # Step 3.3: Load features and combine with ego features
    print("  Loading and saving features...")
    feat = np.loadtxt(feat_file)[:, 1:]  # Exclude the first column (node IDs)
    ego_feat = np.loadtxt(egofeat_file).reshape(1, -1)
    features = np.vstack([feat, ego_feat])
    np.save(feats_file, features)

    # Step 3.4: Process circles into class map
    print("  Processing circles...")
    class_map = {}

    # Read circles
    with open(circles_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            circle_name = parts[0]
            members = list(map(int, parts[1:]))
            for member in members:
                if member not in class_map:
                    class_map[member] = []
                class_map[member].append(circle_name)

    # Convert class map to one-hot encoding
    all_classes = sorted(set(c for classes in class_map.values() for c in classes))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    one_hot_class_map = {
        node: [1 if c in class_map[node] else 0 for c in all_classes]
        for node in class_map
    }

    # Save class map to JSON
    with open(class_map_file, "w") as f:
        json.dump(one_hot_class_map, f)

    print(f"  Node {node_id} processed!")

print("All nodes processed! Output folder has been cleared and refreshed.")
