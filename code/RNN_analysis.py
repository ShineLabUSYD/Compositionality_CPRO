# -*- coding: utf-8 -*-
"""
RNN_analysis.py

This script is the initial exploration of Recurrent Neural 
Networks from Yang et al. (2019) - Multitasking neural networks

Code made with the help of claude ai

Josh

"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import tensorflow as tf

# Set your path here
model_dir = "C:/Users/JoshB/Documents/Projects/compositionality_cole/RNN/train_all/0"

# ============================
# EXPLORE MODEL SETUP
# ============================

# Load hyperparameters
with open(os.path.join(model_dir, 'hp.json'), 'r') as f:
    hyperparameters = json.load(f)

print("Network has", hyperparameters['n_rnn'], "hidden units")


# Load performance for taskset1
with open(os.path.join(model_dir, 'taskset1_perf.pkl'), 'rb') as f:
    perf_data = pickle.load(f, encoding='latin-1')

print("Available performance metrics:", list(perf_data.keys()) if hasattr(perf_data, 'keys') else type(perf_data))


# Load training log
with open(os.path.join(model_dir, 'log.json'), 'r') as f:
    training_log = json.load(f)

# Plot training curve
plt.plot(training_log['cost_dm1'])
plt.title('Training Cost Over Time')
plt.xlabel('Training Step')
plt.ylabel('Cost')
plt.show()

# Plot learning curve
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(training_log['perf_avg'])
plt.title('Average Performance Over Training')
plt.xlabel('Training Step')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Plot a few individual task learning curves
task_keys = [k for k in training_log.keys() if k.startswith('perf_') and k != 'perf_avg' and k != 'perf_min']
for task in task_keys:
    plt.plot(training_log[task], label=task.replace('perf_', ''), alpha=0.7)
plt.title('Individual Task Learning')
plt.xlabel('Training Step')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Load in and check out other files
with open(os.path.join(model_dir, 'variance_epoch.pkl'), 'rb') as f:
    variance_epoch = pickle.load(f, encoding='latin-1')
    
with open(os.path.join(model_dir, 'variance_epoch_rr.pkl'), 'rb') as f:
    variance_epoch_rr = pickle.load(f, encoding='latin-1')
    
with open(os.path.join(model_dir, 'variance_rule.pkl'), 'rb') as f:
    variance_rule = pickle.load(f, encoding='latin-1')

with open(os.path.join(model_dir, 'variance_rule_rr.pkl'), 'rb') as f:
    variance_rule_rr = pickle.load(f, encoding='latin-1')
    
with open(os.path.join(model_dir, 'varytime_contextdm1.pkl'), 'rb') as f:
    varytime = pickle.load(f, encoding='latin-1')
    
with open(os.path.join(model_dir, 'taskset1_space.pkl'), 'rb') as f:
    taskset_space = pickle.load(f, encoding='latin-1')


# =======================
# Load in model
# =======================

from tensorflow.python.training import py_checkpoint_reader

NewCheckpointReader = py_checkpoint_reader.NewCheckpointReader
checkpoint_path = model_dir + "/model.ckpt"
reader = NewCheckpointReader(checkpoint_path)

var_to_shape_map = reader.get_variable_to_shape_map()

# Get the combined kernel
combined_kernel = reader.get_tensor('rnn/leaky_rnn_cell/kernel')
print(f"Combined kernel shape: {combined_kernel.shape}")  # Should be [341, 256]

# From hp.json: n_input = 85, n_rnn = 256
# So: 85 (input) + 256 (recurrent) = 341 total

# Split into input and recurrent weights:
W_in = combined_kernel[:85, :]    # First 85 rows = input weights [85, 256]
W_rec = combined_kernel[85:, :]   # Remaining rows = recurrent weights [256, 256]

# Get output weights
W_out = reader.get_tensor('output/weights')  # [256, 33]

# Get biases
b_rec = reader.get_tensor('rnn/leaky_rnn_cell/bias')  # [256]
b_out = reader.get_tensor('output/biases')            # [33]

print(f"W_in shape: {W_in.shape}")   # [85, 256]
print(f"W_rec shape: {W_rec.shape}") # [256, 256] 
print(f"W_out shape: {W_out.shape}") # [256, 33]


# List all variables to see what's available
var_names = reader.get_variable_to_shape_map().keys()
print("Variables in checkpoint:")
for name in var_names:
    print(f"  {name}: {reader.get_variable_to_shape_map()[name]}")
    



# ===============================
# EXPLORE MODEL NEURAL PATTERNS
# ===============================

# Load just one file to explore structure
with open(f"{model_dir}/variance_rule.pkl", 'rb') as f:
    rule_data = pickle.load(f, encoding='latin-1')

# Plot heatmap
figsize = (12,8)
h_var_all = rule_data['h_var_all']
task_names = rule_data['keys']

plt.figure(figsize=figsize)   
# Create heatmap
im = plt.imshow(h_var_all, cmap='viridis', aspect='auto', interpolation='nearest')
    
# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Variance', rotation=270, labelpad=20)
    
# Labels and title
plt.xlabel('Task Index')
plt.ylabel('Neuron Index')
plt.title('Neuron Variance Across Tasks')
    
# Add task names on x-axis if not too crowded
if len(task_names) <= 20:
    plt.xticks(range(len(task_names)), task_names, rotation=45, ha='right')
else:
    plt.xticks(range(0, len(task_names), 5))  # Show every 5th task
    
plt.tight_layout()
plt.show()

# Line plot
n_neurons = 256
figsize = (12,8)
# Randomly sample some neurons to visualize
n_total_neurons = h_var_all.shape[0]
sample_indices = np.random.choice(n_total_neurons, size=min(n_neurons, n_total_neurons), replace=False)
sample_indices = np.sort(sample_indices)  # Sort for easier reading
    
plt.figure(figsize=figsize)
    
# Plot each sampled neuron
for i, neuron_idx in enumerate(sample_indices):
    plt.plot(range(len(task_names)), h_var_all[neuron_idx, :], 
            marker='o', linewidth=1.5, markersize=4, 
            label=f'Neuron {neuron_idx}', alpha=0.8)
    
plt.xlabel('Task Index')
plt.ylabel('Variance')
plt.title(f'Variance Patterns for {len(sample_indices)} Sample Neurons')
    
# Add task names on x-axis
plt.xticks(range(len(task_names)), task_names, rotation=45, ha='right')
    
# Add legend (but not if too many lines)
if len(sample_indices) <= 10:
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Summary statistics
print("=== BASIC STATISTICS ===")
print(f"Overall variance range: {h_var_all.min():.6f} to {h_var_all.max():.6f}")
print(f"Overall mean variance: {h_var_all.mean():.6f}")
print(f"Overall std variance: {h_var_all.std():.6f}")
    
print("\nPer-task statistics:")
for i, task_name in enumerate(task_names):
    task_var = h_var_all[:, i]
    print(f"{task_name:15}: mean={task_var.mean():.6f}, std={task_var.std():.6f}, "
          f"min={task_var.min():.6f}, max={task_var.max():.6f}")
    
print("\nPer-neuron statistics (first 10 neurons):")
for i in range(min(10, h_var_all.shape[0])):
    neuron_var = h_var_all[i, :]
    print(f"Neuron {i:3d}: mean={neuron_var.mean():.6f}, std={neuron_var.std():.6f}, "
          f"min={neuron_var.min():.6f}, max={neuron_var.max():.6f}")


# Save as .mat file
mat_data = {
    'h_var_all': h_var_all,
    'task_names': task_names,
    'num_neurons': h_var_all.shape[0],
    'num_tasks': h_var_all.shape[1]
}
    
# Save to .mat file
mat_file_path = 'C:/Users/JoshB/Documents/Projects/compositionality_cole/RNN/Data/rnn_varRule.mat'
savemat(mat_file_path, mat_data)
    
# =======================================
# Get Task Variance data for all networks
# =======================================


path_dir = "C:/Users/JoshB/Documents/Projects/compositionality_cole/RNN/train_all/"
output_dir = 'C:/Users/JoshB/Documents/Projects/compositionality_cole/RNN/Data/'
n_networks = sum(1 for item in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, item)))

for net_id in range(n_networks):
    net_dir = os.path.join(path_dir, str(net_id))
        
    # Load in task variance file
    with open(f"{net_dir}/variance_rule.pkl", 'rb') as f:
        rule_data = pickle.load(f, encoding='latin-1')

    h_var_all = rule_data['h_var_all']
    task_names = rule_data['keys']
    
    # Save as .mat file
    mat_data = {
        'h_var_all': h_var_all,
        'task_names': task_names,
        'num_neurons': h_var_all.shape[0],
        'num_tasks': h_var_all.shape[1]
    }
        
    # Save to .mat file
    mat_file_path = output_dir + str(net_id) + "varRule.mat"
    savemat(mat_file_path, mat_data)


# =============================
# Get Performance for all RNNs
# =============================


path_dir = "C:/Users/JoshB/Documents/Projects/compositionality_cole/RNN/train_all/"
output_dir = 'C:/Users/JoshB/Documents/Projects/compositionality_cole/RNN/Data/'
n_networks = sum(1 for item in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, item)))

for net_id in range(n_networks):
    net_dir = os.path.join(path_dir, str(net_id))
        
    # Load in task variance file
    with open(f"{net_dir}/log.json", 'rb') as f:
        training_log = json.load(f)

    task_keys = [k for k in training_log.keys() if k.startswith('perf_') and k != 'perf_avg' and k != 'perf_min']
    task_data = {k.replace('perf_', ''): training_log[k] for k in task_keys}
           
    # Save to .mat file
    mat_file_path = output_dir + str(net_id) + "performance.mat"
    savemat(mat_file_path, task_data)