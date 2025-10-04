# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 10:58:04 2025

@author: JoshB
"""

# -*- coding: utf-8 -*-
"""
Enhanced neural activity generator for Yang multitask models
Includes .mat file export and time series plotting
"""

import os
import sys
import json
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.io import savemat

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def analyze_model_directory(model_dirs):
    """Analyze multiple model directories to understand training setup"""
    
    print("Analyzing model directories...")
    print("="*60)
    
    model_info = {}
    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            continue
            
        model_name = os.path.basename(model_dir)
        hp_file = os.path.join(model_dir, 'hp.json')
        
        if os.path.exists(hp_file):
            with open(hp_file, 'r') as f:
                hp = json.load(f)
            
            model_info[model_name] = {
                'rules': hp.get('rules', []),
                'rule_trains': hp.get('rule_trains', []),
                'ruleset': hp.get('ruleset', 'unknown'),
                'n_rnn': hp.get('n_rnn', 0),
                'seed': hp.get('seed', None)
            }
    
    # Analyze patterns
    rulesets = {}
    seeds = {}
    for name, info in model_info.items():
        ruleset = info['ruleset']
        seed = info['seed']
        
        if ruleset not in rulesets:
            rulesets[ruleset] = []
        rulesets[ruleset].append(name)
        
        if seed not in seeds:
            seeds[seed] = []
        seeds[seed].append(name)
    
    print(f"Found {len(model_info)} models:")
    print(f"  Rulesets: {list(rulesets.keys())}")
    print(f"  Unique seeds: {len(seeds)} ({list(seeds.keys())[:10]}...)")
    print(f"  Models per ruleset: {[(k, len(v)) for k, v in rulesets.items()]}")
    
    return model_info

def extract_task_timing(trial_batch, task_name, hp):
    """Extract timing information based on Yang et al. paper specifications"""
    
    # Get trial structure
    inputs = trial_batch.x   # Shape: (time, trials, input_dim)
    targets = trial_batch.y  # Shape: (time, trials, output_dim)
    
    timing_info = {}
    
    try:
        n_time, n_trials, n_input = inputs.shape
        n_output = targets.shape[2]
        
        # Use first trial for timing analysis
        trial_inputs = inputs[:, 0, :]   # (time, input_dim)
        trial_targets = targets[:, 0, :] # (time, output_dim)
        
        # 1. FIXATION TIMING
        # Fixation input is usually the first input dimension
        fixation_input = trial_inputs[:, 0]
        
        # Find when fixation input is ON (high)
        fixation_on_indices = np.where(fixation_input > 0.5)[0]
        if len(fixation_on_indices) > 0:
            timing_info['fixation_start'] = int(fixation_on_indices[0])
            # Find when fixation input goes OFF (indicates response/go epoch)
            fixation_off_indices = np.where(fixation_input <= 0.5)[0]
            if len(fixation_off_indices) > 0:
                # Find first OFF after the ON period
                off_after_on = fixation_off_indices[fixation_off_indices > timing_info['fixation_start']]
                if len(off_after_on) > 0:
                    timing_info['fixation_end'] = int(off_after_on[0] - 1)
                    timing_info['go_cue'] = int(off_after_on[0])  # Go/response epoch start
                else:
                    timing_info['fixation_end'] = int(fixation_on_indices[-1])
            else:
                timing_info['fixation_end'] = int(fixation_on_indices[-1])
        
        # 2. RESPONSE TIMING based on target values
        # Look for fixation output target changes (0.85 -> 0.05)
        if n_output > 0:
            # Assume first output is fixation (common in Yang models)
            fixation_target = trial_targets[:, 0]
            
            # High fixation target (~0.85) vs low fixation target (~0.05)
            high_fix_target = np.where(fixation_target > 0.4)[0]  # Above midpoint
            low_fix_target = np.where(fixation_target < 0.4)[0]   # Below midpoint
            
            if len(high_fix_target) > 0 and len(low_fix_target) > 0:
                # Response epoch starts when fixation target drops
                response_start_candidates = low_fix_target[low_fix_target > high_fix_target[0]]
                if len(response_start_candidates) > 0:
                    timing_info['response_start'] = int(response_start_candidates[0])
                    timing_info['response_end'] = n_time - 1
            
            # Look for directional response targets (ring outputs with ~0.8 peak)
            if n_output > 1:  # Has ring outputs
                ring_outputs = trial_targets[:, 1:]  # Exclude fixation output
                max_ring_activity = np.max(ring_outputs, axis=1)
                
                # Find when ring outputs are active (target > 0.3, well above baseline 0.05)
                active_response = np.where(max_ring_activity > 0.3)[0]
                if len(active_response) > 0:
                    if 'response_start' not in timing_info:
                        timing_info['response_start'] = int(active_response[0])
                    timing_info['response_end'] = int(active_response[-1])
        
        # 3. STIMULUS TIMING
        # Look for stimulus inputs (typically after fixation input)
        if n_input > 1:
            # Common structure: [fixation, stim_mod1_ring1, stim_mod1_ring2, stim_mod2_ring1, stim_mod2_ring2, ...]
            stimulus_inputs = trial_inputs[:, 1:min(n_input, 65)]  # Exclude rule inputs (start at 65)
            
            # Find when any stimulus is active
            stimulus_activity = np.sum(np.abs(stimulus_inputs), axis=1)
            stim_active = np.where(stimulus_activity > 0.01)[0]
            
            if len(stim_active) > 0:
                timing_info['stimulus_start'] = int(stim_active[0])
                timing_info['stimulus_end'] = int(stim_active[-1])
        
        # 4. RULE TIMING (from hyperparameters and inputs)
        if 'rule_start' in hp:
            timing_info['rule_start'] = hp['rule_start']
            
            # Rule inputs should be active from rule_start onwards
            if hp['rule_start'] < n_input:
                rule_inputs = trial_inputs[:, hp['rule_start']:]
                rule_activity = np.sum(rule_inputs, axis=1)
                rule_active = np.where(rule_activity > 0.5)[0]
                if len(rule_active) > 0:
                    timing_info['rule_active_start'] = int(rule_active[0])
                    timing_info['rule_active_end'] = int(rule_active[-1])
        
        # Total trial length
        timing_info['trial_length'] = n_time
        
        # Create phase descriptions
        phases = []
        
        # Fixation phase
        if 'fixation_start' in timing_info and 'fixation_end' in timing_info:
            phases.append(f"Fixation: {timing_info['fixation_start']}-{timing_info['fixation_end']}")
        
        # Stimulus phase
        if 'stimulus_start' in timing_info and 'stimulus_end' in timing_info:
            phases.append(f"Stimulus: {timing_info['stimulus_start']}-{timing_info['stimulus_end']}")
        
        # Response/Go phase
        if 'response_start' in timing_info:
            end_time = timing_info.get('response_end', n_time-1)
            phases.append(f"Response: {timing_info['response_start']}-{end_time}")
        elif 'go_cue' in timing_info:
            phases.append(f"Go/Response: {timing_info['go_cue']}-{n_time-1}")
        
        # Rule phase (if different from trial start)
        if 'rule_start' in timing_info and timing_info['rule_start'] > 0:
            phases.append(f"Rule: {timing_info['rule_start']}-{n_time-1}")
        
        timing_info['phases'] = phases
        
        # Add task-specific interpretation
        timing_info['interpretation'] = {
            'fixation_on_target': 0.85,    # Target fixation during fixation epoch
            'fixation_off_target': 0.05,   # Target fixation during response epoch  
            'response_target_peak': 0.8,   # Peak target for directional response
            'baseline_target': 0.05        # Baseline target when no response required
        }
        
    except Exception as e:
        timing_info['error'] = str(e)
    
    return timing_info

def generate_all_task_activity(model_dir, n_trials=5, extract_timing=True):
    """Generate neural activity for all tasks in a model"""
    
    print(f"Generating neural activity from: {model_dir}")
    print("="*60)
    
    try:
        # Import required modules
        import tensorflow as tf
        import network
        import task
        
        # Load hyperparameters
        hp_file = os.path.join(model_dir, 'hp.json')
        with open(hp_file, 'r') as f:
            hp = json.load(f)
        
        # Fix missing RNG in hyperparameters (common issue with pretrained models)
        if 'rng' not in hp:
            hp['rng'] = np.random.RandomState(0)  # Fixed seed for reproducibility
            print("  ✓ Added missing RNG to hyperparameters")
        
        print(f"Model info:")
        print(f"  Tasks: {len(hp['rules'])}")
        print(f"  RNN units: {hp['n_rnn']}")
        print(f"  Generating {n_trials} trials per task")
        
        # Configure TensorFlow
        config = tf.ConfigProto(
            device_count={'GPU': 0},
            allow_soft_placement=True,
            log_device_placement=False
        )
        
        # Create model FIRST (it will reset the graph itself)
        model = network.Model(model_dir, hp=hp)
        
        # Then create session
        with tf.Session(config=config) as sess:
            
            # Try to restore using different methods
            restored = False
            
            # Method 1: Use model's restore if it exists
            if hasattr(model, 'restore'):
                try:
                    model.restore()
                    restored = True
                    print("✓ Restored using model.restore()")
                except:
                    pass
            
            # Method 2: Standard TF restore
            if not restored:
                try:
                    # Initialize variables first
                    sess.run(tf.global_variables_initializer())
                    
                    # Create saver and restore
                    saver = tf.train.Saver()
                    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                    saver.restore(sess, checkpoint_path)
                    restored = True
                    print("✓ Restored using tf.train.Saver")
                except Exception as e:
                    print(f"❌ Standard restore failed: {e}")
            
            if not restored:
                print("❌ Could not restore model weights")
                return None
            
            # Generate activity for each task
            all_activity = {}
            timing_data = {}
            
            for i, rule_name in enumerate(hp['rules']):
                print(f"\n[{i+1}/{len(hp['rules'])}] {rule_name}...", end="")
                
                try:
                    # Generate trials (RNG is now in hp)
                    trial_batch = task.generate_trials(
                        rule_name, hp, mode='test', 
                        noise_on=False, batch_size=n_trials
                    )
                    
                    # Extract timing information
                    if extract_timing:
                        timing_info = extract_task_timing(trial_batch, rule_name, hp)
                        timing_data[rule_name] = timing_info
                    
                    # Run network
                    feed_dict = {model.x: trial_batch.x, model.y: trial_batch.y}
                    h_activity = sess.run(model.h, feed_dict=feed_dict)
                    
                    # Store results
                    all_activity[rule_name] = {
                        'activity': h_activity,  # Shape: (time, trials, units)
                        'inputs': trial_batch.x,
                        'targets': trial_batch.y,
                        'shape': h_activity.shape,
                        'timing': timing_data.get(rule_name, {}) if extract_timing else {}
                    }
                    
                    # Show timing info
                    timing_str = ""
                    if extract_timing and rule_name in timing_data:
                        phases = timing_data[rule_name].get('phases', [])
                        if phases:
                            timing_str = f" | {phases[0]}"
                    
                    print(f" ✓ {h_activity.shape}{timing_str}")
                    
                except Exception as e:
                    print(f" ❌ Failed: {e}")
                    continue
            
            print(f"\n🎉 Successfully generated activity for {len(all_activity)}/{len(hp['rules'])} tasks")
            
            # Print timing summary
            if extract_timing and timing_data:
                print(f"\n⏱️ Timing Summary:")
                for task_name, timing in timing_data.items():
                    if 'phases' in timing and timing['phases']:
                        print(f"  {task_name}: {' | '.join(timing['phases'])}")
            
            return all_activity, hp, timing_data
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def plot_example_time_series(all_activity, hp, model_name, output_dir=".", show_plots=True):
    """Plot example time series with timing annotations"""
    
    if not all_activity:
        print("No activity data to plot")
        return
    
    print(f"\nPlotting example time series...")
    
    # Select first task for example
    task_name = list(all_activity.keys())[0]
    task_data = all_activity[task_name]
    activity_data = task_data['activity']  # (time, trials, units)
    timing_info = task_data.get('timing', {})
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Neural Activity Examples - {model_name}\nTask: {task_name}', fontsize=14)
    
    # Helper function to add timing markers with Yang et al. specifications
    def add_timing_markers(ax):
        if timing_info:
            # Fixation epoch (fixation input ON, target fixation = 0.85)
            if 'fixation_start' in timing_info and 'fixation_end' in timing_info:
                ax.axvspan(timing_info['fixation_start'], timing_info['fixation_end'], 
                          alpha=0.2, color='blue', label='Fixation (0.85)')
            
            # Stimulus epoch  
            if 'stimulus_start' in timing_info and 'stimulus_end' in timing_info:
                ax.axvspan(timing_info['stimulus_start'], timing_info['stimulus_end'], 
                          alpha=0.2, color='green', label='Stimulus')
            
            # Response/Go epoch (fixation input OFF, target fixation = 0.05)
            if 'response_start' in timing_info:
                response_end = timing_info.get('response_end', activity_data.shape[0]-1)
                ax.axvspan(timing_info['response_start'], response_end, 
                          alpha=0.2, color='red', label='Response (0.05)')
            elif 'go_cue' in timing_info:
                ax.axvspan(timing_info['go_cue'], activity_data.shape[0]-1, 
                          alpha=0.2, color='red', label='Go/Response')
            
            # Rule onset
            if 'rule_start' in timing_info and timing_info['rule_start'] > 0:
                ax.axvline(timing_info['rule_start'], color='orange', linestyle='--', 
                          alpha=0.7, label=f"Rule Start ({timing_info['rule_start']})")
    
    # Plot 1: Single unit across time (multiple trials)
    ax1 = axes[0, 0]
    unit_idx = 0
    for trial in range(min(3, activity_data.shape[1])):  # Show up to 3 trials
        ax1.plot(activity_data[:, trial, unit_idx], alpha=0.7, label=f'Trial {trial+1}')
    add_timing_markers(ax1)
    ax1.set_title(f'Unit {unit_idx} Activity Across Trials')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Activation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Multiple units, single trial
    ax2 = axes[0, 1]
    trial_idx = 0
    n_units_to_show = min(10, activity_data.shape[2])
    for unit in range(n_units_to_show):
        ax2.plot(activity_data[:, trial_idx, unit], alpha=0.6, label=f'Unit {unit}')
    add_timing_markers(ax2)
    ax2.set_title(f'Multiple Units - Trial {trial_idx}')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Activation')
    if n_units_to_show <= 5:
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Activity heatmap (time x units) for one trial
    ax3 = axes[1, 0]
    im = ax3.imshow(activity_data[:, 0, :].T, aspect='auto', cmap='viridis')
    ax3.set_title(f'Activity Heatmap - Trial 0')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Unit Index')
    plt.colorbar(im, ax=ax3, label='Activation')
    
    # Plot 4: Population average over time
    ax4 = axes[1, 1]
    # Average across trials and units
    pop_avg_time = np.mean(activity_data, axis=(1, 2))
    pop_std_time = np.std(activity_data, axis=(1, 2))
    time_steps = np.arange(len(pop_avg_time))
    
    ax4.plot(time_steps, pop_avg_time, 'b-', linewidth=2, label='Population Mean')
    ax4.fill_between(time_steps, pop_avg_time - pop_std_time, 
                     pop_avg_time + pop_std_time, alpha=0.3, label='±1 STD')
    ax4.set_title('Population Activity Over Time')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Mean Activation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f"{model_name}_neural_activity_examples.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved example plots: {plot_file}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Print detailed timing statistics with Yang et al. specifications
    print(f"\n📊 Activity & Timing Statistics for {task_name}:")
    print(f"  Shape: {activity_data.shape} (time, trials, units)")
    print(f"  Mean activation: {np.mean(activity_data):.4f}")
    print(f"  Std activation: {np.std(activity_data):.4f}")
    print(f"  Min activation: {np.min(activity_data):.4f}")
    print(f"  Max activation: {np.max(activity_data):.4f}")
    
    if timing_info:
        print(f"  Task timing phases:")
        for phase in timing_info.get('phases', []):
            print(f"    {phase}")
        
        # Show Yang et al. specifications being used
        interp = timing_info.get('interpretation', {})
        if interp:
            print(f"  Yang et al. target specifications:")
            print(f"    Fixation ON target: {interp.get('fixation_on_target', 0.85)}")
            print(f"    Fixation OFF target: {interp.get('fixation_off_target', 0.05)}")
            print(f"    Response peak target: {interp.get('response_target_peak', 0.8)}")
            print(f"    Baseline target: {interp.get('baseline_target', 0.05)}")
        
        # Additional timing details
        if 'go_cue' in timing_info:
            print(f"    Go cue (fixation OFF): timestep {timing_info['go_cue']}")
        if 'rule_start' in timing_info:
            print(f"    Rule input start: timestep {timing_info['rule_start']}")

def save_activity_data(all_activity, hp, model_name, output_dir=".", save_formats=['mat', 'npz', 'pkl']):
    """Save neural activity with detailed Yang et al. timing information"""
    
    if not all_activity:
        print("No activity data to save")
        return
    
    print(f"\nSaving neural activity data...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as .mat file (MATLAB format)
    if 'mat' in save_formats:
        mat_file = os.path.join(output_dir, f"{model_name}_neural_activity.mat")
        mat_data = {}
        
        # Prepare data for .mat format
        for task_name, data in all_activity.items():
            # Clean task name for MATLAB variable naming
            clean_task = task_name.replace('-', '_').replace(' ', '_')
            mat_data[f"{clean_task}_activity"] = data['activity']
            mat_data[f"{clean_task}_inputs"] = data['inputs'] 
            mat_data[f"{clean_task}_targets"] = data['targets']
            
            # Add detailed timing information from Yang et al. specifications
            if 'timing' in data and data['timing']:
                timing = data['timing']
                
                # Core timing events
                for key in ['fixation_start', 'fixation_end', 'stimulus_start', 'stimulus_end', 
                           'response_start', 'response_end', 'go_cue', 'rule_start', 'trial_length']:
                    if key in timing and isinstance(timing[key], (int, float)):
                        mat_data[f"{clean_task}_{key}"] = timing[key]
                
                # Yang et al. target specifications  
                if 'interpretation' in timing:
                    interp = timing['interpretation']
                    mat_data[f"{clean_task}_fixation_on_target"] = interp.get('fixation_on_target', 0.85)
                    mat_data[f"{clean_task}_fixation_off_target"] = interp.get('fixation_off_target', 0.05)
                    mat_data[f"{clean_task}_response_target_peak"] = interp.get('response_target_peak', 0.8)
                    mat_data[f"{clean_task}_baseline_target"] = interp.get('baseline_target', 0.05)
        
        # Add metadata
        mat_data['task_names'] = np.array(list(all_activity.keys()), dtype='U50')
        mat_data['n_units'] = hp['n_rnn']
        mat_data['n_tasks'] = len(all_activity)
        mat_data['model_name'] = model_name
        
        # Add shape information
        shapes = {}
        for task_name, data in all_activity.items():
            clean_task = task_name.replace('-', '_').replace(' ', '_')
            shapes[f"{clean_task}_shape"] = np.array(data['shape'])
        mat_data.update(shapes)
        
        savemat(mat_file, mat_data, do_compression=True)
        print(f"✓ Saved MATLAB: {mat_file}")
    
    # Save as pickle (Python format)
    if 'pkl' in save_formats:
        pickle_file = os.path.join(output_dir, f"{model_name}_neural_activity.pkl")
        with open(pickle_file, 'wb') as f:
            save_data = {
                'activity': all_activity,
                'hyperparams': hp,
                'model_name': model_name
            }
            pickle.dump(save_data, f)
        print(f"✓ Saved pickle: {pickle_file}")
    
    # Save as numpy arrays
    if 'npz' in save_formats:
        np_file = os.path.join(output_dir, f"{model_name}_neural_activity.npz")
        np_data = {}
        
        for task_name, data in all_activity.items():
            np_data[f"{task_name}_activity"] = data['activity']
            np_data[f"{task_name}_inputs"] = data['inputs'] 
            np_data[f"{task_name}_targets"] = data['targets']
        
        # Add metadata
        np_data['task_names'] = np.array(list(all_activity.keys()), dtype='U20')
        np_data['n_units'] = hp['n_rnn']
        np_data['n_tasks'] = len(all_activity)
        
        np.savez_compressed(np_file, **np_data)
        print(f"✓ Saved numpy: {np_file}")
    
    # Save summary info as text
    summary_file = os.path.join(output_dir, f"{model_name}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Yang Multitask RNN Neural Activity\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"RNN Units: {hp['n_rnn']}\n")
        f.write(f"Tasks: {len(all_activity)}\n\n")
        
        f.write("Activity shapes (time, trials, units):\n")
        for task_name, data in all_activity.items():
            shape = data['shape']
            f.write(f"  {task_name}: {shape} ({shape[0]} timesteps, {shape[1]} trials)\n")
    
    print(f"✓ Saved summary: {summary_file}")
    print(f"\n📁 All files saved to: {output_dir}")

def batch_process_models(model_base_dir, n_trials=5, save_formats=['mat'], output_base_dir="."):
    """Process multiple models in batch"""
    
    print("Batch Processing Multiple Models")
    print("="*60)
    
    # Find all model directories
    model_dirs = []
    if os.path.exists(model_base_dir):
        for item in os.listdir(model_base_dir):
            model_path = os.path.join(model_base_dir, item)
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'hp.json')):
                model_dirs.append(model_path)
    
    if not model_dirs:
        print(f"No valid model directories found in {model_base_dir}")
        return
    
    print(f"Found {len(model_dirs)} model directories")
    
    # Analyze models first
    model_info = analyze_model_directory(model_dirs)
    
    # Process each model
    for i, model_dir in enumerate(model_dirs):  # Process first 5 for demo
        model_name = os.path.basename(model_dir)
        print(f"\n[{i+1}/{max(5, len(model_dirs))}] Processing {model_name}...")
        
        result = generate_all_task_activity(model_dir, n_trials, extract_timing=True)
        
        if result[0] is None:
            print(f"❌ Failed to process {model_name}")
            continue
        
        all_activity, hp, timing_data = result
        
        # Create output directory for this model
        model_output_dir = os.path.join(output_base_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Save results
        save_activity_data(all_activity, hp, model_name, model_output_dir, save_formats)
        print(f"✓ Completed {model_name}")

def main():
    print("Yang Neural Activity Generator with Timing & Batch Processing")
    print("="*70)
    
    # Choose processing mode
    mode = input("Process single model (s) or batch process (b)? [s/b]: ").strip().lower()
    
    if mode == 'b':
        # Batch processing mode
        model_base_dir = input("Enter directory containing model folders: ").strip().strip('"\'')
        
        if not os.path.exists(model_base_dir):
            print(f"❌ Directory doesn't exist")
            return
        
        try:
            n_trials = int(input("Number of trials per task (default 5): ").strip() or "5")
        except:
            n_trials = 5
        
        print("\nAvailable formats: mat (MATLAB), npz (NumPy), pkl (Pickle)")
        formats_input = input("Formats to save (default: mat): ").strip() or "mat"
        save_formats = [f.strip() for f in formats_input.split(',')]
        
        output_base_dir = input("Output base directory (default: ./batch_output): ").strip() or "./batch_output"
        
        batch_process_models(model_base_dir, n_trials, save_formats, output_base_dir)
        return
    
    # Single model processing mode
    model_dir = input("Enter model directory: ").strip().strip('"\'')
    
    if not os.path.exists(model_dir):
        print(f"❌ Directory doesn't exist")
        return
    
    # Get model name from directory
    model_name = os.path.basename(model_dir)
    if not model_name:
        model_name = "unknown_model"
    
    # Get number of trials
    try:
        n_trials = int(input("Number of trials per task (default 5): ").strip() or "5")
    except:
        n_trials = 5
    
    # Generate activity
    result = generate_all_task_activity(model_dir, n_trials, extract_timing=True)
    
    if result[0] is None:
        print("❌ Failed to generate activity")
        return
    
    all_activity, hp, timing_data = result
    
    # Plot examples first
    plot_choice = input("\nPlot example time series? (y/n): ").strip().lower()
    if plot_choice == 'y':
        show_plots = input("Show plots interactively? (y/n): ").strip().lower() == 'y'
        output_dir = input("Output directory for plots (default: current): ").strip() or "."
        plot_example_time_series(all_activity, hp, model_name, output_dir, show_plots)
    
    # Save results
    save_choice = input("\nSave results? (y/n): ").strip().lower()
    if save_choice == 'y':
        output_dir = input("Output directory (default: current): ").strip() or "."
        
        # Ask about formats
        print("\nAvailable formats: mat (MATLAB), npz (NumPy), pkl (Pickle)")
        formats_input = input("Formats to save (default: mat,npz): ").strip() or "mat,npz"
        save_formats = [f.strip() for f in formats_input.split(',')]
        
        save_activity_data(all_activity, hp, model_name, output_dir, save_formats)
    
    # Quick analysis
    print(f"\n📊 Quick Analysis:")
    total_timesteps = sum(data['shape'][0] for data in all_activity.values())
    total_trials = sum(data['shape'][1] for data in all_activity.values()) 
    print(f"  Total timesteps across all tasks: {total_timesteps:,}")
    print(f"  Total trials: {total_trials}")
    print(f"  Units: {hp['n_rnn']}")
    print(f"  Total data points: {total_timesteps * hp['n_rnn']:,}")
    

if __name__ == "__main__":
    main()