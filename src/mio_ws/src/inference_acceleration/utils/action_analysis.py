import os
import numpy as np
import matplotlib.pyplot as plt

def action_analysis(gt_actions_list, pred_actions_list, save_path):
    gt_actions = np.array(gt_actions_list)
    pred_actions = np.array(pred_actions_list)

    mean_error = np.mean(np.abs(gt_actions - pred_actions))
    print(f"Action Mean Error: {mean_error}")
    
    seq_len, action_dim = gt_actions.shape
    
    fig, axes = plt.subplots(action_dim, 1, figsize=(10, 2.5 * action_dim), sharex=True)
    
    if action_dim == 1:
        axes = [axes]
        
    for i in range(action_dim):
        axes[i].plot(gt_actions[:, i], label='Ground Truth', linestyle='--', color='blue', alpha=0.8)
        axes[i].plot(pred_actions[:, i], label='Predicted', linestyle='-', color='red', alpha=0.8)
        axes[i].set_ylabel(f'Dim {i}')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, linestyle=':', alpha=0.6)
        
    axes[-1].set_xlabel('Frame Step')
    fig.suptitle(f'Action Dimension Comparison, Mean Error: {mean_error:.4f}', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95) 
    
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, 'actions_error.png')
    plt.savefig(save_path, dpi=300)

