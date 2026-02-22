#!/usr/bin/env python3
"""
h5_visualizer.py - Robust HDF5 visualizer.
Forces TkAgg backend to bypass Qt/XCB conflicts.
"""

import os
# Force non-Qt backend for matplotlib
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

# Disable OpenCV's bundled Qt to prevent crashes on import
os.environ["QT_QPA_PLATFORM"] = "offscreen" 

import argparse
import h5py
import cv2
import numpy as np

# Restore platform for matplotlib's window after cv2 is imported
os.environ.pop("QT_QPA_PLATFORM", None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', required=True, help='Path to HDF5 file')
    args = parser.parse_args()

    if not os.path.exists(args.h5):
        print(f"Error: File {args.h5} not found.")
        return

    f = h5py.File(args.h5, 'r')
    voxels = f['voxels']
    actions = f['actions']
    timestamps = f['timestamps_ns']
    rgb_mask = f['rgb_mask']
    rgb_images = f['rgb_images']
    rgb_indices = f['rgb_indices'] if 'rgb_indices' in f else None

    print(f"Total Steps: {len(voxels)}")
    print("Use 'd' or Right-Arrow for Next, 'a' or Left-Arrow for Previous, 'q' to quit.")

    # Precompute step_idx → rgb_image_idx mapping for O(1) lookup
    rgb_lookup = {}
    if rgb_indices is not None:
        ri = rgb_indices[:]
        for img_i, step_i in enumerate(ri):
            rgb_lookup[int(step_i)] = img_i

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2)
    ax_vox = fig.add_subplot(gs[0, 0])
    ax_rgb = fig.add_subplot(gs[0, 1])
    ax_act = fig.add_subplot(gs[1, :])

    state = {'idx': 0}

    def update_plot():
        idx = state['idx']
        fig.suptitle(f"Step {idx} / {len(voxels)-1}\nTS: {timestamps[idx]} ns", fontsize=14)
        
        # 1. Voxel
        ax_vox.clear()
        v = voxels[idx]
        v_img = np.sum(v, axis=0)
        ax_vox.imshow(v_img, cmap='viridis')
        ax_vox.set_title(f"Event Voxel (720x1280)\nSum of {len(v)} bins")
        ax_vox.axis('off')

        # 2. RGB
        ax_rgb.clear()
        if rgb_mask[idx]:
            try:
                img_idx = rgb_lookup.get(idx)
                if img_idx is None:
                    raise KeyError(f"No RGB for step {idx}")
                img = rgb_images[img_idx]
                ax_rgb.imshow(img)
                ax_rgb.set_title(f"RGB Goal Frame ({img.shape[1]}x{img.shape[0]})")
            except Exception as e:
                ax_rgb.text(0.5, 0.5, f"Load Error: {e}", ha='center', color='orange')
        else:
            ax_rgb.text(0.5, 0.5, "NO RGB FRAME\n(This window skipped)", ha='center', va='center', color='red', fontsize=12)
        ax_rgb.axis('off')

        # 3. Action Trajectory
        ax_act.clear()
        act = actions[idx]
        xs = act[:, 0]
        ys = act[:, 1]
        
        # Plot path
        ax_act.plot(xs, ys, 'b-o', linewidth=2, markersize=6, label='Predicted Path')
        # Robot heading at each point (approximation using yaw)
        for i in range(len(act)):
            yaw = act[i, 2]
            ax_act.arrow(xs[i], ys[i], 0.05*np.cos(yaw), 0.05*np.sin(yaw), head_width=0.02, color='green')
            
        ax_act.scatter(0, 0, c='red', marker='X', s=150, label='Robot (Current)')
        ax_act.set_xlabel("Local X (meters forward)")
        ax_act.set_ylabel("Local Y (meters left/right)")
        ax_act.set_title("8-Step Action Chunk (Relative to current pose)")
        ax_act.grid(True, linestyle='--', alpha=0.7)
        ax_act.axis('equal')
        ax_act.legend(loc='upper left')

        plt.draw()

    def on_key(event):
        if event.key in ['right', 'd']:
            state['idx'] = min(state['idx'] + 1, len(voxels) - 1)
            update_plot()
        elif event.key in ['left', 'a']:
            state['idx'] = max(state['idx'] - 1, 0)
            update_plot()
        elif event.key == 'q':
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_plot()
    plt.tight_layout()
    plt.show()
    f.close()

if __name__ == '__main__':
    main()
