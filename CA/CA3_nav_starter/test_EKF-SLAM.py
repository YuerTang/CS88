#!/usr/bin/env python3
"""
Demonstration of uncertainty visualization in EKF-SLAM.
Shows how uncertainty evolves over time as landmarks are observed.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from environment import Environment, RobotSimulator
from slam import EKFSLAM, get_covariance_ellipse


def visualize_uncertainty_evolution(ax, env, ekf, true_path, current_pose, step, total_steps):
    """Visualize SLAM with focus on uncertainty."""
    ax.clear()
    ax.set_title(f"EKF-SLAM Uncertainty Evolution - Step {step}/{total_steps}", 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(env.x_min, env.x_max)
    ax.set_ylim(env.y_min, env.y_max)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Draw true obstacles (very faded)
    for lm_id, lm_pos in env.landmarks.items():
        circle = Circle((lm_pos[0], lm_pos[1]), env.obstacle_radius, 
                       color='gray', alpha=0.1)
        ax.add_patch(circle)
        ax.plot(lm_pos[0], lm_pos[1], 'k+', markersize=8, alpha=0.3)
    
    # Get estimated landmarks and their covariances
    est_landmarks = ekf.get_estimated_landmarks()
    landmark_covs = ekf.get_landmark_covariances()
    
    # Visualize landmarks with uncertainty
    max_uncertainty = 0
    min_uncertainty = float('inf')
    
    for lm_id, lm_pos in est_landmarks.items():
        if lm_id not in landmark_covs:
            continue
            
        cov = landmark_covs[lm_id]
        
        # Calculate uncertainty magnitude (trace of covariance)
        uncertainty = np.trace(cov)
        max_uncertainty = max(max_uncertainty, uncertainty)
        min_uncertainty = min(min_uncertainty, uncertainty)
        
        # Color code by uncertainty (blue = low, red = high)
        if max_uncertainty > min_uncertainty:
            normalized_unc = (uncertainty - min_uncertainty) / (max_uncertainty - min_uncertainty)
        else:
            normalized_unc = 0
        color = plt.cm.RdYlBu_r(normalized_unc)
        
        # Draw landmark estimate
        ax.plot(lm_pos[0], lm_pos[1], 'o', color=color, markersize=10, 
               markeredgecolor='black', markeredgewidth=1, zorder=5)
        
        # Draw 1-sigma ellipse (68% confidence)
        width_1, height_1, angle = get_covariance_ellipse(cov, n_std=1.0)
        ellipse_1 = Ellipse((lm_pos[0], lm_pos[1]), width_1, height_1, angle=angle,
                           facecolor=color, edgecolor='black', linewidth=1,
                           alpha=0.3, zorder=4)
        ax.add_patch(ellipse_1)
        
        # Draw 2-sigma ellipse (95% confidence)
        width_2, height_2, angle = get_covariance_ellipse(cov, n_std=2.0)
        ellipse_2 = Ellipse((lm_pos[0], lm_pos[1]), width_2, height_2, angle=angle,
                           facecolor='none', edgecolor=color, linewidth=2,
                           linestyle='--', alpha=0.7, zorder=4)
        ax.add_patch(ellipse_2)
        
        # Label
        ax.text(lm_pos[0] + 0.4, lm_pos[1] + 0.4, f"L{lm_id}", 
               fontsize=7, color='black', fontweight='bold')
    
    # Draw paths (faded)
    if len(true_path) > 1:
        ax.plot(true_path[:, 0], true_path[:, 1], 'k-', 
               linewidth=1, alpha=0.2, label='True path')
    
    est_path = ekf.get_estimated_path()
    if len(est_path) > 1:
        ax.plot(est_path[:, 0], est_path[:, 1], 'b--', 
               linewidth=1, alpha=0.3, label='Est. path')
    
    # Draw current robot pose (true)
    x_r, y_r, th_r = current_pose
    ax.plot(x_r, y_r, 'ko', markersize=10, label='True robot', zorder=6)
    ax.arrow(x_r, y_r, 0.8 * np.cos(th_r), 0.8 * np.sin(th_r),
             head_width=0.3, length_includes_head=True, color='black', 
             linewidth=2, zorder=6)
    
    # Draw current robot pose (estimated) with uncertainty
    x_e, y_e, th_e = ekf.get_current_pose()
    ax.plot(x_e, y_e, 'bo', markersize=10, label='Est. robot', zorder=6)
    ax.arrow(x_e, y_e, 0.8 * np.cos(th_e), 0.8 * np.sin(th_e),
             head_width=0.3, length_includes_head=True, color='blue', 
             linewidth=2, zorder=6)
    
    # Draw robot pose uncertainty ellipses
    pose_cov = ekf.get_pose_covariance()
    
    # 1-sigma
    width_1, height_1, angle = get_covariance_ellipse(pose_cov, n_std=1.0)
    pose_ellipse_1 = Ellipse((x_e, y_e), width_1, height_1, angle=angle,
                            facecolor='blue', edgecolor='darkblue', linewidth=1,
                            alpha=0.3, zorder=5, label='Robot 1σ')
    ax.add_patch(pose_ellipse_1)
    
    # 2-sigma
    width_2, height_2, angle = get_covariance_ellipse(pose_cov, n_std=2.0)
    pose_ellipse_2 = Ellipse((x_e, y_e), width_2, height_2, angle=angle,
                            facecolor='none', edgecolor='blue', linewidth=2,
                            linestyle='--', alpha=0.7, zorder=5, label='Robot 2σ')
    ax.add_patch(pose_ellipse_2)
    
    # Info box with statistics
    pose_error = np.linalg.norm([x_r - x_e, y_r - y_e])
    pose_uncertainty = np.sqrt(np.trace(pose_cov))
    
    if len(est_landmarks) > 0:
        avg_landmark_unc = np.mean([np.trace(landmark_covs[lm_id]) 
                                    for lm_id in landmark_covs])
        max_landmark_unc = np.max([np.trace(landmark_covs[lm_id]) 
                                   for lm_id in landmark_covs]) if landmark_covs else 0
    else:
        avg_landmark_unc = 0
        max_landmark_unc = 0
    
    info_text = (f"Landmarks: {len(est_landmarks)}/{len(env.landmarks)}\n"
                f"Pose error: {pose_error:.3f}m\n"
                f"Pose unc: {pose_uncertainty:.3f}m\n"
                f"Avg LM unc: {avg_landmark_unc:.3f}\n"
                f"Max LM unc: {max_landmark_unc:.3f}")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    


def main():
    """Run uncertainty visualization demo."""
    print("\n" + "="*60)
    print("EKF-SLAM UNCERTAINTY VISUALIZATION")
    print("="*60)
    print("\nThis demo focuses on uncertainty estimation in SLAM.")
    print("\nVisualization elements:")
    print("  • Ellipses show uncertainty regions (1σ and 2σ)")
    print("  • Colors indicate uncertainty level:")
    print("    - Blue = low uncertainty (well-observed)")
    print("    - Red = high uncertainty (fewer observations)")
    print("  • Ellipse size shows position uncertainty")
    print("  • Ellipse orientation shows correlation")
    print("\nWatch how:")
    print("  • Uncertainty decreases with repeated observations")
    print("  • New landmarks start with high uncertainty")
    print("  • Robot pose uncertainty grows between observations")
    print("="*60 + "\n")
    
    # Setup
    env = Environment(seed=42)
    
    # Higher noise for more visible uncertainty
    R = np.diag([0.1, 0.1, np.deg2rad(3.0)]) ** 2  # Process noise
    Q = np.diag([0.5, np.deg2rad(8.0)]) ** 2  # Measurement noise
    
    init_pose = [0.0, 0.0, 0.0]
    ekf = EKFSLAM(init_pose=init_pose, R=R, Q=Q)
    
    simulator = RobotSimulator(
        environment=env,
        init_pose=init_pose,
        process_noise=R,
        measurement_noise=Q
    )
    
    # Visualization setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 11))
    
    num_steps = 360
    update_interval = 5  # Update every N steps
    
    print("Starting exploration with uncertainty tracking...")
    print("(Higher noise settings for clearer visualization)\n")
    
    try:
        for t in range(num_steps):
            # Control pattern
            if t < num_steps//4:
                v, omega = 1.0, 0
            elif t < 2*num_steps//4:
                v, omega = 1.0, 0.2
            elif t < 2*num_steps//4:
                v, omega = 1.0, 0
            else:
                v, omega = 1.0, 0.1
            
            control = (v, omega)
            
            # Simulate and update SLAM
            measurements = simulator.step(control)
            ekf.step(control, measurements, dt=simulator.dt)
            
            # Update visualization
            if t % update_interval == 0:
                visualize_uncertainty_evolution(
                    ax, env, ekf,
                    simulator.get_true_path(),
                    simulator.true_pose,
                    t, num_steps
                )
                plt.pause(0.01)
            
            # Progress updates
            if (t + 1) % 50 == 0:
                n_landmarks = len(ekf.landmark_indices)
                avg_trace = np.mean([np.trace(cov) for cov in 
                                    ekf.get_landmark_covariances().values()]) if n_landmarks > 0 else 0
                print(f"  Step {t+1}/{num_steps} - "
                      f"Landmarks: {n_landmarks}, Avg uncertainty: {avg_trace:.4f}")
        
        print(f"\n✓ Exploration complete!")
        print(f"  Final landmarks: {len(ekf.landmark_indices)}/{len(env.landmarks)}")
        
        # Compute final statistics
        landmark_covs = ekf.get_landmark_covariances()
        uncertainties = [np.trace(cov) for cov in landmark_covs.values()]
        
        print(f"\nUncertainty Statistics:")
        print(f"  Average: {np.mean(uncertainties):.4f}")
        print(f"  Std dev: {np.std(uncertainties):.4f}")
        print(f"  Min: {np.min(uncertainties):.4f}")
        print(f"  Max: {np.max(uncertainties):.4f}")
        
        # Keep final visualization open
        plt.ioff()
        print("\nFinal state displayed. Close window to exit.")
        plt.show()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        plt.close()


if __name__ == "__main__":
    main()
