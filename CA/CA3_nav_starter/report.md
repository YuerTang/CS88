# CS188 Coding Assignment 3: SLAM and Motion Planning — Report

---

## Part A: EKF-SLAM — Trajectory and Uncertainty

### A1. Path Design: Which Trajectory is Better for SLAM?

**Trajectory Descriptions.**
I designed two qualitatively different control strategies for the robot operating in the same 15-landmark environment.

- **Path A — "Greedy Forward"**: The robot drives in an L-shaped sweep. It moves straight east for 150 steps, turns left ~90 degrees, drives north for 150 steps, turns left again, then drives west. Each landmark is typically observed only once or twice as the robot passes by.

- **Path B — "Looping / Revisit"**: The robot heads northeast, enters a gentle curve, then executes tight orbital loops (angular velocity w=0.25) that cause it to revisit the same region repeatedly. After completing one orbit, it translates and enters a second orbit. Landmarks near the orbits are observed 4–8 times each.

**Quantitative Results (500 steps, medium noise):**

| Metric                     | Greedy Forward | Looping Revisit |
|----------------------------|----------------|-----------------|
| Final pose error           | 0.669 m        | 1.204 m         |
| Mean pose error            | 0.837 m        | 0.775 m         |
| Final pose uncertainty     | 0.572 m        | 0.607 m         |
| Final avg landmark unc.    | 0.215          | 0.217           |
| Landmarks observed         | 15/15          | 15/15           |

**Which path leads to lower pose error and lower map uncertainty?**
The Greedy Forward path achieves a lower *final* pose error (0.669 m vs 1.204 m) and slightly lower final pose uncertainty (0.572 m vs 0.607 m). However, the Looping path achieves a lower *mean* pose error across the full trajectory (0.775 m vs 0.837 m). This is because the looping trajectory frequently re-observes landmarks, which triggers EKF corrections that pull the estimate back toward truth. The greedy path accumulates drift in its long straight segments but ends in a region where nearby landmarks anchor the final estimate well.

Final landmark uncertainties are nearly identical (~0.215 vs ~0.217), because both paths observe all 15 landmarks and the looping path's repeated observations have diminishing returns once covariance is already small.

**How do uncertainty ellipses behave over time?**
Under Path A, landmark ellipses shrink rapidly when first observed but remain at that level afterward, since the robot moves on and never returns. Under Path B, the ellipses for landmarks near the orbit centers continue to shrink over time as the robot repeatedly passes by and collects new measurements. The trajectory comparison figure (a1_trajectories.png) shows tighter ellipses near the orbit center for Path B compared to landmarks far from the path.

**Concrete example of revisit reducing uncertainty.**
In Path B, the robot orbits near landmarks L4 (8,9), L0 (5,10), and L6 (12,12). During the first pass (~step 50–100), these landmarks are initialized with large uncertainty ellipses. On each subsequent orbit (~steps 100–350), the robot re-observes these same landmarks from different angles, and the EKF update step shrinks the covariance. The measurement Jacobian H maps the innovation (difference between predicted and actual range/bearing) into a correction via the Kalman gain K, which is proportional to the current uncertainty. Each re-observation reduces the landmark covariance by incorporating new information, and the effect is clearly visible as the 2-sigma ellipses contract in the metrics plot (a1_metrics.png) where average landmark uncertainty drops steeply in the first orbit and continues to decrease in subsequent orbits.

*See figures: a1_trajectories.png, a1_metrics.png*

### A2. Noise Sensitivity

I used Path B (Looping/Revisit) for the noise study because its repeated observations provide the EKF with the most opportunities to correct, making it the stronger baseline to stress-test.

**Process Noise Sweep (Q fixed at Medium):**

| R Level | sigma_xy | sigma_theta | Final Pose Error |
|---------|----------|-------------|------------------|
| Low     | 0.05     | 1.5 deg     | 0.278 m          |
| Medium  | 0.10     | 3.0 deg     | 1.204 m          |
| High    | 0.30     | 8.0 deg     | 1.857 m          |

**Measurement Noise Sweep (R fixed at Medium):**

| Q Level | sigma_range | sigma_bearing | Final Pose Error |
|---------|-------------|---------------|------------------|
| Low     | 0.2         | 3.0 deg       | 1.263 m          |
| Medium  | 0.5         | 8.0 deg       | 1.204 m          |
| High    | 1.5         | 15.0 deg      | 1.351 m          |

**How does increasing each noise type affect performance?**

*Process noise (R)* has a dramatic effect: going from Low to High increases final pose error by 6.7x (0.278 to 1.857 m). This is because process noise corrupts the robot's true motion at every single timestep. Even though the EKF predict step accounts for this noise in the covariance, the *actual* trajectory drifts more, making it harder for measurement updates to fully correct.

*Measurement noise (Q)* has a much smaller effect: final pose error only varies from 1.204 to 1.351 m across the sweep. Noisier measurements reduce the precision of each individual EKF correction, but because the looping trajectory provides many repeated observations of each landmark, the statistical averaging effect keeps overall accuracy relatively stable.

**Which type of noise is more damaging?**
Process noise is clearly more damaging. A 6x increase in process noise standard deviation causes a 6.7x increase in pose error, while a 7.5x increase in measurement noise standard deviation causes only a 1.1x increase. This makes physical sense: process noise corrupts the state at every prediction step (500 times in our experiment), whereas measurement noise only affects corrections and can be mitigated through repeated observations. In EKF terms, high R inflates P during every predict step, and no amount of update steps can fully compensate if the underlying trajectory has drifted significantly.

**Does the looping trajectory still work under high noise?**
The looping trajectory remains functional even at high noise levels — all 15 landmarks are still observed and the filter does not diverge. However, performance degrades significantly under high process noise (1.857 m final error). The repeated observations prevent catastrophic failure but cannot fully compensate for the large accumulated drift. Under high measurement noise, the trajectory is remarkably robust (only 1.351 m error), confirming that revisiting landmarks provides redundancy against sensor noise.

*See figures: a2_process_noise.png, a2_measurement_noise.png, a2_summary_table.png*

---

## Part B: PRM vs RRT — Planning Comparison

### B1. Start/Goal Scenarios

I designed two scenarios using the same 15 circular landmarks but varying the obstacle radius:

- **Scenario 1 — "Open"**: Default obstacle radius (0.8), start=[1,1], goal=[19,1]. A horizontal traverse with wide gaps between obstacles.
- **Scenario 2 — "Constrained"**: Enlarged obstacle radius (1.0), start=[1,1], goal=[17,17]. A diagonal traverse through tighter passages.

**Prediction:** I expected PRM to perform well in both scenarios because its global roadmap sampling provides good coverage. For the constrained scenario, I expected RRT might occasionally struggle with narrow passages since its incremental tree growth depends on random sampling to discover corridors. I predicted PRM would produce shorter paths (since A* finds optimal paths on the roadmap) while RRT would be faster (since it doesn't need to build a full roadmap).

*See figures: b1_open.png, b1_constrained.png*

### B2. Metrics Comparison (Constrained Scenario, N=30 trials)

| Metric          | PRM     | RRT     |
|-----------------|---------|---------|
| Success rate    | 100%    | 100%    |
| Avg path length | 24.48 m | 28.72 m |
| Avg time        | 1.762 s | 0.068 s |
| Avg nodes       | 500     | 217     |

**Which planner has higher success rate?**
Both achieve 100% success in the constrained scenario with obstacle radius 1.0. The passages are tight enough to be interesting but not so narrow as to cause frequent failures.

**Which produces shorter paths?**
PRM produces significantly shorter paths (24.48 m vs 28.72 m, a 15% difference). This is expected: PRM builds a dense roadmap and uses A* to find the shortest path on it, while RRT finds the *first* feasible path by growing a tree, with no optimality guarantee. RRT paths tend to have unnecessary zigzags inherited from the random tree growth.

**Failure modes?**
At this obstacle radius, neither planner exhibits clear failure modes. However, RRT's path quality is noticeably worse — its paths contain many small detours from the random tree structure. PRM's main weakness is its longer planning time (26x slower), since it must build the entire roadmap before planning.

*See figures: b1_constrained.png, b2_comparison.png*

### B3. Parameter Studies

**PRM — n_samples:**

| n_samples | Success | Avg Length |
|-----------|---------|------------|
| 100       | 100%    | 25.17 m    |
| 200       | 100%    | 24.85 m    |
| 400       | 100%    | 24.57 m    |
| 800       | 100%    | 24.27 m    |

Increasing n_samples monotonically improves path length, but with diminishing returns — going from 100 to 800 samples (8x) only improves path length by 3.6%. Success rate is already 100% at 100 samples for this scenario. More samples increase planning time quadratically (due to nearest-neighbor computation), so the cost-benefit tradeoff suggests 200–400 samples is a sweet spot here.

**RRT — step_size:**

| step_size | Success | Avg Length |
|-----------|---------|------------|
| 0.2       | 100%    | 28.40 m    |
| 0.5       | 100%    | 28.72 m    |
| 1.0       | 100%    | 28.98 m    |

Step size has minimal effect on path quality in this environment. Smaller steps require more iterations to reach the goal but produce slightly shorter paths because the tree has finer granularity and can navigate more precisely around obstacles. Larger steps are faster per iteration but can overshoot narrow gaps. In environments with very tight passages, smaller step sizes would be necessary to avoid collision-check failures.

**RRT — goal_sample_rate:**

| goal_rate | Success | Avg Length |
|-----------|---------|------------|
| 0.0       | 95%     | 29.97 m    |
| 0.1       | 100%    | 28.72 m    |
| 0.3       | 100%    | 28.10 m    |
| 0.7       | 100%    | 26.84 m    |

Goal bias has the strongest effect among RRT parameters. With zero bias (pure random sampling), success drops to 95% because the tree may not reach the goal within 5000 iterations. Higher bias consistently improves both success rate and path length — at 0.7, paths are 10% shorter than at 0.0. This works well here because the environment has relatively open paths toward the goal. In environments where the goal is behind an obstacle wall, high goal bias could hurt by wasting many iterations extending toward the goal through blocked regions.

*See figure: b3_parameter_studies.png*

---

## Part C: Improving PRM and RRT

### Improvement: Path Shortcutting (Post-Processing)

**Description.**
I implemented a simple shortcutting algorithm applied as a post-processing step to paths from both PRM and RRT. The algorithm works as follows: for a fixed number of iterations (200), randomly select two non-adjacent waypoints i and j on the path. If the straight line from waypoint i to waypoint j is collision-free, remove all intermediate waypoints between them. This is repeated until the iteration budget is exhausted or the path has only 2 waypoints.

This improvement addresses a common weakness of both planners: their paths follow the structure of the roadmap/tree rather than the shortest collision-free route. PRM paths zigzag between sampled nodes, and RRT paths inherit the random walk pattern of the tree. Shortcutting removes unnecessary detours by finding direct connections.

**Experiment Design.**
I ran both PRM and RRT 20 times each on the constrained scenario (obstacle radius 1.0, start=[1,1], goal=[17,17]) and applied shortcutting to every successful path. I recorded path length and waypoint count before and after.

**Results:**

| Planner | Success | Len Before | Len After | Len Reduction | WP Before | WP After | WP Reduction |
|---------|---------|------------|-----------|---------------|-----------|----------|--------------|
| PRM     | 20/20   | 24.50 m    | 24.03 m   | 2.0%          | 14.8      | 4.5      | 69.5%        |
| RRT     | 20/20   | 28.72 m    | 24.46 m   | 14.8%         | 59.0      | 5.4      | 90.9%        |

**Analysis.**
The shortcutting improvement dramatically benefits RRT: path length drops by 14.8% (from 28.72 m to 24.46 m), nearly matching PRM's baseline quality, and waypoint count drops by 91% (from 59 to 5.4 on average). This makes sense because RRT paths contain many small, incremental steps from tree growth that create unnecessary detours — exactly the kind of redundancy shortcutting eliminates.

PRM benefits less in path length (only 2.0% reduction) because A* already finds near-optimal paths on the roadmap. However, waypoint count still drops by 69.5% (from 14.8 to 4.5), meaning the path is simplified significantly even though its length barely changes. This is useful for downstream execution — a robot controller needs fewer waypoints to follow.

After shortcutting, both planners produce paths of nearly identical length (~24 m), suggesting that the shortcutted path is close to the true shortest collision-free path for this scenario. The improvement behaved exactly as expected: it is a simple, effective post-processing step that closes the quality gap between RRT and PRM at negligible computational cost.

*See figures: c_before_after.png, c_summary_table.png, c_boxplot.png*

---

## Use of AI Tools

I used Claude Code (an AI coding assistant) to help write the experiment scripts (`experiment_slam.py`, `experiment_planning.py`, `experiment_improve.py`, `path_utils.py`), including the plotting code and metric collection logic. I designed the experimental parameters (trajectory shapes, noise levels, parameter sweep values) and the AI assistant implemented the scripts based on my specifications. I verified that all scripts ran correctly and inspected the output figures to confirm they matched expectations. The analysis, interpretation, and written explanations in this report are my own, informed by the numerical results and figures produced by the experiments.
