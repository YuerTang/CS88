# CA3: SLAM and Motion Planning Experiments

## Prerequisites

```bash
pip install numpy matplotlib
```

## Running the experiments

All scripts use the `Agg` backend so no display is needed. Results are saved as PNG files.

```bash
cd CA/CA3_nav_starter/

# Part A: EKF-SLAM experiments
python experiment_slam.py

# Part B: PRM vs RRT experiments
python experiment_planning.py

# Part C: Path shortcutting improvements
python experiment_improve.py
```

## Output

| Directory       | Contents                                         |
|-----------------|--------------------------------------------------|
| `results/partA/`| Trajectory plots, metric curves, noise tables    |
| `results/partB/`| Scenario visualizations, comparison table, param studies |
| `results/partC/`| Before/after figures, summary table, box plots   |
