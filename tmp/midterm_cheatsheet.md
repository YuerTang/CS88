# CS 188 — Midterm Cheatsheet
---

## 1. DH PARAMETERS

```
  Frame Assignment Rules
  ──────────────────────
  z_i  →  along joint axis (rotation or sliding)
  x_i  →  common normal between z_{i-1} and z_i

  ┌──────────────┬────────────────────────────────────────────┐
  │  Parameter   │  What it measures                         │
  ├──────────────┼────────────────────────────────────────────┤
  │  α_{i-1}     │  ∠ from z_{i-1} to z_i  around x_{i-1}   │
  │  a_{i-1}     │  dist from z_{i-1} to z_i along x_{i-1}   │
  │  d_i         │  dist from x_{i-1} to x_i along z_i       │
  │  θ_i         │  ∠ from x_{i-1} to x_i  around z_i        │
  └──────────────┴────────────────────────────────────────────┘

  Revolute  joint  →  θ_i is VARIABLE,  d_i is fixed
  Prismatic joint  →  d_i is VARIABLE,  θ_i is fixed
```

### DH Transformation Matrix

```
  T_{i-1,i}  =  Rot(x, α) · Trans(x, a) · Trans(z, d) · Rot(z, θ)

  ┌                                              ┐
  │  cosθ      -sinθ       0        a            │
  │  sinθ·cosα  cosθ·cosα  -sinα   -sinα·d       │
  │  sinθ·sinα  cosθ·sinα   cosα    cosα·d       │
  │  0          0           0       1             │
  └                                              ┘

  Full FK:  T_0n = T_01 · T_12 · T_23 · ... · T_{n-1,n}
```

### RPR Example (Practice Midterm Q4)

```
  ┌─────┬─────────┬─────────┬───────┬───────┐
  │  i  │ α_{i-1} │ a_{i-1} │  d_i  │  θ_i  │
  ├─────┼─────────┼─────────┼───────┼───────┤
  │  1  │    0    │    0    │   0   │  θ₁*  │
  │  2  │   90°   │    0    │  d₂*  │   0   │
  │  3  │  -90°   │    0    │   0   │  θ₃*  │
  └─────┴─────────┴─────────┴───────┴───────┘
                              * = variable
```

---

## 2. FORWARD & INVERSE KINEMATICS (2-Link Arm)

```
          (x, y)
           ╱
       l₂╱  θ₂
        ╱───●
  l₁ ╱  θ₁
    ╱
   ●───── base
```

### Forward Kinematics
```
  x = l₁·cos(θ₁) + l₂·cos(θ₁ + θ₂)
  y = l₁·sin(θ₁) + l₂·sin(θ₁ + θ₂)
```

### Inverse Kinematics
```
  Step 1 — Find θ₂ (Law of Cosines):
  ┌─────────────────────────────────────────────────┐
  │  cos(θ₂) = (x² + y² - l₁² - l₂²) / (2·l₁·l₂) │
  └─────────────────────────────────────────────────┘
  → TWO solutions: θ₂ and -θ₂  (elbow up / elbow down)

  Step 2 — Find θ₁:
  ┌──────────────────────────────────────────────────────────────┐
  │  θ₁ = atan2(y, x) - atan2(l₂·sin(θ₂), l₁ + l₂·cos(θ₂))    │
  └──────────────────────────────────────────────────────────────┘
```

### Jacobian (2-Link)
```
       ┌  ∂x/∂θ₁   ∂x/∂θ₂  ┐     ┌  ẋ  ┐       ┌  θ̇₁  ┐
  J =  │                     │     │     │ = J · │      │
       └  ∂y/∂θ₁   ∂y/∂θ₂  ┘     └  ẏ  ┘       └  θ̇₂  ┘

  J = ┌ -l₁sinθ₁ - l₂sin(θ₁+θ₂)    -l₂sin(θ₁+θ₂) ┐
      └  l₁cosθ₁ + l₂cos(θ₁+θ₂)     l₂cos(θ₁+θ₂) ┘

  ★ SINGULARITY when det(J) = 0  →  loses a degree of freedom
    (arm fully extended or fully folded back)
```

---

## 3. CAMERA CALIBRATION & PROJECTION

### The Big Picture
```
  World 3D          Camera 3D           Image 2D
  Coordinates  ───►  Coordinates  ───►  Pixels
     X_w           [R|t]           K
   (4×1)          (3×4)          (3×3)


  ┌─────────────────────────────────────────┐
  │         x  =  K · [R|t] · X            │
  │       (3×1)  (3×3)(3×4) (4×1)          │
  └─────────────────────────────────────────┘
```

### Intrinsic Matrix K
```
       ┌  f_x   0    u₀  ┐
  K =  │   0   f_y   v₀  │
       └   0    0     1   ┘

  f_x, f_y = focal length (in pixels)
  (u₀, v₀) = principal point (image center)
```

### Extrinsic Matrix [R|t]
```
  [R|t] = camera_T_world     ← world TO camera, NOT camera pose!
         (3×4 matrix)

  R = 3×3 rotation     t = 3×1 translation

  ⚠️  COMMON TRAP: [R|t] transforms FROM world TO camera frame
      Camera pose in world = inverse of this!
```

### Pinhole Projection
```
  u = f_x · (X/Z) + u₀
  v = f_y · (Y/Z) + v₀

  ★ Division by Z = why we lose depth info!
```

### Back-Projection (2D → 3D, need depth Z)
```
  X = (u - u₀) / f_x · Z
  Y = (v - v₀) / f_y · Z
  Z = Z  (from depth sensor)
```

### Depth Sensing Methods
```
  Stereo        →  two cameras, depth from disparity
  Time-of-Flight → measures light round-trip time
  Structured Light → projects pattern, measures distortion
```

---

## 4. MOTION PLANNING

### Configuration Space
```
  C-space = all possible robot configurations (e.g., joint angles)
  C_free  = collision-free configs       ┐
  C_obs   = configs causing collision    ┘  C_free ∪ C_obs = C-space
```

### PRM (Probabilistic Roadmap) — MULTI-QUERY
```
  ┌──────────────────────────────────────────────────┐
  │  Phase 1: BUILD ROADMAP (preprocessing)          │
  │  ─────────────────────────────────────           │
  │  1. Sample N random configs in C_free            │
  │     → these are "milestones" (graph nodes)       │
  │  2. For each milestone:                          │
  │     → find neighbors within distance R           │
  │     → check local path for collisions            │
  │     → if clear, add edge (undirected graph)      │
  │                                                  │
  │  Phase 2: QUERY                                  │
  │  ──────────────                                  │
  │  1. Connect q_start & q_goal to roadmap          │
  │  2. Graph search (Dijkstra / A*)                 │
  └──────────────────────────────────────────────────┘

  Properties:
  ✓ Probabilistically complete (N→∞ guarantees finding path)
  ✗ NOT complete (finite samples can miss solutions)
  ✓ Good for MULTIPLE queries in same environment
```

### RRT (Rapidly-exploring Random Tree) — SINGLE-QUERY
```
  ┌──────────────────────────────────────────────────┐
  │  1. Tree rooted at q_start                       │
  │  2. Loop:                                        │
  │     a. Sample random q_rand                      │
  │     b. Find q_near (nearest node in tree)        │
  │     c. EXTEND: move from q_near toward q_rand    │
  │        by step ε → get q_new                     │
  │     d. If path q_near→q_new is collision-free    │
  │        → add q_new to tree                       │
  │  3. Stop when close enough to q_goal             │
  └──────────────────────────────────────────────────┘

       q_rand
         ╳
        ╱
  q_near───→ q_new   (step size ε)
      │
      │ (existing tree)
   q_start
```

### PRM vs RRT at a Glance
```
  ┌──────────────┬──────────────────┬──────────────────┐
  │              │      PRM         │      RRT         │
  ├──────────────┼──────────────────┼──────────────────┤
  │ Structure    │ Undirected graph │ Tree             │
  │ Query type   │ Multi-query      │ Single-query     │
  │ Preprocessing│ Yes (build once) │ No               │
  │ Completeness │ Prob. complete   │ Prob. complete   │
  │ Best for     │ Static env,      │ Dynamic env,     │
  │              │ many queries     │ one-shot query   │
  └──────────────┴──────────────────┴──────────────────┘
```

---

## 5. MDPs (Markov Decision Processes)

### Definition
```
  MDP = ( S,  A,  P,  R,  γ )
         │   │   │   │   └─ discount factor (0 ≤ γ < 1)
         │   │   │   └───── reward R(s, a, s')
         │   │   └───────── transition P(s'|s, a)
         │   └───────────── actions
         └───────────────── states

  ★ Markov Property: future depends ONLY on current state
```

### Value Functions
```
  V^π(s)    = expected total discounted reward from s, following π
  Q^π(s, a) = expected total discounted reward from s, doing a, then π

  Relationship:  V*(s) = max_a Q*(s, a)
```

### Bellman Equations

```
  ┌─────────────────────────────────────────────────────────────┐
  │  FIXED POLICY π:                                            │
  │  V^π(s) = Σ_{s'} P(s'|s, π(s)) [R(s,π(s),s') + γ·V^π(s')]│
  ├─────────────────────────────────────────────────────────────┤
  │  OPTIMAL (Bellman Optimality):                              │
  │  V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ·V*(s')]   │
  │  Q*(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ·max_{a'} Q*(s',a')] │
  └─────────────────────────────────────────────────────────────┘
```

### Value Iteration
```
  ┌──────────────────────────────────────────────────────────┐
  │  Initialize V₀(s) arbitrarily for all s                 │
  │  Repeat until convergence:                               │
  │                                                          │
  │    V_{k+1}(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γV_k(s')]│
  │                  ▲                                       │
  │                  └── pick the BEST action each time      │
  │                                                          │
  │  Extract policy:                                         │
  │    π*(s) = argmax_a Σ P(s'|s,a)[R + γV*(s')]            │
  │                                                          │
  │  Cost per iteration: O(|S|² · |A|)                      │
  └──────────────────────────────────────────────────────────┘
```

### Policy Iteration
```
  ┌──────────────────────────────────────────────────────────┐
  │  Initialize π₀ arbitrarily                              │
  │  Repeat until π stops changing:                          │
  │                                                          │
  │  ① POLICY EVALUATION  (solve for V^π)                   │
  │     V^π(s) = Σ P(s'|s,π(s))[R + γV^π(s')]              │
  │     → solve linear system: O(|S|³)                      │
  │                                                          │
  │  ② POLICY IMPROVEMENT  (update π greedily)              │
  │     π_{new}(s) = argmax_a Σ P(s'|s,a)[R + γV^π(s')]    │
  │     → cost: O(|S|² · |A|)                               │
  └──────────────────────────────────────────────────────────┘
```

### Value Iteration vs Policy Iteration
```
  ┌─────────────────┬────────────────────┬────────────────────┐
  │                 │ Value Iteration    │ Policy Iteration   │
  ├─────────────────┼────────────────────┼────────────────────┤
  │ Updates         │ V only             │ π + V              │
  │ Per-iter cost   │ O(|S|²·|A|)       │ O(|S|³+|S|²·|A|)  │
  │ # Iterations    │ More               │ Fewer              │
  │ Simpler?        │ ✓ Yes              │ More complex       │
  │ In practice     │ ─                  │ Often faster       │
  └─────────────────┴────────────────────┴────────────────────┘
```

### POMDP (extends MDP)
```
  POMDP adds:
    O          = set of observations
    O(o|s',a)  = observation probability

  Agent does NOT know true state
    → maintains BELIEF state b(s) = probability distribution over S
```

---

## 6. CONTROL

```
  ┌─────────────────────────────────────────────┐
  │  Open-loop (feedforward):                   │
  │    Input ──► Controller ──► Plant ──► Output │
  │    (no feedback, can drift)                 │
  │                                             │
  │  Closed-loop (feedback):                    │
  │    Input ──► Controller ──► Plant ──► Output │
  │       ▲                              │      │
  │       └────────── Sensor ◄───────────┘      │
  │    (corrects errors using feedback)         │
  └─────────────────────────────────────────────┘

  Achievement goal = reach a target (e.g., move to position)
  Maintenance goal = stay at target (e.g., hold position)
```

---

## 7. PARTICLE FILTER & LOCALIZATION (PS3 Q1: 33pts)

### 4-Step Algorithm (each timestep)
```
  1. INITIALIZE: scatter N particles randomly across state space

  2. MOVE (PREDICT): apply motion model + Gaussian noise to each particle
     Θ' = Θ + N(0, σ_Θ)       d' = d + N(0, σ_d)
     x_{t+1} = x_t + d'·cos(Θ')
     y_{t+1} = y_t + d'·sin(Θ')
     ★ Robot TURNS FIRST (Θ), THEN MOVES (x,y)  [PS3 Q1e]

  3. UPDATE (WEIGHT): assign weight = P(reading | location)
     ┌──────────────────────────────────────────────────────────┐
     │  Measurement model (PS3 Q1c: B):                        │
     │    P(sensor reading | robot at location) — a likelihood  │
     │                                                          │
     │  Posterior via Bayes (PS3 Q1b: C):                       │
     │    P(loc|read) = P(read|loc) · P(loc) / P(read)         │
     └──────────────────────────────────────────────────────────┘
     Uses Gaussian PDF: closer predicted↔actual → higher weight

  4. RESAMPLE: draw N new particles ∝ weights (roulette wheel)
     ★ Purpose (PS3 Q1a: A): concentrate on HIGH-PROB particles
       (discard low-weight, duplicate high-weight)
     ★ N_eff = 1/Σw_i² — resample when N_eff is low
     ★ Use log(p) to avoid numerical underflow
     ★ After resampling: all weights reset to 1/N
```

### PF vs KF (PS3 Q1g — 10 points!)
```
  ┌──────────────────────────────────────────┬────┬────┐
  │ Concept                                  │ PF │ KF │
  ├──────────────────────────────────────────┼────┼────┤
  │ Belief with particles (samples)          │ ✓  │    │
  │ Assumes Gaussian noise + linear system   │    │ ✓  │
  │ Uses motion & measurement models         │ ✓  │ ✓  │
  │ Handles nonlinear / non-Gaussian         │ ✓  │    │
  │ Single mean + covariance matrix          │    │ ✓  │
  │ Prediction-update (Bayes filtering)      │ ✓  │ ✓  │
  │ May require resampling                   │ ✓  │    │
  │ Uses optimal gain (Kalman gain)          │    │ ✓  │
  │ Requires initial estimate                │ ✓  │ ✓  │
  │ Can do localization AND SLAM             │ ✓  │ ✓  │
  └──────────────────────────────────────────┴────┴────┘

  ★ PF advantage over KF (PS3 Q1d: A):
    Handles NON-LINEAR + NON-GAUSSIAN systems
```

---

## 8. SLAM (PS3 Q2: 20pts, Practice Midterm Q6: 15pts)

### What is SLAM? (PS3 Q2a: C)
```
  Map an UNKNOWN environment WHILE tracking robot's position
  Chicken-and-egg: need map to localize, need location to map

  Why it's hard:
  • Uncertainty compounds over time
  • Data association: which observation ↔ which landmark?
  • Loop closure: recognizing previously visited places
```

### Why EKF, not basic KF? (PS3 Q2b-c)
```
  Why not KF?  → SLAM models are NONLINEAR (Q2b: B)

  EKF = recursive estimator that uses JACOBIANS
        to locally LINEARIZE nonlinear models (Q2c: B)
        [also tested in Practice Midterm Q6a]
```

### Loop Closure (PS3 Q2d: C)
```
  Robot recognizes a previously visited location
    → REDUCES UNCERTAINTY in both pose and map
```

### EKF-SLAM (PS3 Q2e, Practice Midterm Q2d fill-in)
```
  State vector: [x, y, θ, x₁, y₁, ..., xₙ, yₙ]
                 └─pose─┘  └──── n landmarks ────┘

  Covariance matrix: (2n+3) × (2n+3)

  ┌──────────────────────────────────────────────────────┐
  │ PREDICTION step: uses MOTION MODEL    (fill-in: L)   │
  │ UPDATE step:     uses SENSOR DATA     (fill-in: K)   │
  │ Belief updated using BAYES rule       (fill-in: N)   │
  └──────────────────────────────────────────────────────┘

  ★ Motion update complexity:  O(n²)   (covariance matrix update)
  ★ Overall EKF-SLAM:         O(n³)   per step — impractical for large n
```

### FastSLAM (Practice Midterm Q6 — 15 points!)
```
  ┌──────────────────────────────────────────────────────────┐
  │ FastSLAM = particle filter (trajectory)                  │
  │          + local EKFs (one per landmark per particle)    │
  │                                                          │
  │ Each particle maintains its own EKF for EACH landmark    │
  │                                                          │
  │ Total # EKFs = M × N  (Q6b)                             │
  │   Ex: 50 particles × 20 landmarks = 1000 EKFs           │
  │                                                          │
  │ Complexity per step: O(M · n)  (Q6c)                    │
  │                                                          │
  │ ★ FastSLAM more efficient when M ≪ n  (Q6d)             │
  │   Because O(M·n) < O(n²) when M < n                     │
  └──────────────────────────────────────────────────────────┘
```

### EKF-SLAM vs FastSLAM — Summary
```
  ┌──────────────┬───────────────────┬───────────────────┐
  │              │    EKF-SLAM       │    FastSLAM       │
  ├──────────────┼───────────────────┼───────────────────┤
  │ State        │ Joint (pose+map)  │ Factored          │
  │ Belief       │ One Gaussian      │ Particles + EKFs  │
  │ Per-step     │ O(n²)             │ O(M · n)          │
  │ # EKFs       │ 1 (size 2n+3)    │ M × n (size 2)    │
  │ Scales to    │ Small n           │ Large n           │
  └──────────────┴───────────────────┴───────────────────┘

  M = # particles,  n = # landmarks
```

---

## 9. DYNAMIC MOVEMENT PRIMITIVES (DMPs)

### Q1: What is a DMP?
```
  A DMP is a framework for encoding and reproducing robot motions
  as dynamical systems. It combines:
    • A spring-damper attractor (pulls toward goal)
    • A learned forcing function (shapes the trajectory)

  Input:  a demonstrated trajectory (e.g., from kinesthetic teaching)
  Output: a dynamical system that reproduces and generalizes the motion
```

### Q2: Core DMP Equations
```
  ┌──────────────────────────────────────────────────────────┐
  │  TRANSFORMATION SYSTEM (the main dynamics):              │
  │                                                          │
  │    τ·v̇  =  K(g - x) - D·v + (g - x₀)·f(s)             │
  │    τ·ẋ  =  v                                             │
  │                                                          │
  │  where:                                                   │
  │    x   = position           v    = velocity               │
  │    g   = goal position      x₀   = start position        │
  │    K   = spring constant    D    = damping constant       │
  │    τ   = temporal scaling   f(s) = learned forcing fn     │
  │                                                          │
  │  ★ Without f(s):  pure spring-damper → straight to goal  │
  │  ★ With f(s):     shapes the trajectory path             │
  ├──────────────────────────────────────────────────────────┤
  │  FORCING FUNCTION (learned from demo):                   │
  │                                                          │
  │    f(s) = Σᵢ wᵢ·ψᵢ(s)·s  /  Σᵢ ψᵢ(s)                  │
  │                                                          │
  │  where:                                                   │
  │    wᵢ   = learned weights (one per basis function)       │
  │    ψᵢ(s)= Gaussian basis:  exp(-hᵢ·(s - cᵢ)²)          │
  │    cᵢ   = center of i-th basis function                  │
  │    hᵢ   = width of i-th basis function                   │
  │    s    = phase variable from canonical system            │
  ├──────────────────────────────────────────────────────────┤
  │  CANONICAL SYSTEM (phase clock):                         │
  │                                                          │
  │    τ·ṡ  =  -α·s           (s: 1 → 0 over time)          │
  │                                                          │
  │  ★ s decays exponentially                                │
  │  ★ As s→0, f(s)→0, so system converges to attractor      │
  │  ★ Replaces explicit time → temporal invariance          │
  └──────────────────────────────────────────────────────────┘
```

### Q3: How Are the Weights wᵢ Learned?
```
  Step 1 — Compute the target forcing function from demo:

    f_target(s) = [ -K(g - x) + D·v + τ·v̇ ] / (g - x₀)
                    ↑ rearrange transformation system,
                      plug in demo trajectory (x, v, v̇)

  Step 2 — Fit weights via LINEAR REGRESSION:

    Minimize:  J = Σ_s ( f_target(s) - f(s) )²

    Since f(s) is linear in wᵢ, this is a simple
    least-squares problem → closed-form solution!

  ★ Key insight: only the weights wᵢ are learned.
    K, D, α, τ, basis centers/widths are fixed hyperparameters.
```

### Q4: Why Does a DMP Guarantee Convergence?
```
  ┌──────────────────────────────────────────────────────────┐
  │  1. Canonical system: s decays exponentially (s→0)       │
  │  2. Forcing function: f(s) = (...) · s → f→0 as s→0     │
  │  3. Remaining system:  τv̇ = K(g-x) - Dv                 │
  │     → This is a critically damped spring-damper          │
  │     → It ALWAYS converges to equilibrium at x = g        │
  │                                                          │
  │  ★ The forcing term "dies out" → attractor takes over    │
  │  ★ No matter what perturbation happens mid-execution,    │
  │    the system will still converge to the goal g          │
  └──────────────────────────────────────────────────────────┘
```

### Q5: Key Properties That Make DMPs Useful
```
  1. GUARANTEED CONVERGENCE to goal (see above)
  2. SPATIAL INVARIANCE
     → Change goal g → trajectory scales/shifts automatically
     → Same weights work for different start/goal positions
  3. TEMPORAL INVARIANCE
     → Change τ → speed up or slow down without changing shape
     → Phase variable s replaces explicit time
  4. ROBUSTNESS TO PERTURBATION
     → Push the robot mid-execution → it recovers
     → Attractor dynamics pull it back toward the goal
  5. SIMPLE LEARNING
     → Linear regression (closed-form, fast, one-shot)
     → Only need a single demonstration
```

### Q6: Limitations of DMPs
```
  1. PURELY KINEMATIC
     → Only encodes positions/velocities, NOT forces/torques
     → Cannot handle contact-rich tasks (e.g., insertion)
  2. LIMITED COORDINATION
     → Each DOF gets its own DMP; coupling between DOFs
       is not explicitly modeled
  3. COMBINING DMPs IS UNCLEAR
     → How to sequence or blend DMPs for complex multi-step
       tasks is an open problem
  4. SINGLE DEMONSTRATION
     → Learns one trajectory shape; doesn't capture
       variability across multiple demonstrations
```

---

## 10. BEHAVIORAL CLONING & DAgger

```
  Types of Demonstrations
  ───────────────────────
  • Kinesthetic teaching: physically move robot's joints
  • Teleoperation: remote-control the robot
  • Passive observation: watch human do it (hardest — correspondence problem)

  Behavioral Cloning (BC)
  ───────────────────────
  • Supervised learning: learn π_θ(a|s) from expert demos
  • Loss:  L(θ) = -E[ log π_θ(a|s) ]   (max likelihood)
  • i.i.d. assumption BREAKS: current state depends on previous actions
  • Problem: DISTRIBUTION SHIFT / COMPOUNDING ERROR
    → Small errors → drift to unseen states → more errors → crash
    → Error grows O(T²) with horizon T

  DAgger (Dataset Aggregation)  — fixes distribution shift
  ─────────────────────────────
  Loop:
    1. Run LEARNED policy → collect visited states
    2. Ask EXPERT to label those states with correct actions
    3. Aggregate new data with existing dataset
    4. Retrain policy
  ★ Key limitation: requires interactive expert (human in the loop)

  Multimodal Demonstration Problem
  ─────────────────────────────────
  • Different demos show different strategies for same state
    (e.g., go left OR right around obstacle)
  • BC with unimodal policy (e.g., Gaussian) → averages modes
    → dangerous: goes straight into obstacle!
  • Solutions: GMMs, Diffusion Policy (generative models)

  Diffusion Policy (Generative Approach)
  ───────────────────────────────────────
  • Uses denoising diffusion to model action distribution
  • Start from noise → iteratively denoise → produce action
  • Can represent MULTIMODAL distributions (multiple peaks)
  • Captures full diversity of expert strategies without averaging

  HYDRA (Hybrid Robot Actions)
  ────────────────────────────
  • Combines sparse WAYPOINT actions + dense LOW-LEVEL actions
  • Benefits: high-level planning with fine-grained control
  • Hybrid action space for imitation learning
```

---

## 11. INVERSE REINFORCEMENT LEARNING (IRL)

### BC vs IRL
```
  ┌────────────────────┬────────────────────────────────────┐
  │ Behavioral Cloning │ Learns POLICY directly: π(a|s)    │
  │                    │ "Copy what expert does"            │
  │                    │ Cheap but brittle (dist. shift)    │
  ├────────────────────┼────────────────────────────────────┤
  │ Inverse RL         │ Learns REWARD R(s) from demos     │
  │                    │ "Infer WHY expert acts that way"   │
  │                    │ Then optimize policy via RL         │
  │                    │ More robust, transfers better      │
  └────────────────────┴────────────────────────────────────┘
```

### IRL Formulation
```
  Given:  MDP\R = (S, A, T, γ, D)   ← everything EXCEPT reward
          Expert demos: (s₀,a₀), (s₁,a₁), ..., (sₙ,aₙ) ~ π^E

  Find:   R* such that expert is optimal under R*:

    E[Σ γᵗ R*(sₜ) | π^E]  ≥  E[Σ γᵗ R*(sₜ) | π]   ∀π

  i.e., expert policy gets higher value than ANY other policy
```

### Q7: What Is Guided Cost Learning?
```
  ┌──────────────────────────────────────────────────────────┐
  │  Guided Cost Learning (Finn et al., 2016)                │
  │                                                          │
  │  A deep IRL algorithm for learning nonlinear reward      │
  │  functions when dynamics are UNKNOWN.                    │
  │                                                          │
  │  Key idea: Iteratively alternate between:                │
  │    ① Learning a reward (cost) function from demos        │
  │    ② Optimizing a policy under that reward               │
  │    ③ Using policy samples to improve reward learning     │
  │                                                          │
  │  Assumption: Don't know dynamics T(s'|s,a), but CAN     │
  │  sample (interact with environment), like standard RL.   │
  ├──────────────────────────────────────────────────────────┤
  │  ALGORITHM (Nonlinear IOC with stochastic gradients):    │
  │                                                          │
  │  for iteration k = 1 to K:                               │
  │    1. Sample demo batch D̂_demo ⊂ D_demo                 │
  │    2. Sample background batch D̂_samp ⊂ D_samp            │
  │       (trajectories from current policy π)               │
  │    3. Append demo batch to background batch:             │
  │       D̂_samp ← D̂_demo ∪ D̂_samp                          │
  │    4. Estimate gradient dL_IOC/dθ using D̂_demo, D̂_samp  │
  │    5. Update reward parameters θ using gradient          │
  │    6. Update policy π w.r.t. new reward                  │
  │  return optimized cost parameters θ                      │
  ├──────────────────────────────────────────────────────────┤
  │  WHY IT WORKS:                                           │
  │  • Policy samples approximate the partition function     │
  │    (normalization constant) in max-entropy IRL           │
  │  • As policy improves → better samples → better reward   │
  │  • Bootstrapping loop: reward ↔ policy improve together  │
  │                                                          │
  │  KEY ADVANTAGES:                                         │
  │  • Handles unknown dynamics (model-free)                 │
  │  • Learns complex nonlinear cost functions (neural nets) │
  │  • Demonstrated on real robot tasks (e.g., dish placing) │
  └──────────────────────────────────────────────────────────┘
```

---

## 12. RL CHALLENGES IN ROBOTICS

```
  Four main practical challenges:
  ┌────────────────────┬─────────────────────────────────────┐
  │ Sample Efficiency  │ Needs massive amounts of data       │
  │ Safety             │ Risk of damage during exploration   │
  │ Reset              │ Hard to return to initial state     │
  │ Reward Specification│ Designing reward capturing intent  │
  └────────────────────┴─────────────────────────────────────┘

  Reward Hacking: RL agent exploits loopholes in the reward
  function to get high reward without accomplishing the task.

  Imitation Learning addresses reward specification by learning
  directly from human demonstrations — the human implicitly
  encodes the objective through their behavior.
```

---

## EXAM TIPS

```
  ★ DH: DRAW FRAMES FIRST! Then read off parameters from geometry
  ★ Camera: Remember [R|t] goes FROM world TO camera (not camera pose)
  ★ IK: Always two solutions (elbow up / down) — check both
  ★ Bellman: Know the difference between fixed-π and optimal equations
  ★ PRM vs RRT: Multi-query vs single-query is the key distinction
  ★ Policy Iteration: Fewer iterations but each is more expensive
  ★ DMP: f(s)→0 as s→0 is WHY convergence is guaranteed
  ★ BC vs IRL: BC copies actions, IRL infers the reward function
  ★ Guided Cost Learning: iterates reward learning ↔ policy optimization
  ★ PF: turn FIRST then move. Posterior = Bayes theorem (PS3 Q1b: C)
  ★ PF vs KF table: 10 pts on PS3! Memorize all 10 rows
  ★ SLAM: KF fails because nonlinear → need EKF (Jacobians)
  ★ FastSLAM: Total EKFs = M×N, complexity O(M·n), wins when M ≪ n
  ★ EKF-SLAM: predict=motion model(L), update=sensor data(K), Bayes(N)
  ★ Loop closure → REDUCES uncertainty (not increases, not restarts)
  ★ Self-loop V(s) = R/(1-γ). Q(s,a) = R(s,a) + γ·V(s')
```
