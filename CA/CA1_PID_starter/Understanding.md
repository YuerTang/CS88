The Big Picture: How Everything Works Together                                                                               
                                                                                                                               
  The Simulation Loop                                                                                                          
                                                                                                                               
  ┌─────────────────────────────────────────────────────────────────────┐                                                      
  │                         test.py                                      │                                                     
  │                                                                      │                                                     
  │   env = suite.make("Stack", ...)     ← Create simulation world       │                                                     
  │                                                                      │                                                     
  │   for episode in range(5):           ← Run 5 episodes                │                                                     
  │       obs = env.reset()              ← Randomize object positions    │                                                     
  │       policy = StackPolicy(obs)      ← Create YOUR policy            │                                                     
  │                                                                      │                                                     
  │       while True:                    ← Main control loop             │                                                     
  │           action = policy.get_action(obs)   ← YOUR CODE decides      │                                                     
  │           obs, reward, done, _ = env.step(action)  ← Physics sim     │                                                     
  │           if reward == 1.0: break    ← Task complete!                │                                                     
  │                                                                      │                                                     
  └─────────────────────────────────────────────────────────────────────┘                                                      
                                                                                                                               
  Data Flow Diagram                                                                                                            
                                                                                                                               
  ┌──────────────────────────────────────────────────────────────────────────┐                                                 
  │                                                                          │                                                 
  │  RoboSuite Environment                                                   │                                                 
  │  ┌────────────────────┐                                                  │                                                 
  │  │ Physics Simulation │                                                  │                                                 
  │  │ - Robot arm        │                                                  │                                                 
  │  │ - Cubes/Objects    │                                                  │                                                 
  │  │ - Collisions       │                                                  │                                                 
  │  └─────────┬──────────┘                                                  │                                                 
  │            │                                                             │                                                 
  │            ▼                                                             │                                                 
  │   obs = {                          ┌─────────────────────────────────┐   │                                                 
  │     'robot0_eef_pos': [x,y,z],     │        YOUR POLICY              │   │                                                 
  │     'cubeA_pos': [x,y,z],    ───────►                                │   │                                                 
  │     'cubeB_pos': [x,y,z],          │  1. Check current phase         │   │                                                 
  │     ...                            │  2. Get current waypoint        │   │                                                 
  │   }                                │  3. Call PID.update()           │   │                                                 
  │                                    │  4. Build action vector         │   │                                                 
  │            ▲                       │                                 │   │                                                 
  │            │                       └───────────────┬─────────────────┘   │                                                 
  │            │                                       │                     │                                                 
  │   env.step(action) ◄───────────────────────────────┘                     │                                                 
  │                                                                          │                                                 
  │   action = [dx, dy, dz, ax, ay, az, gripper]                            │                                                  
  │             └────┬────┘                  └──┬──┘                         │                                                 
  │                  │                         │                             │                                                 
  │           PID output                  -1=open, +1=close                  │                                                 
  │                                                                          │                                                 
  └──────────────────────────────────────────────────────────────────────────┘                                                 
                                                                                                                               
  How PID and Policy Work Together                                                                                             
                                                                                                                               
  ┌─────────────────────────────────────────────────────────────────┐                                                          
  │                        StackPolicy                               │                                                         
  │                                                                  │                                                         
  │  __init__(obs):                                                  │                                                         
  │  ┌─────────────────────────────────────────────────────────┐    │                                                          
  │  │ 1. Read cube positions from obs                          │    │                                                         
  │  │ 2. Calculate waypoints relative to cubes                 │    │                                                         
  │  │ 3. Create PID controller with first waypoint as target   │    │                                                         
  │  │ 4. Set phase = 0                                         │    │                                                         
  │  └─────────────────────────────────────────────────────────┘    │                                                          
  │                                                                  │                                                         
  │  get_action(obs):                                                │                                                         
  │  ┌─────────────────────────────────────────────────────────┐    │                                                          
  │  │                                                          │    │                                                         
  │  │   current_pos = obs['robot0_eef_pos']  ← Where am I?     │    │                                                         
  │  │                                                          │    │                                                         
  │  │   # Check if reached current waypoint                    │    │                                                         
  │  │   if self.pid.get_error() < 0.01:                        │    │                                                         
  │  │       self.phase += 1                  ← Next phase      │    │                                                         
  │  │       self.pid.reset(next_waypoint)    ← New target      │    │                                                         
  │  │                                                          │    │                                                         
  │  │   # Get control signal from PID                          │    │                                                         
  │  │   control = self.pid.update(current_pos, dt)             │    │                                                         
  │  │              └──────────┬──────────────┘                 │    │                                                         
  │  │                         │                                │    │                                                         
  │  │                         ▼                                │    │                                                         
  │  │              ┌─────────────────────┐                     │    │                                                         
  │  │              │     PID Controller  │                     │    │                                                         
  │  │              │                     │                     │    │                                                         
  │  │              │ error = target - pos│                     │    │                                                         
  │  │              │ P = Kp * error      │                     │    │                                                         
  │  │              │ I = Ki * integral   │                     │    │                                                         
  │  │              │ D = Kd * derivative │                     │    │                                                         
  │  │              │ return P + I + D    │                     │    │                                                         
  │  │              └──────────┬──────────┘                     │    │                                                         
  │  │                         │                                │    │                                                         
  │  │                         ▼                                │    │                                                         
  │  │   # Build 7D action                                      │    │                                                         
  │  │   action = [control[0], control[1], control[2],          │    │                                                         
  │  │             0, 0, 0,           ← No rotation             │    │                                                         
  │  │             gripper_state]     ← -1 or +1                │    │                                                         
  │  │                                                          │    │                                                         
  │  │   return action                                          │    │                                                         
  │  └─────────────────────────────────────────────────────────┘    │                                                          
  │                                                                  │                                                         
  └─────────────────────────────────────────────────────────────────┘                                                          
                                                                                