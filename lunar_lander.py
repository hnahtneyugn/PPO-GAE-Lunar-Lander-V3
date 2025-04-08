import gymnasium as gym
import numpy as np
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo

# Create the environment
env = gym.make(
    "LunarLander-v3",
    continuous=False,      # Discrete actions
    gravity=-10.0,         # Gravity value
    enable_wind=False,     # No wind
    wind_power=15.0,       # Wind power (not used since wind is disabled)
    turbulence_power=1.5,  # Turbulence strength
    render_mode="rgb_array" # Required for video recording
)

# Wrap the environment with RecordVideo to save the simulation
env = RecordVideo(env, video_folder="lunar_lander_videos", episode_trigger=lambda x: True)

# Set number of episodes to run
n_episodes = 10

# Run the simulation
for episode in tqdm(range(n_episodes)):
    observation, info = env.reset()  # Reset environment at start of each episode
    done = False
    total_reward = 0
    
    while not done:
        # Simple random agent: choose a random action
        action = env.action_space.sample()
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Check if episode is done
        done = terminated or truncated
    
    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

# Close the environment
env.close()