import gymnasium as gym
import numpy as np
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo


env = gym.make('CartPole-v1')
state = env.reset()


def get_obs(state):
    obs_real = state
    bins = [np.linspace(-2.4, 2.4, 15),
            np.linspace(-2, 2, 15),
            np.linspace(-0.42, 0.42, 15),
            np.linspace(-3.5, 3.5, 15)]
    obs = np.array([0, 0, 0, 0])
    for i in range(4):
        # print('---', obs_real[i], bins[i])
        obs[i] = np.digitize(obs_real[i], bins[i])
    return obs


Q = np.zeros((16, 16, 16, 16, 2))

# training
n_epoch = 12000
gamma = 0.9
alpha = 0.1
epsilon = 0.03
n_steps = []
pbar = tqdm(range(n_epoch))
max_ave_steps = 0
for k in pbar:
    state = env.reset()  # s0
    obs = get_obs(state[0])
    done = False
    count = 0
    while not done:
        if np.random.rand() < epsilon: 
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[obs[0], obs[1], obs[2], obs[3], :])
        state_prime, reward, done, truncated, _ = env.step(action)
        obs_prime = get_obs(state_prime)
        if (count < 499) and (done or truncated):
            reward -= (500-count)*10
        Q[obs[0], obs[1], obs[2], obs[3], action] = \
            (1 - alpha) * Q[obs[0], obs[1], obs[2], obs[3], action] + \
            alpha * \
            (reward + gamma *
             np.max(Q[obs_prime[0], obs_prime[1], obs_prime[2], obs_prime[3], :]))
        obs = obs_prime
        count += 1
        done = done or truncated
    n_steps.append(count)
    ave_steps = sum(n_steps[-50:]) / len(n_steps[-50:])
    if ave_steps > max_ave_steps:
        maxQ = Q.copy()
        max_ave_steps = ave_steps
    pbar.set_description(f"Epoch {k+1} Average steps: {ave_steps}")

# render the environment in 1000 steps
# and take best action by Q table
# save to a video file
env = RecordVideo(gym.make("CartPole-v1", render_mode='rgb_array'),"./basic_cartpole")
state, _ = env.reset()
Q = maxQ
for _ in range(1000):
    obs = get_obs(state)
    action = np.argmax(Q[obs[0], obs[1], obs[2], obs[3], :])
    state, reward, done, _, _ = env.step(action)
    if done:
        break
env.close()
