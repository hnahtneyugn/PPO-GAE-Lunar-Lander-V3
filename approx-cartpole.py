import gymnasium as gym
import numpy as np
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo


def get_feature(state, action):
    limits = np.array([4.8, 3.0, 0.418, 4.0])
    state = np.clip(state, -limits, limits) / limits
    pos, vel, angle, ang_vel = state

    if action == 0:
        f = np.concatenate([state, [vel * angle, vel * ang_vel], [0, 0, 0, 0, 0, 0]])
    else:
        f = np.concatenate([[0, 0, 0, 0, 0, 0], state, [vel * angle, vel * ang_vel]])
    return f


def get_Q_value(state, action, w):
    return np.dot(get_feature(state, action), w)


def get_max_Q_value(state, w):
    return max(get_Q_value(state, 0, w), get_Q_value(state, 1, w))


def get_difference(state, action, reward, state_prime, w, gamma):
    return reward + gamma * get_max_Q_value(state_prime, w) - get_Q_value(state, action, w)


def get_max_action(state, w):
    return 0 if get_Q_value(state, 0, w) > get_Q_value(state, 1, w) else 1


# Training parameters
n_epoch = 2000
gamma = 0.99
alpha = 0.02
epsilon_start = 0.8
epsilon_end = 0.02
epsilon_decay = 0.995
C = 0.0
target_steps = 500

env = gym.make('CartPole-v1')
w = np.zeros(12)
n_steps = []
best_score = 0
pbar = tqdm(range(n_epoch))

epsilon = epsilon_start
for k in pbar:
    state, _ = env.reset()
    done = False
    count = 0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = get_max_action(state, w)

        state_prime, reward, done, truncated, _ = env.step(action)
        done = done or truncated

        difference = get_difference(
            state, action, reward, state_prime, w, gamma)
        w += alpha * difference * get_feature(state, action)

        state = state_prime
        count += 1

    n_steps.append(count)
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    ave_steps = np.mean(n_steps[-100:])
    if ave_steps > best_score:
        best_score = ave_steps
        maxW = w.copy()
    pbar.set_description(
        f"Ep {k+1} Steps: {count} Avg100: {ave_steps:.1f} Eps: {epsilon:.3f} MaxQ: {get_max_Q_value(state, w):.2f}"
    )

np.save("maxW.npy", maxW)
print(f"Best 100-episode average: {best_score}")

# Evaluation
env = RecordVideo(gym.make("CartPole-v1", render_mode='rgb_array'), "./mp4", name_prefix="approx")
state, _ = env.reset()
maxW = np.load("maxW.npy")
total_reward = 0
steps = 0

for _ in range(1000):
    action = get_max_action(state, maxW)
    state, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        break

env.close()
