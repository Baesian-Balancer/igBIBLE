import gym
import time
import functools
from gym_ignition.utils import logger
from igBIBLE import randomizers

import numpy as np
import random

# Set verbosity
# logger.set_level(gym.logger.ERROR)
logger.set_level(gym.logger.DEBUG)

# Available tasks
env_id = "Monopod-Gazebo-v1"


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    import igBIBLE
    return gym.make(env_id, **kwargs)


# Create a partial function passing the environment id
make_env = functools.partial(make_env_from_id, env_id=env_id)

# Wrap the environment with the randomizer.
# This is a simple example no randomization are applied.

env = randomizers.monopod_no_rand.MonopodEnvNoRandomizations(env=make_env)

# Wrap the environment with the randomizer.
# This is a complex example that randomizes both the physics and the model.
# env = randomizers.monopod.MonopodEnvRandomizer(
#     env=make_env, seed=42, num_physics_rollouts=5)

# Enable the rendering
env.render('human')

# Initialize the seed
env.seed(42)

epochs = 100
Q = np.zeros((state_size, action_size))

for epoch in range(epochs):

    # Reset the environment
    observation = env.reset()

    # Initialize returned values
    done = False
    totalReward = 0
    epsilon = 0.2 * (1 - epoch / epochs) + 0.05

    while not done:

        # Render the environment.
        # It is not required to call this in the loop if physics is not randomized.
        # env.render('human')

        if random.uniform(0, 1) < epsilon:
            """
            Explore: select a random action    """
            action = env.action_space.sample()
        else:
            """
            Exploit: select the action with max value (future reward)    """

        observation, reward, done, _ = env.step(action)

        # Accumulate the reward
        totalReward += reward

        # Print the observation
        msg = ""
        for value in observation:
            msg += "\t%.6f" % value
        logger.debug(msg)

    print(f"Reward episode #{epoch}: {totalReward}")

env.close()
time.sleep(5)
