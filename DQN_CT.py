import numpy as np
from env.CTEnv import CTEnv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Permute
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, MaxPooling3D

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import pdb

WINDOW_LENGTH = 4

# 1. Get the environment and extract the number of actions.
env = CTEnv()
nb_actions = env.action_space.n # 2 options (SART or Superiorization)
shape = env.observation_space.shape

# 2. Next, we build a very simple model.
pool_size = 2

input_shape = (WINDOW_LENGTH, 512, 512)
model = Sequential()
# model.add(Permute((2, 3, 1), input_shape = input_shape))
model.add(Convolution2D(32, 3, padding="same", input_shape=input_shape)) # nb_filters = 32, conv1_size = 3 for reference
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size))) # reduces to 256x256

model.add(Convolution2D(64, 2, padding="same")) 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size))) # reduces to 128x128

model.add(Convolution2D(64, 2, padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size))) # reduces to 64x64 (16 million parameters)

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


# 3. Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(learning_rate=.00025), metrics=['mae'])

# 4. Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# 5. After training is done, we save the final weights.
dqn.save_weights(f'dqn_CTEnv_weights.h5f', overwrite=True)

# 6. Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)