## Init Environment ##
# ENV_Name = 'CTEnv-v0'

## Get the environment and extract the number of actions. ##

## Build model ##

## Configure and compile our agent. ##
# Example
# memory = SequentialMemory(limit=50000, window_length=1)
# policy = BoltzmannQPolicy()
# dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
#                target_model_update=1e-2, policy=policy)
# dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

## Agent learning and training ##
# Example
# dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

## Save weights ##
# Example
# dqn.save_weights(f'dqn_{ENV_NAME}_weights.h5f', overwrite=True)

## Evaluate algorithm after 'n' episodes ##
# Example
# dqn.test(env, nb_episodes=5, visualize=True)