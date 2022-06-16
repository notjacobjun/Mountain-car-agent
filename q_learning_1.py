from xml.dom.pulldom import END_ELEMENT
import numpy as np
import gym

env = gym.make("MountainCar-v0")


def get_discrete_state(state):
    """
    Converts the given state into a discrete form
    - This is used to help reduce the size of the q-table so that we don't have trouble training our agent
    """
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))


# INITIALIZE some variables before starting training
# this is how much we are changing our q values are observing a new reward
LEARNING_RATE = 0.1
# this is the time-decay to ensure that more recent observations are weighted more than less recent observations
DISCOUNT = 0.95
# this is for epslion greedy learning to help explore more
EPISODES = 25000
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

SHOW_EVERY = 2000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
# Change continuous observation space into discrete space. This is to help save time in Q-learning
discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low) / DISCRETE_OS_SIZE

# Creating the Q-table with every possible state-action pair. (noticed that we reduced the # of possible states by making it discrete)
q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
    # get the initial state from env.reset function and convert into discrete state
    discrete_state = get_discrete_state(env.reset())

    # keep iterating episodes until we have reached the optimal policy
    done = False
    while not done:
        # using epislon-greedy approach
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        if not done:
            # this gets the max q value from the q_table where it is choosing the max q values based on the
            # 3 actions available at that new discrete state-action pair
            max_future_q = np.max(q_table[new_discrete_state])
            # since we are indexiing by tuple we appending a tuple (which just stores the action)
            current_q = q_table[discrete_state + (action, )]
            # using the Bellman optimality equation to update our q value
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * \
                (reward + DISCOUNT * max_future_q)
            # update our q table with this new q_value
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] > env.goal_position:
            print(f"We made it on episode: {episode}")
            # set the q value for reaching the goal state
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()
