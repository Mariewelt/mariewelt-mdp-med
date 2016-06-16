import lasagne

from agentnet.environment.SessionPoolEnvironment import append_sessions
from lasagne.layers import batch_norm

class MdpAgent(object):
    def __init__(self, env, pool, train_fun, pool_size=150, n_steps=50):
        self.env = env
        self.pool = pool
        self.train_fun = train_fun
        self.pool_size = pool_size
        self.n_steps = n_steps

    def step(self):
        pass

    def fit(self, n_epochs=100):

        for i in xrange(n_epochs):
            preceding_memory_states = list(self.pool.prev_memory_states)

            # get interaction sessions
            observation_tensor, action_tensor, reward_tensor, _, is_alive_tensor, _ = \
                self.pool.interact(self.step, n_steps=self.n_steps)

            # load new sessions into the replay pool
            append_sessions(observation_tensor, action_tensor, reward_tensor, is_alive_tensor,
                            preceding_memory_states, max_pool_size=self.pool_size)

            loss = self.train_fun()



