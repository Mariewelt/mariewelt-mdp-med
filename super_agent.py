
class MdpAgent(object):
    def __init__(self, env, pool, train_fun, pool_size=150, replay_seq_len=20):
        self.env = env
        self.pool = pool
        self.train_fun = train_fun
        self.train_fun = train_fun
        self.pool_size = pool_size
        self.replay_seq_len = replay_seq_len

    def step(self, observation, prev_memories='zeros'):
        action = None
        memories = None
        return action, memories

    def update_pool(self, observation_tensor, action_tensor, reward_tensor, is_alive_tensor,
                    memory_tensor, preceding_memory_states):
        self.env.append_sessions(observation_tensor, action_tensor, reward_tensor, is_alive_tensor,
                                 preceding_memory_states, max_pool_size=self.pool_size)

    def reload_pool(self, observation_tensor, action_tensor, reward_tensor, is_alive_tensor,
                    memory_tensor, preceding_memory_states):
        self.env.load_sessions(observation_tensor, action_tensor, reward_tensor, is_alive_tensor,
                               preceding_memory_states)

    def fit(self, n_epochs=100):
        for epoch_counter in xrange(n_epochs):
            preceding_memory_states = list(self.pool.prev_memory_states)

            # get interaction sessions
            observation_tensor, action_tensor, reward_tensor, memory_tensor, is_alive_tensor, _ = \
                self.pool.interact(self.step, n_steps=self.replay_seq_len)

            # load new sessions into the replay pool
            if self.pool_size is None:
                self.reload_pool(observation_tensor, action_tensor, reward_tensor, is_alive_tensor,
                                 memory_tensor, preceding_memory_states)
            else:
                self.update_pool(observation_tensor, action_tensor, reward_tensor, is_alive_tensor,
                                 memory_tensor, preceding_memory_states)

            self.train_fun()
