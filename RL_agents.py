import numpy as np
import lasagne

import theano

from lasagne.layers import InputLayer, DimshuffleLayer, Pool2DLayer, DenseLayer, ExpressionLayer, batch_norm
from lasagne.regularization import regularize_network_params, l2

from super_agent import MdpAgent

from agentnet.environment import SessionPoolEnvironment
from agentnet.resolver import EpsilonGreedyResolver
from agentnet.memory import WindowAugmentation
from agentnet.learning import qlearning_n_step
from agentnet.agent import Agent


class BasicRLAgent(MdpAgent):
    def __init__(self, pool, observation_shape, n_actions, n_parallel_games=1,
                 replay_seq_len=20, replay_batch_size=20, pool_size=None, n_steps=3, gamma=0.99):
        """
          :type n_parallel_games: int
                n_actions: int
          """
        # Parameters for training

        self.n_parallel_games = n_parallel_games
        self.replay_seq_len = replay_seq_len
        self.replay_batch_size = replay_batch_size
        self.pool_size = pool_size
        self.n_steps = n_steps
        self.gamma = gamma
        self.loss = None

        # image observation
        self.observation_layer = InputLayer(observation_shape)
        self.n_actions = n_actions
        self.resolver, self.agent = self.build_model()

        weights = lasagne.layers.get_all_params(self.resolver, trainable=True)
        self.applier_fun = self.agent.get_react_function()

        # Prepare replay pool
        env = SessionPoolEnvironment(observations=self.observation_layer,
                                     actions=self.resolver,
                                     agent_memories=self.agent.state_variables)

        preceding_memory_states = list(pool.prev_memory_states)

        # get interaction sessions
        observation_tensor, action_tensor, reward_tensor, _, is_alive_tensor, _ = \
            pool.interact(self.step, n_steps=self.replay_seq_len)
        env.load_sessions(observation_tensor, action_tensor, reward_tensor, is_alive_tensor,
                          preceding_memory_states)

        if pool_size is None:
            batch_env = env
        else:
            batch_env = env.sample_session_batch(self.replay_batch_size)

        self.loss = self.build_loss(batch_env)
        self.eval_fun = self.build_eval_fun(batch_env)

        updates = lasagne.updates.adadelta(self.loss, weights, learning_rate=0.01)
        train_fun = theano.function([], [self.loss], updates=updates)
        super(BasicRLAgent, self).__init__(env, pool, train_fun, pool_size, n_steps)

    def step(self, observation, prev_memories="zeros", batch_size=None):
        """
        returns actions and new states given observation and prev state
        Prev state in default setup should be [prev window,]
        """
        # default to zeros
        if batch_size is None:
            batch_size = self.n_parallel_games
        if prev_memories == 'zeros':
            prev_memories = [np.zeros((batch_size,) + tuple(mem.output_shape[1:]),
                                      dtype='float32')
                             for mem in self.agent.agent_states]
        res = self.applier_fun(np.array(observation), *prev_memories)
        action = res[0]
        memories = res[1:]
        return action, memories

    def build_model(self):

        # reshape to [batch, color, x, y] to allow for convolution layers to work correctly
        observation_reshape = DimshuffleLayer(self.observation_layer, (0, 3, 1, 2))
        observation_reshape = Pool2DLayer(observation_reshape, pool_size=(2, 2))

        # memory
        window_size = 5
        # prev state input
        prev_window = InputLayer((None, window_size) + tuple(observation_reshape.output_shape[1:]),
                                 name="previous window state")

        # our window
        memory_layer = WindowAugmentation(observation_reshape,
                                          prev_window,
                                          name="new window state")

        memory_dict = {memory_layer: prev_window}

        # pixel-wise maximum over the temporal window (to avoid flickering)
        memory_layer = ExpressionLayer(memory_layer, lambda a: a.max(axis=1),
                                       output_shape=(None,) + memory_layer.output_shape[2:])

        # neural network body
        nn = batch_norm(lasagne.layers.Conv2DLayer(memory_layer, num_filters=16, filter_size=(8, 8), stride=(4, 4)))
        nn = batch_norm(lasagne.layers.Conv2DLayer(nn, num_filters=32, filter_size=(4, 4), stride=(2, 2)))
        nn = batch_norm(lasagne.layers.DenseLayer(nn, num_units=256))
        # q_eval
        policy_layer = DenseLayer(nn, num_units=self.n_actions, nonlinearity=lasagne.nonlinearities.linear,
                                  name="QEvaluator")
        # resolver
        resolver = EpsilonGreedyResolver(policy_layer, name="resolver")

        # all together
        agent = Agent(self.observation_layer, memory_dict, policy_layer, resolver)

        return resolver, agent

    def build_loss(self, env):

        _, _, _, _, qvalues_seq = self.agent.get_sessions(
            env,
            session_length=self.replay_seq_len,
            batch_size=self.replay_batch_size,
            optimize_experience_replay=True,
            # unroll_scan=,
        )
        scaled_reward_seq = env.rewards

        elwise_mse_loss = qlearning_n_step.get_elementwise_objective(qvalues_seq,
                                                                     env.actions[0],
                                                                     scaled_reward_seq,
                                                                     env.is_alive,
                                                                     n_steps=self.n_steps,
                                                                     gamma_or_gammas=self.gamma, )

        mse_loss = elwise_mse_loss.sum() / env.is_alive.sum()

        reg_l2 = regularize_network_params(self.resolver, l2) * 10 ** -4

        loss = mse_loss + reg_l2

        return loss

    def build_eval_fun(self, env):

        mean_session_reward = env.rewards.sum(axis=1).mean() / self.replay_seq_len

        eval_fun = theano.function([],[self.loss, mean_session_reward])
        return eval_fun
