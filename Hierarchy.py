import numpy as np
import lasagne

import theano

from lasagne.regularization import regularize_network_params, l2

from super_agent import MdpAgent

from agentnet.environment import SessionPoolEnvironment
from agentnet.utils.layers import get_layer_dtype
from agentnet.learning import qlearning_n_step

from controller import Controller
from metacontroller import MetaController


class HierarchicalAgent(MdpAgent):
    def __init__(self, pool, observation_shape, n_actions, n_parallel_games=1,
                 replay_seq_len=20, replay_batch_size=20, pool_size=None, n_steps=3, gamma=0.99,
                 split_into=1,): #gru0_size=128):
        self.n_parallel_games = n_parallel_games
        self.replay_seq_len = replay_seq_len
        self.replay_batch_size = replay_batch_size
        self.pool_size = pool_size
        self.n_steps = n_steps
        self.n_actions = n_actions
        self.gamma = gamma
        self.split_into = split_into
        self.controller = Controller(observation_shape, n_actions)
        self.metacontroller = MetaController(self.controller)#, gru0_size)

        # Prepare replay pool
        self.controller_env = SessionPoolEnvironment(observations=self.controller.agent.observation_layers,
                                                     actions=self.controller.resolver,
                                                     agent_memories=self.controller.agent.agent_states)
        self.metacontroller_env = SessionPoolEnvironment(observations=self.metacontroller.agent.observation_layers,
                                                         actions=self.metacontroller.resolver,
                                                         agent_memories=self.metacontroller.agent.agent_states)

        # get interaction sessions
        observation_log, action_tensor, extrinsic_reward_log, memory_log, is_alive_tensor, _ = \
            pool.interact(self.step, n_steps=self.replay_seq_len)
        preceding_memory_states = list(pool.prev_memory_states)
        self.reload_pool(observation_log, action_tensor, extrinsic_reward_log, is_alive_tensor,
                         memory_log, preceding_memory_states)

        if pool_size is None:
            controller_batch_env = self.controller_env
            metacontroller_batch_env = self.metacontroller_env
        else:
            controller_batch_env = self.controller_env.sample_session_batch(self.replay_batch_size)
            metacontroller_batch_env = self.metacontroller_env.sample_session_batch(self.replay_batch_size)

        self.loss = self.build_loss(controller_batch_env, self.controller.agent, 50) + \
                    self.build_loss(metacontroller_batch_env, self.metacontroller.agent, 10)
        self.eval_fun = self.build_eval_fun(metacontroller_batch_env)

        weights = self.controller.weights + self.metacontroller.weights
        updates = lasagne.updates.adadelta(self.loss, weights, learning_rate=0.01)
        mean_session_reward = metacontroller_batch_env.rewards.sum(axis=1).mean()
        train_fun = theano.function([], [self.loss, mean_session_reward], updates=updates)
        super(HierarchicalAgent, self).__init__([self.controller_env, self.metacontroller_env],
                                                pool, train_fun, pool_size, replay_seq_len)
        # raise NotImplementedError

    def reload_pool(self, observation_tensor, action_tensor, extrinsic_reward_tensor, is_alive_tensor,
                    memory_tensor, preceding_memory_states):
        batch_size = observation_tensor.shape[0]
        # print observation_tensor.shape
        meta_obs_log, goal_log, meta_V, itrs = memory_tensor[-4:]
        itr = itrs[0]

        pivot = len(self.controller.agent.state_variables)
        controller_preceding_states = preceding_memory_states[:pivot]
        metacontroller_preceding_states = preceding_memory_states[pivot:-4]

        ###CONTROLLER###
        # load them into experience replay environment for controller

        # controller_preceding_states =!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ctrl_shape = (batch_size * self.split_into, self.replay_seq_len / self.split_into)

        intrinsic_rewards = np.concatenate([np.zeros([meta_V.shape[0], 1]), np.diff(meta_V, axis=1)], axis=1)
        # print [observation_tensor.reshape(ctrl_shape + self.controller.observation_shape[1:]),
        #                                  goal_log.reshape(ctrl_shape)][0].shape
        self.controller_env.load_sessions([observation_tensor.reshape(ctrl_shape + self.controller.observation_shape[1:]),
                                          goal_log.reshape(ctrl_shape)],
                                          action_tensor.reshape(ctrl_shape),
                                          intrinsic_rewards.reshape(ctrl_shape),
                                          is_alive_tensor.reshape(ctrl_shape),
                                          # controller_preceding_states
                                          )

        ###METACONTROLLER###
        # separate case for metacontroller
        extrinsic_reward_sums = np.diff(
            np.concatenate(
                [np.zeros_like(extrinsic_reward_tensor[:, 0, None]),
                 extrinsic_reward_tensor.cumsum(axis=-1)[:, itr == 0]],
                axis=1
            )
        )

        self.metacontroller_env.load_sessions(meta_obs_log[:, itr == 0][:, :10],
                                              goal_log[:, itr == 0][:, :10],
                                              extrinsic_reward_sums[:, :10],
                                              is_alive_tensor[:, itr == 0][:, :10],
                                              metacontroller_preceding_states)

    def update_pool(self, observation_tensor, action_tensor, extrinsic_reward_tensor, is_alive_tensor,
                    memory_tensor, preceding_memory_states):
        batch_size = observation_tensor.shape[0]
        meta_obs_log, goal_log, meta_V, itrs = memory_tensor[-4:]
        itr = itrs[0]

        pivot = len(self.controller.agent.state_variables)
        controller_preceding_states = preceding_memory_states[:pivot]
        metacontroller_preceding_states = preceding_memory_states[pivot:-4]

        ###CONTROLLER###
        # load them into experience replay environment for controller

        # controller_preceding_states =!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ctrl_shape = (batch_size * self.split_into, self.replay_seq_len / self.split_into)

        intrinsic_rewards = np.concatenate([np.zeros([meta_V.shape[0], 1]), np.diff(meta_V, axis=1)], axis=1)
        self.controller_env.append_sessions([observation_tensor.reshape(ctrl_shape + self.controller.observation_shape[1:]),
                                            goal_log.reshape(ctrl_shape)],
                                            action_tensor.reshape(ctrl_shape),
                                            intrinsic_rewards.reshape(ctrl_shape),
                                            is_alive_tensor.reshape(ctrl_shape),
                                            controller_preceding_states,
                                            max_pool_size=self.pool_size,
                                           )

        ###METACONTROLLER###
        # separate case for metacontroller
        extrinsic_reward_sums = np.diff(
            np.concatenate(
                [np.zeros_like(extrinsic_reward_tensor[:, 0, None]),
                 extrinsic_reward_tensor.cumsum(axis=-1)[:, itr == 0]],
                axis=1
            )
        )

        self.metacontroller_env.append_sessions(meta_obs_log[:, itr == 0][:, :10],
                                              goal_log[:, itr == 0][:, :10],
                                              extrinsic_reward_sums[:, :10],
                                              is_alive_tensor[:, itr == 0][:, :10],
                                              metacontroller_preceding_states,
                                              max_pool_size=self.pool_size)

    def step(self, env_observation, prev_memories='zeros'):
        """ returns actions and new states given observation and prev state
        Prev state in default setup should be [prev window,]"""

        batch_size = self.n_parallel_games

        if prev_memories == 'zeros':
            controller_mem = metacontroller_mem = 'zeros'
            meta_inp = np.zeros((batch_size,) + tuple(self.metacontroller.observation_shape[1:]), dtype='float32')
            itr = -1
            # goal will be defined by "if itr ==0" clause
        else:
            pivot = len(self.controller.agent.state_variables)
            controller_mem, metacontroller_mem = prev_memories[:pivot], prev_memories[pivot:-4]
            meta_inp, goal, meta_V, itrs = prev_memories[-4:]
            itr = itrs[0]

        itr = (itr + 1) % self.metacontroller.period

        if itr == 0:
            goal, metacontroller_mem, meta_V = self.metacontroller.step(meta_inp, metacontroller_mem, batch_size)

        #print env_observation.shape
        action, controller_mem, meta_inp = self.controller.step(env_observation, goal, controller_mem, batch_size)

        new_memories = controller_mem + metacontroller_mem + [meta_inp, goal, meta_V, [itr] * self.n_parallel_games]

        return action, new_memories

    def build_loss(self, env, agent, replay_seq_len):
        # get agent's Qvalues obtained via experience replay
        _, _, _, _, qvalues_seq = agent.get_sessions(
            env,
            # initial_hidden = env.preceding_agent_memories,
            session_length=replay_seq_len,
            batch_size=env.batch_size,
            optimize_experience_replay=True,
        )

        scaled_reward_seq = env.rewards

        elwise_mse_loss = qlearning_n_step.get_elementwise_objective(qvalues_seq,
                                                                     env.actions[0],
                                                                     scaled_reward_seq,
                                                                     env.is_alive,
                                                                     gamma_or_gammas=self.gamma,
                                                                     n_steps=self.n_steps)

        # compute mean over "alive" fragments
        mse_loss = elwise_mse_loss.sum() / env.is_alive.sum()

        # regularize network weights

        reg_l2 = regularize_network_params(agent.state_variables.keys(), l2) * 10 ** -5

        return mse_loss + reg_l2

    def build_eval_fun(self, env):

        mean_session_reward = env.rewards.sum(axis=1).mean() / self.replay_seq_len

        eval_fun = theano.function([], [mean_session_reward])
        return eval_fun
