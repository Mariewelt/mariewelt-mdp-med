from globals import N_PARALLEL_GAMES

import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, DimshuffleLayer,EmbeddingLayer,ExpressionLayer
from lasagne.layers import DropoutLayer,DenseLayer, batch_norm,Conv2DLayer

from agentnet.memory import GRUCell
from agentnet.resolver import EpsilonGreedyResolver
from agentnet.agent import Agent




class MetaController:
    def __init__(self,
                 controller,
                 gru0_size=128,
                 ):

        #image observation at current tick goes here
        self.observed_state = InputLayer(controller.dnn_output.output_shape,
                                       name="cnn output")

        prev_gru0 = InputLayer((None,gru0_size),name='prev gru0')

        self.gru0 = GRUCell(prev_state=prev_gru0,input_or_inputs=self.observed_state)

        memory_dict = {self.gru0:prev_gru0}


        #q_eval
        q_eval = DenseLayer(self.gru0,
                           num_units = controller.n_goals,
                           nonlinearity=lasagne.nonlinearities.linear,
                           name="QEvaluator")

        #resolver
        self.resolver = EpsilonGreedyResolver(q_eval,name="resolver")
        
        

        #all together
        self.agent = Agent(self.observed_state,
                      memory_dict,
                      q_eval,
                      [self.resolver,q_eval])



        self.controller = controller
        self.observation_shape = controller.dnn_output.output_shape
        self.n_goals = controller.n_goals
        self.period = controller.metacontroller_period
        self.applier_fun = self.agent.get_react_function()
        
        
        self.weights = lasagne.layers.get_all_params(self.resolver,trainable=True)


    def step(self,observation, prev_memories, batch_size):
        """ returns actions and new states given observation and prev state
        Prev state in default setup should be [prev window,]"""
        # default to zeros


        if prev_memories == 'zeros':
            prev_memories = [np.zeros((batch_size,) + tuple(mem.output_shape[1:]),
                                      dtype='float32')
                             for mem in self.agent.agent_states]
        res = self.applier_fun(np.array(observation),
                          *prev_memories)
        action,q_eval = res[:2]
        memories = res[2:]
        return action, memories, q_eval.max(axis=-1)
