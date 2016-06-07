
import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, DimshuffleLayer,EmbeddingLayer,ExpressionLayer
from lasagne.layers import DropoutLayer,DenseLayer, batch_norm,Conv2DLayer

from agentnet.memory import WindowAugmentation
from agentnet.resolver import EpsilonGreedyResolver
from agentnet.memory.logical import CounterLayer
from agentnet.agent import Agent




class Controller:
    def __init__(self,
                 observation_shape,
                 n_actions,
                 n_goals=32,
                 metacontroller_period=5,
                 window_size=3,
                 embedding_size=128,
                 ):

        #image observation at current tick goes here
        self.observation_layer = InputLayer(observation_shape,
                                       name="images input")


        #reshape to [batch, color, x, y] to allow for convolutional layers to work correctly
        observation_reshape = DimshuffleLayer(self.observation_layer,(0,3,1,2))

        observation_reshape = lasagne.layers.Pool2DLayer(observation_reshape,(2,2),mode='average_inc_pad')



        #prev state input
        prev_window = InputLayer((None,window_size)+tuple(observation_reshape.output_shape[1:]),
                                name = "previous window state")
        #our window
        window = WindowAugmentation(observation_reshape,
                                    prev_window,
                                    name = "new window state")
        # pixel-wise maximum over the temporal window (to avoid flickering)
        window_max = ExpressionLayer(window,
                                     lambda a: a.max(axis=1),
                                     output_shape=(None,) + window.output_shape[2:])


        memory_dict = {window: prev_window}

        #a simple lasagne network (try replacing with any other lasagne network and see what works best)
        nn = batch_norm(Conv2DLayer(window_max,16,filter_size=8,stride=(4,4), name='cnn0'))
        nn = batch_norm(Conv2DLayer(nn,32,filter_size=4,stride=(2,2), name='cnn1'))
        nn = batch_norm(Conv2DLayer(nn,64,filter_size=4,stride=(2,2), name='cnn2'))

        #nn = DropoutLayer(nn,name = "dropout", p=0.05) #will get deterministic during evaluation
        self.dnn_output = nn = DenseLayer(nn,num_units=256,name='dense1')


        self.goal_layer = InputLayer((None,), T.ivector(), name='boss goal')
        self.goal_layer.output_dtype = 'int32'
        goal_emb = EmbeddingLayer(self.goal_layer, n_goals, embedding_size)

        nn = lasagne.layers.ConcatLayer([goal_emb,nn])


        #q_eval
        q_eval = DenseLayer(nn,
                           num_units = n_actions,
                           nonlinearity=lasagne.nonlinearities.linear,
                           name="QEvaluator")

        #resolver
        self.resolver = EpsilonGreedyResolver(q_eval,name="resolver")

        #all together
        self.agent = Agent([self.observation_layer,self.goal_layer],
                      memory_dict,
                      q_eval,
                      [self.resolver,self.dnn_output])



        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.n_goals = n_goals
        self.metacontroller_period = metacontroller_period
        self.window_size = window_size
        self.embedding_size = embedding_size

        self.applier_fun = self.agent.get_react_function()
        
        self.weights = lasagne.layers.get_all_params(self.resolver,trainable=True)



    def step(self,observation,goal,prev_memories, batch_size):
        """ returns actions and new states given observation and prev state
        Prev state in default setup should be [prev window,]"""
        # default to zeros
        if prev_memories == 'zeros':
            prev_memories = [np.zeros((batch_size,) + tuple(mem.output_shape[1:]),
                                      dtype='float32')
                             for mem in self.agent.agent_states]
        res = self.applier_fun(np.array(observation),
                          np.array(goal),
                          *prev_memories)
        action,metacontroller_inp = res[:2]
        memories = res[2:]
        return action, memories, metacontroller_inp


