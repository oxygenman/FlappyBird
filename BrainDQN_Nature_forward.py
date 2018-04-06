# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100. # timesteps to observe before training
EXPLORE = 200000. # frames over which to anneal epsilon
FINAL_EPSILON = 0#0.001 # final value of epsilon
INITIAL_EPSILON = 0#0.01 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 100

try:
    tf.mul
except:
    # For new version of tensorflow
    # tf.mul has been removed in new version of tensorflow
    # Using tf.multiply to replace tf.mul
    tf.mul = tf.multiply

class BrainDQN:
    def __init__(self,actions):
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        # init Q network
        self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 ,self.max_h_conv1,self.max_h_conv2,self.max_h_conv3,self.max_h_pool1,self.max_h_fc1,self.max_h_QValue= self.createQNetwork()
        self.session = tf.InteractiveSession()
        # I change somthing
        #self.max_h_conv1=0

        #self.max_h_conv2=0
        #self.max_h_conv3=0
        #self.max_h_pool1=0
        #self.max_h_fc1=0
        #self.max_h_QValue=0
        self.max_c1 = self.max_c2 = self.max_c3 = self.max_p1 = self.max_fc1 = self.max_Q = 0
        self.session.run(tf.initialize_all_variables())
        #self.saver=tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("svaed_wb")
        if (checkpoint and checkpoint.model_checkpoint_path):
            self.saver_wb.restore(self.session, checkpoint.model_checkpoint_path)
            print ("Successfully loaded:", checkpoint.model_checkpoint_path)
            variable_list=tf.train.list_variables(checkpoint.model_checkpoint_path)
            print(variable_list)
        else:
            print ("Could not find old network weights")

    def createQNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([8,8,4,32],'W_conv1')
        print(W_conv1.name)
        b_conv1 = self.bias_variable([32],'b_conv1')
        print(b_conv1.name)

        W_conv2 = self.weight_variable([4,4,32,64],'W_conv2')
        b_conv2 = self.bias_variable([64],'b_conv1')

        W_conv3 = self.weight_variable([3,3,64,64],'W_conv3')
        b_conv3 = self.bias_variable([64],'b_conv3')

        W_fc1 = self.weight_variable([1600,512],'W_fc1')
        b_fc1 = self.bias_variable([512],'b_fc1')

        W_fc2 = self.weight_variable([512,self.actions],'W_fc2')
        b_fc2 = self.bias_variable([self.actions],'b_fc2')

        # input layer

        stateInput = tf.placeholder("float",[None,80,80,4])


        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
        max_h_conv1=tf.reduce_max(h_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        max_h_pool1 =tf.reduce_max(h_pool1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1,W_conv2,2) + b_conv2)

        max_h_conv2 = tf.reduce_max(h_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)

        max_h_conv3 = tf.reduce_max(h_conv3)

        h_conv3_flat = tf.reshape(h_conv3,[-1,1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)
        max_h_fc1 = tf.reduce_max(h_fc1)

        # Q Value layer
        QValue = tf.matmul(h_fc1,W_fc2) + b_fc2
        max_h_QValue = tf.reduce_max(QValue)
        self.saver_wb = tf.train.Saver(
            {'W_conv1': W_conv1, 'W_conv2': W_conv2, 'W_conv3': W_conv3, 'W_fc1': W_fc1, 'W_fc2': W_fc2,
             'b_conv1': b_conv1, 'b_conv2': b_conv2, 'b_conv3': b_conv3, 'b_fc1': b_fc1, 'b_fc2': b_fc2})


                                        #'h_conv1': h_conv1, 'h_conv2':h_conv2, 'h_conv3':h_conv3, 'h_pool1':h_pool1, 'h_conv3_flat':h_conv3_flat,'h_fc1':h_fc1,'QValue':QValue})

        return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2,max_h_conv1,max_h_conv2,max_h_conv3,max_h_pool1,max_h_fc1,max_h_QValue


    def getAction(self):
        QValue,W_conv1,max_h_conv1,max_h_conv2,max_h_conv3,max_h_pool1,max_h_fc1,max_h_QValue= self.session.run([self.QValue,self.W_conv1,self.max_h_conv1,self.max_h_conv2,self.max_h_conv3,self.max_h_pool1,self.max_h_fc1,self.max_h_QValue],feed_dict= {self.stateInput:[self.currentState]})#因为在网络的设置中 QValue是二维的
        QValue = QValue[0]
        print('QValue:',QValue)
        self.max_c1=np.maximum(self.max_c1,max_h_conv1)
        self.max_c2 = np.maximum(self.max_c2, max_h_conv2)
        self.max_c3 = np.maximum(self.max_c3, max_h_conv3)
        self.max_p1 = np.maximum(self.max_p1, max_h_pool1)
        self.max_fc1 = np.maximum(self.max_fc1, max_h_fc1)
        self.max_Q = np.maximum(self.max_Q, max_h_QValue)
        if self.timeStep>500:
            maxvalues=np.array([self.max_c1,self.max_p1,self.max_c2,self.max_c3,self.max_fc1,self.max_Q])
            np.save('maxvalues.npy',maxvalues)
        action = np.zeros(self.actions)
        action_index = 0
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1 # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

        return action

    def setPerception(self, nextObservation, action, reward, terminal):
        # 此处说明输入的是连续的四帧图片
        newState = np.append(self.currentState[:, :, 1:], nextObservation, axis=2)
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.timeStep, "/ STATE", state, \
              "/ EPSILON", self.epsilon)
        print("max value of every layer",self.max_c1,self.max_c2,self.max_c3,self.max_p1,self.max_fc1,self.max_Q)
        self.currentState = newState
        self.timeStep += 1

    def setInitState(self,observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

    def weight_variable(self,shape,name):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial,name=name)

    def bias_variable(self,shape,name):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial,name=name)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

