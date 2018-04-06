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
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100.  # timesteps to observe before training
EXPLORE = 200000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0  # 0.001 # final value of epsilon
INITIAL_EPSILON = 0 # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 100

try:
    tf.mul
except:
    # For new version of tensorflow
    # tf.mul has been removed in new version of tensorflow
    # Using tf.multiply to replace tf.mul
    tf.mul = tf.multiply


class BrainDQN:
    def __init__(self, actions):
        #self.v_conv1 =np.zeros([20,20,32],dtype=np.float32)
        #self.v_pool1 = np.zeros([10,10,32],dtype=np.float32)
        #self.v_conv2 = np.zeros([5,5,64],dtype=np.float32)
        #self.v_conv3 = np.zeros([5,5,64],dtype=np.float32)
        #self.v_conv3_flat = np.zeros([1600],dtype=np.float32)
        #self.v_fc1 =np.zeros([512],dtype=np.float32)
        #self.v_QValue =np.zeros([2],dtype=np.float32)
        maxvalues = np.load('maxvalues.npy')
        self.maxc1 = maxvalues[0]
        self.maxp1 = maxvalues[1]
        self.maxc2 = maxvalues[2]
        self.maxc3 = maxvalues[3]
        self.maxfc1 = maxvalues[4]
        self.maxQvalue = maxvalues[5]
        self.scaler_c1 = self.maxc1 /255
        self.scaler_p1 = self.maxp1 / self.maxc1
        self.scaler_c2 = self.maxc2 / self.maxp1
        self.scaler_c3 = self.maxc3 / self.maxc2
        self.scaler_fc1 = self.maxfc1 / self.maxc3
        self.scaler_Q = self.maxQvalue / self.maxfc1

        self.Twindow=20

        self.threshold=10
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        # init Q network
        self.stateInput, self.output,self.v_QValue, self.v_conv1,self.v_pool1, self.v_conv2, self.v_conv3, self.v_fc1 ,self.w_conv1= self.createQNetwork()

        ''' init Target Q Network
        self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T = self.createQNetwork()

        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                            self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2),
                                            self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3),
                                            self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)] '''
        #self.createTrainingMethod()

        # saving and loading networks
        #self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        writer = tf.summary.FileWriter("tmp", self.session.graph)
        self.session.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("svaed_wb")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver_wb.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        variab_list= tf.train.list_variables(checkpoint.model_checkpoint_path)

        print('maxvalues:',maxvalues)


    def createQNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512, self.actions])
        b_fc2 = self.bias_variable([self.actions])

        # input layer

        stateInput = tf.placeholder("float", [None,80, 80, 4])
        self.fire_conv1=tf.Variable(tf.zeros([1,20,20,32]))
        self.v_conv1=tf.Variable(self.fire_conv1.initial_value)

        v_pool1=tf.Variable(tf.zeros([1,10,10,32]))
        fire_pool1=tf.Variable(v_pool1.initial_value)

        v_conv2=tf.Variable(tf.zeros([1,5,5,64]))
        fire_conv2=tf.Variable(v_conv2.initial_value)

        v_conv3=tf.Variable(v_conv2.initial_value)
        fire_conv3=tf.Variable(v_conv2.initial_value)

        v_fc1=tf.Variable(tf.zeros([1,512]))
        fire_fc1=tf.Variable(v_fc1.initial_value)

        v_QValue = tf.Variable(tf.zeros([1, 2]))



        # hidden layers
        h_conv1 = self.conv2d(stateInput, W_conv1, 4)+b_conv1
        fc1,self.vc1=self.IF(h_conv1,self.v_conv1,self.fire_conv1,self.scaler_c1)
        self.vc1_assign=self.v_conv1.assign(self.vc1)

        self.h_pool1 = self.averge_pool_2x2(fc1)
        #self.fire_conv1.assign(self.rest_f(self.fire_conv1))
        fp1,self.vp1=self.IF(self.h_pool1,v_pool1,fire_pool1,self.scaler_p1)
        self.vp1_assign=v_pool1.assign(self.vp1)

        h_conv2 = self.conv2d(fp1, W_conv2, 2)+b_conv2
       # fire_pool1.assign(self.rest_f(fire_pool1))
        fc2,vc2=self.IF(h_conv2,v_conv2,fire_conv2,self.scaler_c2)
        self.vc2_assign =v_conv2.assign(vc2)

        h_conv3 = self.conv2d(fc2, W_conv3, 1)+b_conv3
        #fire_conv2.assign(self.rest_f(fire_conv2))
        fc3,vc3=self.IF(h_conv3,v_conv3,fire_conv3,self.scaler_c3)
        self.vc3_assign=v_conv3.assign(vc3)

        fire_conv3_flat = tf.reshape(fc3, [1,1600])

        h_fc1 = tf.matmul(fire_conv3_flat, W_fc1)+b_fc1
        #fire_conv3.assign(self.rest_f(fire_conv3))
        self.ffc1,vfc1=self.IF(h_fc1,v_fc1,fire_fc1,self.scaler_fc1)
        self.vfc1_assign=v_fc1.assign(vfc1)

        # Q Value layer
        QValue = tf.matmul(self.ffc1, W_fc2)+b_fc2
        output=tf.assign(v_QValue,tf.add(v_QValue,QValue))
        #fire_fc1.assign(self.rest_f(fire_fc1))
        self.saver_wb = tf.train.Saver(
            {'W_conv1': W_conv1, 'W_conv2': W_conv2, 'W_conv3': W_conv3, 'W_fc1': W_fc1, 'W_fc2': W_fc2,
             'b_conv1': b_conv1, 'b_conv2': b_conv2, 'b_conv3': b_conv3, 'b_fc1': b_fc1, 'b_fc2': b_fc2})

        return stateInput,output, v_QValue, self.v_conv1,v_pool1,v_conv2, v_conv3, v_fc1,W_conv1

    #IF model
    def IF(self,h,v,f,threshold):
        v_add=tf.assign(v, tf.add(v, h))
        v_less_zero=tf.where(tf.less(v_add, 0), tf.zeros_like(v),v_add)
        greater_cond = tf.greater(v_less_zero, threshold)
        fire=tf.where(greater_cond, tf.ones_like(f), f)
        v_greater_thr=tf.where(greater_cond, tf.zeros_like(v), v_less_zero)
        return fire,v_greater_thr

    # reset fire map
    def rest_f(self,f):
        f_new=tf.assign(f,f.initial_value)
        return f_new


    def setPerception(self, nextObservation, action, reward, terminal):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        newState = np.append(self.currentState[:, :, 1:], nextObservation, axis=2)
        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.timeStep, "/ STATE", state, \
              "/ EPSILON", self.epsilon)

        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        Twindow=300



        self.session.run(tf.variables_initializer([self.v_conv1, self.v_pool1, self.v_conv2, self.v_conv3, self.v_fc1,self.v_QValue]))
        while Twindow:
            v_QValue,v_conv1,v_pool1,v_conv2,v_conv3,v_fc1= self.session.run([self.output,self.vc1_assign,self.vp1_assign,self.vc2_assign,self.vc3_assign,self.ffc1],feed_dict={self.stateInput: [self.currentState]})# 因为在网络的设置中 QValue是二维的
            v_QValue=v_QValue[0]
            Twindow=Twindow-1
            #print(v_conv1)

        action = np.zeros(self.actions)
        action_index = 0
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(v_QValue)
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def setInitState(self, observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis=2)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def averge_pool_2x2(self, x):
        return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

