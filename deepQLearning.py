import argparse
import os
from datetime import datetime

import tensorflow as tf
from keras.layers import *
from keras.models import Model, save_model, load_model
from keras.optimizers import Adam

from agents import QLearningAgent


def representation(currState):
    elevators = currState.elevators
    floors = currState.floors

    state = []

    for e in elevators:
        state.append(e.currentFloor)
        state.append(np.power(2, e.pressedFloors).sum())
        state.append(e.getUsersIn() < e.maxPassengerCount)

    up_pressed = []
    down_pressed = []
    for f in floors:
        if f.up_requested():
            up_pressed.append(f.floorNum)

        if f.down_requested():
            down_pressed.append(f.floorNum)

    # state.append(np.power(2, list(set(up_pressed + down_pressed))).sum())

    state.append(sum([2 ** p for p in up_pressed], 0))
    state.append(sum([2 ** p for p in down_pressed], 0))

    return tuple(state)

class DeepQAgent(QLearningAgent):
    """
        ApproximateQLearningAgent

        You should only have to overwrite getQValue
        and update.  All other QLearningAgent functions
        should work as is.
    """

    def __init__(self, numFloors=7, numElevators=1, saveDir='deepLearningAgents', **args):
        super().__init__(**args)
        self.net = None
        self.batch = {'x': [], 'rewards': [], 'nexts': []}
        self.numFloors = numFloors
        self.numElevators = numElevators

        self.configStr = 'floors={}_elevators={}_training={}_exploration={}_epsilon={}_gamma={}'.format(self.numFloors,
                                                                                                        self.numElevators,
                                                                                                        self.numTraining,
                                                                                                        self.numExploration,
                                                                                                        self.epsilon,
                                                                                                        self.discount)

        self.inputShape = self.numElevators * (self.numFloors * 2  + 1) + self.numFloors * 2 + self.numFloors * self.numElevators
        self.cache = {}
        self.cacheHits = 0
        self.saveDir = saveDir
        self.graph = None
        self.avgTrainReward = []
        self.testRewards = []
        self.avgWaitTime = []
        self.layersStr = ''
        self.buildModel([256, 256])

    def getConfig(self):
        d = super().getConfig()
        d['layers'] = self.layersStr

        return d

    def buildModel(self, layerSizes):
        self.graph = tf.get_default_graph()
        self.graph._unsafe_unfinalize()

        inputState = Input((self.inputShape,))

        x = inputState
        for i, size in enumerate(layerSizes):
            x = Dense(size)(x)
            x = LeakyReLU()(x)

        out = Dense(1)(x)

        model = Model(inputState, out)
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=5e-4))

        self.net = model
        self.net._make_predict_function()
        self.net._make_train_function()
        self.graph.finalize()

        for ls in layerSizes:
            self.layersStr += str(ls) + '-'

        self.layersStr = self.layersStr[:-1]

    def saveModel(self, savePath):
        save_model(self.net, savePath + '.h5py')

    def loadModel(self, modelPath, testMode=True):
        self.testMode = testMode
        self.graph = tf.get_default_graph()
        self.graph._unsafe_unfinalize()
        self.net = load_model(modelPath)
        self.net._make_predict_function()
        self.net._make_train_function()
        self.graph.finalize()

    def stateToInput(self, state):
        inputState = []
        for e in state.elevators:
            inputState += [1 if i == e.currentFloor else 0 for i in range(self.numFloors)]
            inputState += [1 if i in e.pressedFloors else 0 for i in range(self.numFloors)]
            if e.getUsersIn() < e.maxPassengerCount:
                inputState.append(1)
            else:
                inputState.append(0)

        inputState += [1 if floor.up_requested() else 0 for floor in state.floors]
        inputState += [1 if floor.down_requested() else 0 for floor in state.floors]
        return inputState

    def actionToInput(self, action):
        inputAction = []
        for i in range(self.numElevators):
            inputAction += [1 if j == action[i] else 0 for j in range(self.numFloors)]

        return inputAction

    def stateActionToInput(self, state, action):
        return np.array(self.stateToInput(state) + self.actionToInput(action))

    def update(self, state, action, nextState, reward):
        """
        transfer states to inputs, get max over action with next state on model,
        calculate target (reward + gamma * max(Qvalues(nextState,actions))
        """
        if self.testMode:
            return
        self.batch['x'].append(self.stateActionToInput(state, action))

        nextActions = self.getLegalActions(nextState)
        inputState = self.stateToInput(nextState)
        inputs = [np.array(inputState + self.actionToInput(nextAction)) for nextAction in nextActions]
        self.batch['rewards'].append(reward)
        self.batch['nexts'] += inputs

    def stopEpisode(self):
        """
            Called by environment when episode is done
        """
        if self.testMode:
            self.testRewards.append(self.episodeRewards)
            return

        # perform backprop
        targets = self.net.predict(np.stack(self.batch['nexts']), batch_size=len(self.batch['nexts']))
        targets = targets.reshape(targets.size // self.numFloors ** self.numElevators, -1).max(1) * self.discount
        targets += np.stack(self.batch['rewards'])

        with self.graph.as_default():
            self.net.train_on_batch(np.stack(self.batch['x']), targets)

        self.batch['x'] = []
        self.batch['rewards'] = []
        self.batch['nexts'] = []

        self.cache = {}
        # print('Cache Hits: ', self.cacheHits)
        self.cacheHits = 0

        # save_model(self.net, os.path.join(self.saveDir, self.configStr + '.h5py'))

        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        self.trained = self.episodesSoFar

        if self.episodesSoFar % 100 == 0:
            self.epsilon = (1. - self.episodesSoFar / self.numTraining) * self.startEpsilon
            # self.alpha = (1. - self.episodesSoFar / self.numTraining) * self.startAlpha
            self.alpha = self.startAlpha * 0.98 ** (self.episodesSoFar * 100 / self.numTraining)
            self.avgTrainReward.append(self.lastWindowAccumRewards / 100)

        if self.episodesSoFar >= self.numExploration:
            self.epsilon = 0.0

        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0        # no exploration
            self.alpha = 0.0            # no learning

    def final(self):
        super().final()
        # if self.episodesSoFar % 100 == 0:
        #     save_model(self.net, os.path.join(self.saveDir, self.configStr + '.h5py'))
        #     self.saveStats()

    def saveStats(self):
        np.savez(os.path.join(self.saveDir, self.configStr + '.npz'), avgTrainReward=self.avgTrainReward, testReward=self.testRewards)

    def getPolicy(self, state):
        """
            Compute the best action to take in a state.    Note that if there
            are no legal actions, which is the case at the terminal state,
            you should return None.
        """
        actions = self.getLegalActions(state)
        stateRepr = representation(state)


        if stateRepr in self.cache.keys():
            vals = self.cache[stateRepr]
            self.cacheHits += 1
        else:
            with self.graph.as_default():
                inputs = np.stack([self.stateActionToInput(state, action) for action in actions])
                vals = np.squeeze(self.net.predict(inputs))
                self.cache[stateRepr] = vals


        i = np.argmax(vals)
        return actions[i], vals



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--layers', type=int, nargs='+', required=True)
    parser.add_argument('-e', '--elevators', type=int, default=2)
    parser.add_argument('-f', '--floors', type=int, default=7)
    parser.add_argument('-t', '--numTraining', type=int, default=80000)
    parser.add_argument('-ex', '--numExploration', type=int, default=65000)
    parser.add_argument('-eps', '--epsilon', type=float, default=0.5)
    parser.add_argument('-a', '--alpha', type=float, default=0.5)
    parser.add_argument('-g', '--gamma', type=float, default=0.9)

    args = parser.parse_args()

    # d = vars(args)
    # order = ['floors', 'elevators', 'numTraining', 'numExploration', 'epsilon', 'alpha', 'gamma']
    # configString = ''
    # for key in order:
    #     configString += '{}={}_'.format(key, d[key])
    #
    # configString += 'layers='
    # for l in d['layers']:
    #     configString += '{}_'.format(l)
    #
    # configString = configString[:-1]

    from simulator import Simulator
    sim = Simulator(num_elevators=args.elevators, num_floors=args.floors)
    agent = DeepQAgent(args.floors, args.elevators,
                       actionFn=sim.mdp.goToAnyFloor,
                       numTraining=args.numTraining,
                       numExploration=args.numExploration,
                       epsilon=args.epsilon,
                       alpha=args.alpha,
                       gamma=args.gamma)
    agent.buildModel(args.layers)
    sim.train(agent)