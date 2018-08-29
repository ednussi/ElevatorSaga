import numpy as np
import pickle

from agents import QLearningAgent, ReinforcementAgent

class MultiAgent(ReinforcementAgent):

    def __init__(self, num_agents, **kwargs):
        super().__init__(**kwargs)
        self.agents = [QLearningAgent(**kwargs) for i in range(num_agents)]
        self.name = 'MultiAgent'

        qTable = self.agents[0].qValues
        for agent in self.agents[1:]:
            agent.qValues = qTable

    def saveModel(self, confString):
        with open(confString + '.pkl', 'wb') as pfile:
            pickle.dump(self.agents[0].qValues, pfile)

    def loadModel(self, modelPath, testMode=True):
        self.testMode = testMode
        with open(modelPath, 'rb') as pfile:
            qTable = pickle.load(pfile)
        for agent in self.agents:
            agent.qValues = qTable
            agent.testMode = testMode

    def startEpisode(self):
        super().startEpisode()
        for agent in self.agents:
            agent.startEpisode()

    def worldStateToManyStates(self, state):
        stateRepr = list(state.repr)
        floors = [stateRepr[-1]]
        jmp = int((len(stateRepr) - 1) / len(self.agents))
        states = [stateRepr[i * jmp : (i + 1) * jmp] + floors for i in range(len(self.agents))]
        locs = [state[0] for state in states]
        for i in range(len(self.agents)):
            l = locs[:]
            l.pop(i)
            states[i] = states[i] + l
        return [tuple(state) for state in states]


    def getAction(self, state):
        states = self.worldStateToManyStates(state)
        action = []
        for i, agent in enumerate(self.agents):
            action += agent.getAction(states[i])[0]

        return tuple(action), 'N/A'


    def observeTransition(self, prev_state, prev_action, state, rewards):
        out_reward = rewards.pop()
        el_rewards = []
        size = int(len(rewards) / len(self.agents))
        for i in range(len(self.agents)):
            el_rewards.append(rewards[i * size : (i + 1) * size] + [out_reward])

        prev_states = self.worldStateToManyStates(prev_state)
        states = self.worldStateToManyStates(state)

        for i, agent in enumerate(self.agents):
            agent.observeTransition(prev_states[i], (prev_action[i],), states[i], el_rewards[i])


    def final(self):
        self.stopEpisode()
        qTableSize = len(self.agents[0].qValues)
        self.addMetricSample('qTableSize', qTableSize)

        self.lastWindowAccumRewards += self.episodeRewards

        if self.episodesSoFar % 100 == 0:
            self.print('Reinforcement Learning Status:')
            windowAvg = self.lastWindowAccumRewards / 100.0
            self.lastWindowAccumRewards = 0.0
            self.addMetricSample('avgReward:sum', windowAvg)


    def stopEpisode(self):
        super().stopEpisode()
        for agent in self.agents:
            agent.stopEpisode()
            self.episodeRewards += agent.episodeRewards
        self.episodeRewards /= len(self.agents)

    def toggleTestMode(self):
        super().toggleTestMode()
        for agent in self.agents:
            agent.toggleTestMode()

    def setNumTraining(self, numTraining):
        super().setNumTraining(numTraining)
        for agent in self.agents:
            agent.setNumTraining(numTraining)

    def setNumExploration(self, numExploration):
        super().setNumExploration(numExploration)
        for agent in self.agents:
            agent.setNumExploration(numExploration)

    def setEpsilon(self, epsilon):
        super().setEpsilon(epsilon)
        for agent in self.agents:
            agent.setEpsilon(epsilon)

    def setAlpha(self, alpha):
        super().setAlpha(alpha)
        for agent in self.agents:
            agent.setAlpha(alpha)

    def setDiscount(self, discount):
        super().setDiscount(discount)
        for agent in self.agents:
            agent.setDiscount(discount)