import pickle
import random
import util
import tensorflow as tf
import numpy as np
from valueEstimationAgent import ValueEstimationAgent

NUM_EPS_UPDATE = 100

from itertools import cycle

class ReinforcementAgent(ValueEstimationAgent):
    """
        Abstract Reinforcement Agent: A ValueEstimationAgent
	    which estimates Q-Values (as well as policies) from experience
	    rather than a model

            What you need to know:
		    - The environment will call
		        observeTransition(state,action,nextState,deltaReward),
		        which will call update(state, action, nextState, deltaReward)
		        which you should override.
            - Use self.getLegalActions(state) to know which actions
		        are available in a state
    """
    def __init__(self, actionFn=None, numTraining=30000, numExploration=15000, epsilon=0.5, alpha=0.5, gamma=0.9):
        """
        actionFn: Function which takes a state and returns the list of legal actions

        alpha        - learning rate
        epsilon    - exploration rate
        gamma        - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        super().__init__(alpha, epsilon, gamma, numTraining)
        if actionFn is None:
                actionFn = lambda state: state.getLegalActions()
        self.actionFn = actionFn
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.numExploration = numExploration
        self.epsilon = float(epsilon)
        self.startEpsilon = float(epsilon)
        self.alpha = float(alpha)
        self.startAlpha = float(alpha)
        self.discount = float(gamma)
        self.lastWindowAccumRewards = 0.0
        self.episodeRewards = 0.0

    def getConfig(self):
        baseConf = super().getConfig()
        baseConf.update({'EXP': self.numExploration})
        return baseConf

    def update(self, state, action, nextState, reward):
        """
	        This class will call this function, which you write, after
	        observing a transition and reward
        """
        util.raiseNotDefined()

    def getLegalActions(self,state):
        """
            Get the actions available for a given
            state. This is what you should use to
            obtain legal actions for a state
        """
        return self.actionFn(state)

    def observeTransition(self, state, action, nextState, rewards):
        """
        	Called by environment to inform agent that a transition has
        	been observed. This will result in a call to self.update
        	on the same arguments

        	NOTE: Do *not* override or call this function
        """
        reward = sum(rewards, 0)
        self.episodeRewards += reward
        if not self.testMode:
            self.update(state, action, nextState, reward)

    def startEpisode(self):
        """
            Called by environment when new episode is starting
        """
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
            Called by environment when episode is done
        """
        if self.testMode:
            return

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

        if self.episodesSoFar >= self.numExploration:
            self.epsilon = 0.0

        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0        # no exploration
            self.alpha = 0.0            # no learning

    def final(self):
        """
            Called by game at the terminal state
        """
        self.stopEpisode()
        self.lastWindowAccumRewards += self.episodeRewards

        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            self.print('Reinforcement Learning Status:')
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            self.lastWindowAccumRewards = 0.0
            self.addMetricSample('avgReward:sum', windowAvg)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                self.print('\tCompleted %d out of %d training episodes' % (self.episodesSoFar,self.numTraining))
                self.print('\tAverage Rewards over all training: %.2f' % trainAvg)
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                self.print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
                self.print('\tAverage Rewards over testing: %.2f' % testAvg)
            self.print('\tAverage Rewards for last %d episodes: %.2f' % (NUM_EPS_UPDATE, windowAvg))


            self.print('\tepsilon: ', self.epsilon)
            self.print('\talpha: ', self.alpha)

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            self.print('%s\n%s' % (msg,'-' * len(msg)))

class QLearningAgent(ReinforcementAgent):
    """
        Q-Learning Agent

        Functions you should fill in:
            - getQValue
            - getAction
            - getValue
            - getPolicy
            - update

        Instance variables you have access to
            - self.epsilon (exploration prob)
            - self.alpha (learning rate)
            - self.discount (discount rate)

        Functions you should use
            - self.getLegalActions(state)
                which returns legal actions
                for a state
    """
    def __init__(self, **args):
        """You can initialize Q-values here..."""
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter()
        self.num_updates = 0
        self.prev_qValues = None
        self.hist = util.Counter()
        self.name = 'QLearningAgent'

    def getQValue(self, state, action):
        """
            Returns Q(state,action)
            Should return 0.0 if we never seen
            a state or (state,action) tuple
        """
        return self.qValues[(state, action)]

    def saveModel(self, confString):
        with open(confString, 'wb') as f:
            pickle.dump(self.qValues, f)

    def loadModel(self, modelPath, testMode=True):
        with open(modelPath, 'rb') as f:
            self.qValues = pickle.load(f)
        self.testMode = testMode

    def getValue(self, state):
        """
            Returns max_action Q(state,action)
            where the max is over legal actions.    Note that if
            there are no legal actions, which is the case at the
            terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0

        return max([self.getQValue(state, action) for action in actions])


    def getPolicy(self, state):
        """
            Compute the best action to take in a state.    Note that if there
            are no legal actions, which is the case at the terminal state,
            you should return None.
        """
        actions = self.getLegalActions(state)
        if not actions:
            return None

        vals = []
        for action in actions:
            vals.append(self.getQValue(state, action))

        maxIndices = [i for i, x in enumerate(vals) if x == max(vals)]
        i = random.choice(maxIndices)
        return actions[i], [round(n, 2) for n in vals]

    def getAction(self, state):
        """
            Compute the action to take in the current state.    With
            probability self.epsilon, we should take a random action and
            take the best policy action otherwise.    Note that if there are
            no legal actions, which is the case at the terminal state, you
            should choose None as the action.

            HINT: You might want to use util.flipCoin(prob)
            HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        if util.flipCoin(self.epsilon) and not self.testMode:
            return random.choice(legalActions), 'N\A'
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
            The parent class calls this to observe a
            state = action => nextState and reward transition.
            You should do your Q-Value update here

            NOTE: You should never call this function,
            it will be called on your behalf
        """
        self.qValues[(state, action)] += self.alpha * (reward + self.discount * self.getValue(nextState) - self.qValues[(state,action)])

    def final(self):
        super(QLearningAgent, self).final()
        if 1 == self.episodesSoFar:
            self.addMetricSample('qTableSize', 0)
        self.addMetricSample('qTableSize', len(self.qValues))


class RandomAgent(ValueEstimationAgent):
    def __init__(self, floors, elevators):
        super().__init__()
        self.episodeRewards = 0.0
        self.name = 'RandomAgent'
        self.floors = floors
        self.elevators = elevators

    def startEpisode(self):
        """
            Called by environment when new episode is starting
        """
        self.episodeRewards = 0.0

    # noinspection PyUnusedLocal
    def observeTransition(self, state, action, nextState, deltaReward):
        """
        	Called by environment to inform agent that a transition has
        	been observed. This is just implemented to maintain "Order"
        """
        self.episodeRewards += deltaReward

    # noinspection PyUnusedLocal
    def getAction(self, state):
        action = []
        for e in range(self.elevators):
            action.append(random.randint(0, self.floors-1))
        return tuple(action), [0] * self.floors

    # noinspection PyMethodMayBeStatic
    def getConfig(self):
        return {}


class ShabatAgent(ValueEstimationAgent):
    def __init__(self, floors, elevators):
        super().__init__()
        self.episodeRewards = 0.0
        self.name = 'ShabatAgent'
        self.floors = floors
        self.elevators = elevators
        self.numTraining = 0
        self.metrics = {}
        self.begin_floors = []
        for e in range(self.elevators):
            self.begin_floors.append(int(round(self.floors / (self.elevators - e))))  # random.randint(0, self.floors-1))

    def startEpisode(self):
        """
            Called by environment when new episode is starting
        """
        self.episodeRewards = 0.0

    # noinspection PyUnusedLocal
    def observeTransition(self, state, action, nextState, deltaReward):
        """
        	Called by environment to inform agent that a transition has
        	been observed. This is just implemented to maintain "Order"
        """
        self.episodeRewards += deltaReward

    # noinspection PyUnusedLocal
    def getAction(self, state):
        action = []
        for i in range(self.elevators):
            self.begin_floors[i] = (self.begin_floors[i]+1) % self.floors
            action.append(self.begin_floors[i])
        return tuple(action), [0] * self.floors

    # noinspection PyMethodMayBeStatic
    def getConfig(self):
        return {}


# noinspection SpellCheckingInspection
class PolicyGradientAgent(ReinforcementAgent):
    """
        Q-Learning Agent

        Functions you should fill in:
            - getQValue
            - getAction
            - getValue
            - getPolicy
            - update

        Instance variables you have access to
            - self.epsilon (exploration prob)
            - self.alpha (learning rate)
            - self.discount (discount rate)

        Functions you should use
            - self.getLegalActions(state)
                which returns legal actions
                for a state
    """
    def __init__(self, iterPerEp, floors=3, elevetors=1, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.floors = int(floors)
        self.elevators = int(elevetors)
        self.qValues = util.Counter()
        self.num_updates = 0
        self.prev_qValues = None
        self.hist = util.Counter()
        self.name = 'PolicyGradientAgent'
        self.epsilon = 0.1 #(Exploration)
        # Set learning parameters
        self.learning_rate = 0.1
        self.possible_actions = [(i,) for i in range(self.floors)]

        self.iter_per_episode = iterPerEp
        self.update_tag = False
        self.update_count = cycle(range(self.iter_per_episode))
        self.update_states = [0] * self.iter_per_episode
        self.update_actions = [0] * self.iter_per_episode
        self.update_rewards = [0] * self.iter_per_episode

        # Initialize default graph
        tf.reset_default_graph()

        # These lines establish the feed-forward part of the network used to choose actions
        input_size = self.getBinaryStateSize()
        # None is actually the iter per episode
        # self.input_state = tf.placeholder(shape=[None,input_size], dtype=tf.float32)
        # self.W = tf.Variable(tf.random_uniform([input_size, self.floors ** self.elevators], 0, 0.01))
        # self.logits = tf.matmul(self.input_state, self.W)
        def w_var(shape, name_num):
            initial = tf.truncated_normal(shape, stddev=0.001)
            return tf.Variable(initial, name='w{}'.format(name_num))

        def b_var(shape, name_num):
            initial = tf.constant(0.0, shape=shape)
            return tf.Variable(initial, name='b{}'.format(name_num))

        # 1 Affine
        # self.input_state = tf.placeholder(shape=[None,input_size], dtype=tf.float32)
        # self.w1 = w_var([input_size, self.floors ** self.elevators], name_num=1)
        # self.b1 = b_var([self.floors ** self.elevators], name_num=1)
        # self.logits = tf.matmul(self.input_state, self.w1) + self.b1
        #
        # self.saver = tf.train.Saver([self.w1, self.b1])

        # Decoders
        self.input_state = tf.placeholder(shape=[None,input_size], dtype=tf.float32)
        self.w1 = w_var([input_size, 32], name_num=1)
        self.b1 = b_var([32], name_num=1)
        self.h1 = tf.nn.leaky_relu(tf.matmul(self.input_state, self.w1) + self.b1)

        self.w2 = w_var([32, 32], name_num=2)
        self.b2 = b_var([32], name_num=2)
        self.h2 = tf.nn.leaky_relu(tf.matmul(self.h1, self.w2) + self.b2)

        self.w3 = w_var([32, self.floors ** self.elevators], name_num=3)
        self.b3 = b_var([self.floors ** self.elevators], name_num=3)
        self.logits = tf.matmul(self.h2, self.w3) + self.b3
        self.saver = tf.train.Saver([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])

        # classification loss - distrbution vector for actions
        self.actions = tf.placeholder(shape=[None, self.floors ** self.elevators], dtype=tf.float32) # True labels
        self.class_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.actions, logits=self.logits)

        # making the reward be taken into account
        self.rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        # l2_factor = 0
        # for l in [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]:
        #     l2_factor += 0.01 * tf.nn.l2_loss(l)
        self.loss = tf.reduce_sum(self.class_loss * tf.reduce_sum(self.rewards)) #+ l2_factor

        # compute gradients
        #                  tf.train.AdagradOptimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha, beta1=0.9, beta2=0.999, epsilon=1e-08)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.alpha, momentum=0.9, name='Momentum', use_nesterov=True)
        self.gradients = self.optimizer.compute_gradients(-self.loss)

        # define the train operation
        self.train_op = self.optimizer.apply_gradients(self.gradients)

        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def getBinaryStateSize(self):
        return (self.floors * 2 + 1) * self.elevators + self.floors

    def state2binaryState(self, state):
        # Representation is (<location>,<buttons>,<load>,<requests>)
        # Representation is (<#floors>,<buttons>,<load>,<requests>)
        location_f = ['0'] * self.floors
        location_f[state.repr[0]] = '1'
        location_f = ''.join(location_f)
        button_f = np.binary_repr(state.repr[1], width=self.floors)
        load_f = np.binary_repr(state.repr[2], width=1)
        req_f = np.binary_repr(state.repr[3], width=self.floors)
        feature_vec = location_f + button_f + load_f + req_f
        binary_feature_vec = list(map(int, feature_vec))
        return binary_feature_vec

    def getAction(self, state):
        """
            Compute the action to take in the current state.    With
            probability self.epsilon, we should take a random action and
            take the best policy action otherwise.    Note that if there are
            no legal actions, which is the case at the terminal state, you
            should choose None as the action.

            HINT: You might want to use util.flipCoin(prob)
            HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions), 'N\A'
        else:
            #Estimate the Q's values by feeding the new state through our network
            class_loss = self.sess.run(self.logits, feed_dict={self.input_state: [self.state2binaryState(state)]})
            #Obtain maxQ' and set our target value for chosen action.
            choice = np.argmax(class_loss)
            action = self.possible_actions[choice]
            return action, 'N\A'

    def update(self, state, action, nextState, reward):
        """
            The parent class calls this to observe a
            state = action => nextState and reward transition.
            You should do your Q-Value update here

            NOTE: You should never call this function,
            it will be called on your behalf
        """
        count = next(self.update_count)

        self.update_states[count] = self.state2binaryState(state)
        # self.update_actions[count] = tf.one_hot(action, depth=self.floors)
        one_hot = [0] * self.floors
        one_hot[action[0]] = 1
        self.update_actions[count] = one_hot
        self.update_rewards[count] = reward

    def final(self):
        super(PolicyGradientAgent, self).final()
        # perform one update of training
        # print("self.input_state: ", np.array(self.update_states))
        # print("self.actions: ", np.array(self.update_actions))
        r = np.expand_dims(np.array(self.update_rewards), axis=1)
        cl_l, logits = self.sess.run([self.class_loss, self.logits], feed_dict={
            self.input_state: np.array(self.update_states),
            self.actions: np.array(self.update_actions),
            self.rewards: np.expand_dims(np.array(self.update_rewards), axis=1)
        })
        # logits = self.sess.run(self.logits, feed_dict={
        #     self.input_state: np.array(self.update_states),
        #     self.actions: np.array(self.update_actions),
        #     self.rewards: np.expand_dims(np.array(self.update_rewards), axis=1)
        # })
        # print("self.logits: ", logits)
        # print("self.rewards: ", r)
        # print("self.class_loss",cl_l)
        # print("self.class_loss * self.rewards", r*cl_l)
        # print("\n")
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            self.addMetricSample('loss', float(self.sess.run(self.loss, feed_dict={
                self.input_state: np.array(self.update_states),
                self.actions: np.array(self.update_actions),
                self.rewards: np.expand_dims(np.array(self.update_rewards), axis=1)
            })))

        self.sess.run(self.train_op, feed_dict={
            self.input_state: np.array(self.update_states),
            self.actions: np.array(self.update_actions),
            self.rewards: np.expand_dims(np.array(self.update_rewards), axis=1)
        })

    def saveModel(self, confString):
        # Save the variables to disk.
        self.saver.save(self.sess, confString)
        print("Model saved in path: %s" % confString)

    def loadModel(self, modelPath):
        # Restore variables from disk.
        self.saver.restore(self.sess, modelPath)
        print("Model restored.")

