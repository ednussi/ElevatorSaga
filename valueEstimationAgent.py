import util


AGENTS_COUNT = 1


class ValueEstimationAgent(object):
    """
        Abstract agent which assigns values to (state,action)
        Q-Values for an environment. As well as a value to a
        state and a policy given respectively by,

        V(s) = max_{a in actions} Q(s,a)
        policy(s) = arg_max_{a in actions} Q(s,a)

        Both ValueIterationAgent and QLearningAgent inherit
        from this agent. While a ValueIterationAgent has
        a model of the environment via a MarkovDecisionProcess
        (see mdp.py) that is used to estimate Q-Values before
        ever actually acting, the QLearningAgent estimates
        Q-Values while acting in the environment.
    """

    def __init__(self, alpha=0.5, epsilon=0.5, gamma=0.9, numTraining = 10, numExploration=5):
        """
        Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
        alpha        - learning rate
        epsilon    - exploration rate
        gamma        - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)
        self.numExploration = int(numExploration)
        self.trained = 0
        self.metrics = {}
        self.logs = []
        global AGENTS_COUNT
        self.__id = AGENTS_COUNT
        AGENTS_COUNT += 1
        self.testMode = False

    def getConfig(self):
        return {
            'N': self.__class__.__name__,
            'A': self.alpha,
            'EPS': self.epsilon,
            'G': self.discount,
            'T': self.numTraining
        }

    def saveModel(self, confString):
        pass

    def loadModel(self, modelPath, testMode=True):
        pass

    def print(self, *args, end='\n'):
        self.logs.append(' '.join([str(a) for a in args]))
        print(*args, end=end)

    ####################################
    #     Override These Functions     #
    ####################################
    def getQValue(self, state, action):
        """
        Should return Q(state,action)
        """
        util.raiseNotDefined()

    def getValue(self, state):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

        V(s) = max_{a in actions} Q(s,a)
        """
        util.raiseNotDefined()

    def getPolicy(self, state):
        """
        What is the best action to take in the state. Note that because
        we might want to explore, this might not coincide with getAction
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        util.raiseNotDefined()

    def getAction(self, state):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        util.raiseNotDefined()

    def reset(self):
        """
        Choose an action and return it.
        """
        util.raiseNotDefined()

    def addMetricSample(self, metric, value, x=None, suffix=None):
        metric += ':' + self.__class__.__name__ + '-' + str(self.__id)
        if suffix is not None:
            metric += '-' + str(suffix)
        if metric not in self.metrics:
            self.metrics[metric] = {'x': [], 'y': []}
        if x is None:
            x = self.episodesSoFar
        self.metrics[metric]['x'].append(x)
        self.metrics[metric]['y'].append(value)

    def addHistogram(self, metric, hist, x=None):
        if metric not in self.metrics:
            self.metrics[metric] = {'x': [], 'type': 'histogram'}
        if x is None:
            x = list(range(len(hist)))
        self.metrics[metric]['x'] = x
        self.metrics[metric]['y'] = hist

    def setNumTraining(self, numTraining):
        self.numTraining = numTraining

    def setNumExploration(self, numExploration):
        self.numExploration = numExploration

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setAlpha(self, alpha):
        self.alpha = alpha

    def setDiscount(self, discount):
        self.discount = discount

    def toggleTestMode(self):
        self.testMode = not self.testMode