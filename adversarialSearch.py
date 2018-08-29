import itertools
#import graphviz

import numpy as np
import util
from mdp import createRandomUser
from state import State
from valueEstimationAgent import ValueEstimationAgent


def getLegalElevatorsActions(state):
    legal_moves = [list(range(len(state.floors)))] * len(state.elevators)

    # hasUsers = sum(map(lambda f: len(f.users), state.floors), 0)
    # for a in itertools.product(*legal_moves):
    #     for i, e in enumerate(state.elevators):
    #         if not (e.currentFloor == a[i] and hasUsers > 0):
    #             yield a
    return itertools.product(*legal_moves)


def applyElevatorsAction(state, action):
        newState = State(state.elevators, state.floors, state.world)
        for i, e in enumerate(newState.elevators):
            e.currentFloor = action[i]  # move elevator
            for sid, u in enumerate(e.userSlots):
                if u['user'] is not None:
                    if u['user'].destinationFloor == e.currentFloor:
                        # unload passenger
                        e.unload(sid)
                    else:
                        u['user'].waitTime += 0.1

            floor = newState.floors[e.currentFloor]
            for user in floor.users[:]:
                if e.getUsersIn() < e.maxPassengerCount:
                    floor.users.remove(user)
                    floor.clear_buttons()
                    e.load(user)
                else:
                    user.pressFloorButton(floor)
        for f in state.floors:
            for u in f.users:
                u.waitTime += 0.1
        return newState


def getLegalWorldActions(state):
    return [(None, None)] + [(f.floorNum, dest.floorNum) for f in state.floors for dest in state.floors if dest.floorNum != f.floorNum]


def applyWorldAction(state, action):
    newState = State(state.elevators, state.floors, state.world)
    # for e in newState.elevators:
    #     for u in e.userSlots:
    #         if u['user'] is not None:
    #             u['user'].waitTime += 0.1
    # for f in newState.floors:
    #     for u in f.users:
    #         u.waitTime += 0.1
    if action[0] is None:
        return newState
    user = createRandomUser()
    user.appearOnFloor(newState.floors[action[0]], action[1])
    for e in newState.elevators:
        if e.currentFloor == action[0] and e.getUsersIn() < e.maxPassengerCount:
            newState.floors[action[0]].users.remove(user)
            e.load(user)
            if user.destinationFloor - user.currentFloor > 0:
                newState.floors[action[0]].buttonStates['up'] = ''
            else:
                newState.floors[action[0]].buttonStates['down'] = ''
            break
    return newState


class ReflexAgent(ValueEstimationAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, state):
        """
        You do not need to change this method, but you're welcome to.
        get_action chooses among the best options according to the evaluation function.
        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = list(getLegalElevatorsActions(state))

        # Choose one of the best actions
        scores = [self.evaluation_function(state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index], 'N/A'

    def evaluation_function(self, state, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.
        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = applyElevatorsAction(state, action)
        return simpleEvaluationFunction(successor_game_state)


class MultiAgentSearchAgent(ValueEstimationAgent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, simulator, evaluationFn, depth=2):
        super().__init__()
        self.sim = simulator
        self.evaluationFn = util.lookup(evaluationFn, globals())
        self.depth = depth

        self.qValues = {}
        self.accumReward = 0.
        self.episodesSoFar = 0

    def getConfig(self):
        return {
            'D': self.depth
        }

    def update(self, state, action, reward, nextState):
        pass

    def startEpisode(self):
        pass

    def final(self):
        self.addMetricSample('reward', self.accumReward / 300)
        self.accumReward = 0.
        self.episodesSoFar += 1
        self.trained = self.episodesSoFar

    # noinspection PyUnusedLocal
    def observeTransition(self, state, action, nextState, rewards):
        self.accumReward += sum(rewards, 0)



class MinimaxAgent(MultiAgentSearchAgent):
    def minimax_search(self, state):
        actions = list(getLegalElevatorsActions(state))
        action = actions[0]
        value = -np.inf
        for a in actions:
            v = self.min_value(applyElevatorsAction(state, a), 0)
            if v > value:
                action = a
                value = v

        return action

    def min_value(self, state, depth):
        if depth >= self.depth:  # or no legal actions
            return self.evaluationFn(state)
        v = np.inf
        actions = getLegalWorldActions(state)
        for a in actions:
            v = min(v, self.max_value(applyWorldAction(state, a), depth + 1))
        return v

    def max_value(self, state, depth):
        if depth >= self.depth:  # or no legal actions
            return self.evaluationFn(state)
        v = -np.inf
        actions = list(getLegalElevatorsActions(state))
        for a in actions:
            v = max(v, self.min_value(applyElevatorsAction(state, a), depth))
        return v

    def getAction(self, state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        return self.minimax_search(state), 'N/A'


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def __init__(self, simulator, evaluationFn, depth=2):
        super().__init__(simulator, evaluationFn, depth=depth)
        self.expanded = 0

    def alpha_beta_search(self, state):
        self.expanded = 0
        value, action = self.max_value(state, -np.inf, np.inf, 0)
        # print(self.expanded)
        return action

    def min_value(self, state, alpha, beta, depth):
        self.expanded += 1
        if depth >= self.depth:  # or no legal actions
            v = self.evaluationFn(state)
            return v, None
        v = np.inf
        actions = getLegalWorldActions(state)
        a = actions[0] if len(actions) else None

        for action in actions:
            newv, newa = self.max_value(applyWorldAction(state, action),
                                        alpha, beta, depth + 1)
            if newv < v:
                v, a = newv, action
            if v <= alpha:
                return v, action
            if v <= beta:
                beta = v
                a = action
        return v, a

    def max_value(self, state, alpha, beta, depth):
        self.expanded += 1
        # print(state.__repr__(), 'depth: ', depth, 'value: ', self.evaluationFn(state))
        if depth >= self.depth:  # or no legal actions
            v = self.evaluationFn(state)
            # print('value', v)
            return v, None
        v = -np.inf
        actions = list(getLegalElevatorsActions(state))
        a = actions[0] if len(actions) else None

        for action in actions:
            newv, newa = self.min_value(applyElevatorsAction(state, action),
                                        alpha, beta, depth)
            if newv > v:
                v, a = newv, action
            if v >= beta:
                return v, action
            if alpha < v:
                alpha = v
                a = action
        return v, a

    def getAction(self, state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alpha_beta_search(state), 'N/A'


counter = 0


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def __init__(self, simulator, evaluationFn, depth=2):
        super().__init__(simulator, evaluationFn, depth=depth)
        self.expanded = 0

    def expectimax_search(self, state):
        self.expanded = 0
        actions = list(getLegalElevatorsActions(state))
        action = actions[0]
        value = -np.inf
        for a in actions:
            v = self.exp_value(applyElevatorsAction(state, a), 0)
            if v > value:
                action = a
                value = v
        return action

    def exp_value(self, state, depth):
        self.expanded += 1
        if depth >= self.depth:  # or no legal actions
            return self.evaluationFn(state)
        actions = getLegalWorldActions(state)
        values = [self.max_value(applyWorldAction(state, a), depth + 1)
                  for a in actions]
        weights = [0.5] + [0.5 / len(actions)] * (len(values) - 1)
        return np.dot(values, weights)

    def max_value(self, state, depth):
        self.expanded += 1
        # print(state.__repr__(), 'depth: ', depth, 'value: ', self.evaluationFn(state))
        if depth >= self.depth:  # or no legal actions
            return self.evaluationFn(state)
        v = -np.inf
        for a in list(getLegalElevatorsActions(state)):
            v = max(v, self.exp_value(applyElevatorsAction(state, a), depth))
        return v

    def getAction(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        action = self.expectimax_search(game_state)
        # print('Expanded:', self.expanded)
        return action, 'N/A'


def simpleEvaluationFunction(state: State):
    v = 0.
    for f in state.floors:
        for u in f.users:
            v -= 1.1  # * weight
    for e in state.elevators:
        for u in e.userSlots:
            if u['user'] is not None:
                v -= 1.
    return v