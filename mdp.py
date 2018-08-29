
import itertools
import random
from typing import List, Dict

from elevator import Elevator
from floor import Floor
from state import State
from user import User


def createRandomUser():
    weight = random.randint(55, 100)
    user = User(weight, -1, -1)
    if random.random() < 1. / 40:
        user.displayType = 'child'
    elif random.random() < 0.5:
        user.displayType = 'female'
    else:
        user.displayType = 'male'
    return user


class MDP(object):

    def __init__(self,
                 num_floors,
                 capacities,
                 reward_user_in=-1.,
                 reward_user_out=-2.,
                 reward_move=-1.,
                 accl_time=0.2,
                 const_speed_time=0.3):

        self.num_floors = num_floors

        self.reward_user_in = reward_user_in
        self.reward_user_out = reward_user_out
        self.reward_move = reward_move
        self.capacities = capacities
        self.accl_time = accl_time
        self.const_speed_time = const_speed_time

        self.state = None
        self.action = None
        self.prev_state = None

    def worldToState(self, elevators: List[Elevator], floors: List[Floor], world: Dict):
        self.prev_state = self.state

        state = State(elevators, floors, world)

        self.state = state
        return state

    def actionToWorld(self, elevators: List[Elevator], floors: List[Floor], action):
        self.action = action

        for i, a in enumerate(action):
            elevator = list(filter(lambda e: e.id == i, elevators))[0]
            if len(elevator.destinationQueue):
                continue
            # elevator.now(elevator.goToFloor(elevator.currentFloor() + a, True))
            elevator.now(elevator.goToFloor(a, True))
            elevator.now(elevator.checkDestinationQueue())

        return elevators, floors

    def goUpDown(self, state):
        locs = state.repr[:-1:3]

        legal_moves = []
        for loc in locs:
            curr_elev = [0]
            if loc != 0:
                curr_elev.append(-1)
            if loc != self.num_floors - 1:
                curr_elev.append(1)

            legal_moves.append(curr_elev)

        return list(itertools.product(*legal_moves))

    def goToAnyFloor(self, state):
        legal_moves = [list(range(self.num_floors))] * len(self.capacities)

        return list(itertools.product(*legal_moves))

    def goToAnyFloorMultiAgent(self, state):
        legal_moves = [list(range(self.num_floors))] # * len(self.capacities)

        return list(itertools.product(*legal_moves))


    def updateAction(self, action):
        self.action = action

    def getReward(self, state: State, action, nextState: State):
        """
        Get the reward for the state, action, nextState transition.

        Not available in reinforcement learning.
        """
        capacity = sum([e.maxPassengerCount for e in state.elevators])
        num_users_out = sum([min(len(floor.users), capacity) for floor in state.floors])
        requests = [floor.floorNum for floor in state.floors if floor.up_requested() or floor.down_requested()]

        rewards = []
        for i, e in enumerate(state.elevators):
            el_users_in = e.getUsersIn()
            rewards.append(self.reward_user_in * el_users_in)

            users_exisiting = 0
            users = e.getUserList()
            for user in users:
                if user.destinationFloor == action[i]:
                    users_exisiting += 1
            rewards.append(users_exisiting * 2)

            if el_users_in == e.maxPassengerCount and users_exisiting == 0:
                rewards.append(-200.0)
            else:
                rewards.append(0.0)

            if el_users_in == 0 and action[i] not in requests and requests:
                rewards.append(-200.0)
            else:
                rewards.append(0.0)

            if el_users_in > 0 and action[i] not in e.pressedFloors + requests:
                rewards.append(-100.0)
            else:
                rewards.append(0.0)

            if self.action[i] != e.currentFloor:
                rewards.append(((abs(self.action[i] - e.currentFloor) - 1) * self.const_speed_time + 2 * self.accl_time) * self.reward_move)

        rewards.append(self.reward_user_out * num_users_out)

        return rewards


# class ValueIterationMDP(MDP):
#
#     def __init__(self, num_floors, capacities, spawn_factor):
#         super().__init__(num_floors, capacities)
#
#         self.spawnFactor = spawn_factor
#
#         self.allStates, self.statesMap = self.generateStatesMap()
#         self.startState = self.generateStartState()
#
#     def generateStatesMap(self):
#         elevators = {}
#         for i in range(len(self.capacities)): # for each elevator
#             if i not in elevators:
#                 elevators[i] = []
#             for loc in range(self.num_floors):  # for each possible location of this elevator
#                 for numUsers in range(self.capacities[i] + 1):  # for each possible number of users inside the elevator
#                     slots = {}
#                     for uid in range(numUsers):  # for each user in filled slots
#                         if uid not in slots:
#                             slots[uid] = []
#                         for d in range(self.num_floors):  # for each possible destination of this user
#                             if d != loc:
#                                 slots[uid].append(User(100, loc, d))
#                     if len(slots) > 0:
#                         used = []
#                         for s in itertools.product(*slots.values()):
#                             dd = sorted(list(map(lambda q: q.destinationFloor, s)))
#                             # we don't care if slot 0 goes to floor 0 and slot 1 to floor 1 or vice versa
#                             if dd not in used:
#                                 used.append(dd)
#                                 el = Elevator(eid=i,
#                                               currentFloor=loc,
#                                               maxPassengerCount=self.capacities[i],
#                                               loadFactor=0,
#                                               destinationDirection='stopped',
#                                               destinationQueue=[],
#                                               pressedFloors=[],
#                                               userSlots=[{'user': None} for _ in range(self.capacities[i])],
#                                               numFloors=self.num_floors)
#                                 for ss in s:
#                                     el.load(ss)
#                                 elevators[i].append(el)
#                     else:
#                         elevators[i].append(Elevator(eid=i,
#                                                      currentFloor=loc,
#                                                      maxPassengerCount=self.capacities[i],
#                                                      loadFactor=0,
#                                                      destinationDirection='stopped',
#                                                      destinationQueue=[],
#                                                      pressedFloors=[],
#                                                      userSlots=[{'user': None} for _ in range(self.capacities[i])],
#                                                      numFloors=self.num_floors))
#         allElevators = list(itertools.product(*elevators.values()))
#
#         maxUsersOnTheFloor = 2  # TODO: this is very limiting!
#
#         floors = {}
#         for f in range(self.num_floors):
#             if f not in floors:
#                 floors[f] = []
#             for numUsers in range(maxUsersOnTheFloor + 1):
#                 users = {}
#                 for uid in range(numUsers):
#                     if uid not in users:
#                         users[uid] = []
#                     for d in range(self.num_floors):
#                         if d != f:
#                             users[uid].append(User(100, f, d))
#                 if len(users):
#                     for u in itertools.product(*users.values()):
#                         fl = Floor(f, {'up': '', 'down': ''}, [])
#                         for uu in u:
#                             uu.appearOnFloor(fl, uu.destinationFloor)
#                         floors[f].append(fl)
#                 else:
#                     floors[f].append(Floor(f, {'up': '', 'down': ''}, []))
#         allFloors = list(itertools.product(*floors.values()))
#
#         states = []
#         for ee in allElevators:
#             for ff in allFloors:
#                 goodFloors = True
#                 for e in ee:
#                     f = ff[e.currentFloor]
#                     goodFloor = True
#                     # elevator e is on the floor f
#                     # this means (1) there cannot be passengers going to f in e
#                     for s in e.userSlots:
#                         if s['user'] is not None:
#                             if s['user'].destinationFloor == f.floorNum:
#                                 goodFloor = False
#                                 break
#
#                     # (2) people on f will enter e if there is place
#                     load = e.getUsersIn()
#                     if len(f.users) > 0 and e.maxPassengerCount - load > 0:
#                         goodFloor = False
#
#                     if not goodFloor:
#                         goodFloors = False
#                         break
#
#                 if goodFloors:
#                     states.append(State(ee, ff, {}))
#
#         statesMap = {}
#         keyCount = {}
#         for state in states:
#             for action in self.getPossibleActions(state):
#                 if (state, action) not in statesMap:
#                     statesMap[(state, action)] = []
#                     keyCount[(state, action)] = 0.
#                 statesMap[(state, action)].extend(self.applyAction(self.num_floors, self.spawnFactor, state, action))
#                 keyCount[(state, action)] += 1
#
#         # normalize probabilities
#         for k, v in statesMap.items():
#             for i, ss in enumerate(v):
#                 statesMap[k][i] = (ss[0], ss[1] / keyCount[k])
#
#         return states, statesMap
#
#     def generateStartState(self):
#         elevators = [
#             Elevator(eid=i,
#                      currentFloor=0,
#                      maxPassengerCount=self.capacities[i],
#                      loadFactor=0,
#                      destinationDirection='stopped',
#                      destinationQueue=[],
#                      pressedFloors=[],
#                      userSlots=[{'user': None} for _ in range(self.capacities[i])],
#                      numFloors=self.num_floors)
#             for i in range(len(self.capacities))
#         ]
#         floors = [Floor(f, {'up': '', 'down': ''}, []) for f in range(self.num_floors)]
#
#         return State(elevators, floors, {})
#
#     @staticmethod
#     def applyElevatorsAction(state, action):
#         newState = State(state.elevators, state.floors, state.world)
#         for i, e in enumerate(newState.elevators):
#             e.currentFloor = action[i]  # move elevator
#             for sid, u in enumerate(e.userSlots):
#                 if u['user'] is not None:
#                     if u['user'].destinationFloor == e.currentFloor:
#                         # unload passenger
#                         e.unload(sid)
#
#             floor = newState.floors[e.currentFloor]
#             for user in floor.users:
#                 if e.getUsersIn() < e.maxPassengerCount:
#                     floor.users.remove(user)
#                     floor.clear_buttons()
#                     e.load(user)
#                 else:
#                     user.pressFloorButton(floor)
#         return newState
#
#     @staticmethod
#     def applyAction(numFloors, spawnFactor, state, action):
#         statesCopy = ValueIterationMDP.applyElevatorsAction(state, action)
#
#         nextStates = util.Counter()
#         nextStates[statesCopy] = 1. - spawnFactor  # with probability 1 - spawnFactor no user will appear
#
#         for currentFloor in range(numFloors):  # spawned user can appear on any floor
#             # spawn new user
#             user = createRandomUser()
#             user.currentFloor = currentFloor
#             if currentFloor == 0:
#                 # Definitely going up
#                 destinationFloor = random.randint(1, numFloors - 1)
#                 user.destinationFloor = destinationFloor
#                 newCopy = State(statesCopy.elevators, statesCopy.floors, statesCopy.world)
#                 user.appearOnFloor(newCopy.floors[currentFloor], destinationFloor)
#                 if newCopy in nextStates:
#                     nextStates[newCopy] += 0.5 * spawnFactor
#                 else:
#                     nextStates[newCopy] = 0.5 * spawnFactor
#             else:
#                 # Usually going down, but sometimes not
#                 for destinationFloor in [df for df in range(numFloors) if df != currentFloor]:
#                     user.destinationFloor = destinationFloor
#                     newCopy = State(statesCopy.elevators, statesCopy.floors, statesCopy.world)
#                     user.appearOnFloor(newCopy.floors[currentFloor], destinationFloor)
#                     if newCopy in nextStates:
#                         nextStates[newCopy] += 0.5 * spawnFactor * 1. / numFloors
#                     else:
#                         nextStates[newCopy] = 0.5 * spawnFactor * 1. / numFloors
#
#         nextStates.normalize()
#
#         return list(nextStates.items())
#
#
#     def getStates(self):
#         """
#         Return a list of all states in the MDP.
#         Not generally possible for large MDPs.
#         """
#         return self.allStates
#
#     def getStartState(self):
#         """
#         Return the start state of the MDP.
#         """
#         return self.startState
#
#     def getPossibleActions(self, state):
#         """
#         Return list of possible actions from 'state'.
#         """
#         return self.goToAnyFloor(state)
#
#     def getTransitionStatesAndProbs(self, state, action):
#         """
#         Returns list of (nextState, prob) pairs
#         representing the states reachable
#         from 'state' by taking 'action' along
#         with their transition probabilities.
#
#         Note that in Q-Learning and reinforcement
#         learning in general, we do not know these
#         probabilities nor do we directly model them.
#         """
#         return self.statesMap[(state, action)]
#
#     def getReward(self, state: State, action, nextState: State):
#         """
#         Get the reward for the state, action, nextState transition.
#
#         Not available in reinforcement learning.
#         """
#         num_users_out = sum([len(floor.users) for floor in state.floors])
#
#         num_users_in = 0
#         time_reward = 0
#         for i, e in enumerate(state.elevators):
#             num_users_in += e.getUsersIn()
#             if action[i] != e.currentFloor:
#                 time_reward += ((abs(
#                     action[i] - e.currentFloor) - 1) * self.const_speed_time + 2 * self.accl_time) * -0.1
#
#         return self.reward_user_in * num_users_in + self.reward_user_out * num_users_out + time_reward
#
#     def isTerminal(self, state: State):
#         """
#         Returns true if the current state is a terminal state.  By convention,
#         a terminal state has zero future rewards.  Sometimes the terminal state(s)
#         may have no possible actions.  It is also common to think of the terminal
#         state as having a self-loop action 'pass' with zero reward; the formulations
#         are equivalent.
#         """
#         return False
#
#     # @staticmethod
#     # def numAllStates(numFloors, numElevators, floorCapacity, elevatorCapacity):
#     #
#     #     return numFloors ** numElevators * \
#     #            sum([comb(numFloors - 2 + k, k, exact=True) for k in range(elevatorCapacity + 1)]) ** numElevators * \
#     #            sum([comb(numFloors - 2 + l, l, exact=True) for l in range(floorCapacity + 1)]) ** numFloors