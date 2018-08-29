import os
import random

import util
from elevator import Elevator
from floor import Floor
from mdp import MDP
from user import User


class Simulator(object):

    def __init__(self,
                 num_elevators=1,
                 num_floors=3,
                 num_users_per_episode=100,
                 num_iterations_per_episode=300,
                 spawn_factor=0.4,
                 spawn_factor_deviation=0.25,
                 max_wait_time=10,
                 max_passenger_per_elevator=4,
                 reward_user_in=-1.,
                 reward_user_out=-2.,
                 reward_move=-1.):

        self.logs = []

        self.reward_user_in = reward_user_in
        self.reward_user_out = reward_user_out
        self.reward_move = reward_move

        self.rewIn = 0
        self.rewOut = 0
        self.rewMov = 0
        self.print('Creating simulator with {} elevators, {} floors'.format(num_elevators, num_floors))

        self.num_elevators = num_elevators
        self.num_floors = num_floors
        self.num_users_per_episode = num_users_per_episode
        self.iter_per_episode = num_iterations_per_episode
        self.spawn_factor = spawn_factor
        self.current_spawn_factor = spawn_factor
        self.spawn_factor_deviation = spawn_factor_deviation
        self.max_wait_time = max_wait_time
        self.capacities = [max_passenger_per_elevator for _ in range(num_elevators)]
        self.mdp = MDP(num_floors, self.capacities,
                       reward_user_in=reward_user_in, reward_user_out=reward_user_out, reward_move=reward_move)
        self.vimdp = None
        self.max_passenger_per_elevator = max_passenger_per_elevator
        self.average_wait_time = 0
        self.throughput = 0
        self.elevators = []
        self.floors = []
        self.world = {}
        self.transported = 0
        self.avgWaitTime = 0.
        self.moves = 0
        self.distance = 0
        self.total_wait_time = 0
        self.shouldStopTraining = False

    def print(self, *args, end='\n'):
        self.logs.append(' '.join([str(a) for a in args]))
        print(*args, end=end)

    def getConfig(self):
        return {
            'E': self.num_elevators,
            'F': self.num_floors,
            'C': self.capacities,
            'IPE': self.iter_per_episode,
            'SF': self.spawn_factor,
            'RI': self.reward_user_in,
            'RO': self.reward_user_out,
            'RM': self.reward_move,
        }

    def update_wait_time(self, user: User):
        self.total_wait_time += user.waitTime

    def restart(self):
        self.elevators = []
        for i in range(self.num_elevators):
            self.elevators.append(Elevator(eid=i,
                                           currentFloor=0,
                                           maxPassengerCount=self.max_passenger_per_elevator,
                                           loadFactor=0,
                                           destinationDirection='stopped',
                                           destinationQueue=[],
                                           pressedFloors=[],
                                           userSlots=[{'user': None} for _ in range(self.max_passenger_per_elevator)],
                                           numFloors=self.num_floors))

        self.floors = []
        for i in range(self.num_floors):
            self.floors.append(Floor(floorNum=i,
                                     buttonStates={'up': '', 'down': ''},
                                     users=[]))

        self.current_spawn_factor = self.spawn_factor + random.random() * self.spawn_factor_deviation

        self.world = {}
        self.transported = 0
        self.total_wait_time = 0
        self.moves = 0
        self.distance = 0
        self.rewMov = self.rewIn = self.rewOut = 0

    def train(self, agent):
        # save config before training, as it may change in the process
        agentConfig = agent.getConfig()
        simConfig = self.getConfig()

        for i in range(agent.numTraining):
            if self.shouldStopTraining:
                self.print('Stopped training!')
                self.shouldStopTraining = False
                break
            self.runEpisode(agent)
            agent.final()

            if i % 100 == 0:
                self.print_stats()

                agent.addMetricSample('avgWaitTime', self.total_wait_time / self.transported if self.transported > 0 else 0)

                agent.toggleTestMode()
                self.runEpisode(agent)
                agent.stopEpisode()
                agent.addMetricSample('TestReward', agent.episodeRewards)
                agent.toggleTestMode()

        self.saveModel(agent, agentConfig, simConfig)

    # noinspection PyMethodMayBeStatic
    def saveModel(self, agent, agentConfig, simConfig):
        configStr = '_'.join(['{}={}'.format(k, v) for k, v in sorted(simConfig.items())]) + '_' + \
            '_'.join(['{}={}'.format(k, v) for k, v in sorted(agentConfig.items())]) #+ '_' + 'TT=' + datetime.datetime.utcnow().isoformat().replace(':', '-')

        agent.saveModel(os.path.join(os.getcwd(), 'models', agent.__class__.__name__, configStr))

    @staticmethod
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

    @staticmethod
    def getRandomFloor(numFloors):
        return 0 if util.flipCoin(0.5) else random.randint(0, numFloors - 1)

    @staticmethod
    def spawnUserRandomly(numFloors, currentFloor, floorsContainer):
        user = Simulator.createRandomUser()

        if currentFloor == 0:
            # Definitely going up
            destinationFloor = random.randint(1, numFloors - 1)
        else:
            # Usually going down, but sometimes not
            if random.random() < 0.1:
                destinationFloor = (currentFloor + random.randint(1, numFloors - 1)) % numFloors
            else:
                destinationFloor = 0

        user.currentFloor = currentFloor
        user.destinationFloor = destinationFloor
        user.appearOnFloor(floorsContainer[currentFloor], destinationFloor)
        floorsContainer[currentFloor].users = list(set(floorsContainer[currentFloor].users))

    def runEpisode(self, agent):
        self.restart()
        agent.startEpisode()

        i = 0
        state = self.mdp.worldToState(self.elevators, self.floors, self.world)
        action = agent.getAction(state)[0]
        self.take_action(action)

        while i < self.iter_per_episode: # and self.transported < self.num_users_per_episode:
            if util.flipCoin(self.current_spawn_factor):
                self.spawnUserRandomly(self.num_floors, Simulator.getRandomFloor(self.num_floors), self.floors)

            state = self.mdp.worldToState(self.elevators, self.floors, self.world)
            rewards = self.mdp.getReward(self.mdp.prev_state, self.mdp.action, state)

            agent.observeTransition(self.mdp.prev_state, self.mdp.action, state, rewards)
            action = agent.getAction(state)[0]
            self.take_action(action)
            i += 1

        self.throughput = self.transported / self.moves if self.moves > 0 else 0
        self.avgWaitTime = self.total_wait_time / self.transported if self.transported > 0 else 0

    def take_action(self, action):
        self.moves += 1
        self.mdp.updateAction(action)
        for i, e in enumerate(self.elevators):
            e.currentFloor = action[i] # move elevator
            for sid, u in enumerate(e.userSlots):
                if u['user'] is not None:
                    if u['user'].destinationFloor == e.currentFloor:
                        # unload passenger
                        self.update_wait_time(u['user'])
                        e.unload(sid)
                        self.transported += 1 # update total users moved
                    else:
                        u['user'].increaseWaitTime()

            floor = self.floors[e.currentFloor]
            for user in floor.users[:]:
                if e.getUsersIn() < e.maxPassengerCount:
                    floor.users.remove(user)
                    floor.clear_buttons()
                    e.load(user)
                else:
                    user.pressFloorButton(floor)

                # Increase the wait time for each of the users
                user.increaseWaitTime()


    def print_stats(self):
        self.print('\tThroughput', self.transported / self.moves if self.moves > 0 else 0)
        self.print('\tAvgWaitTime', self.total_wait_time / self.transported if self.transported > 0 else 0)

    def __str__(self):
        res = ''
        for ii in reversed(range(len(self.floors))):
            res += str(self.floors[ii])
            for e in filter(lambda ee: ee.currentFloor == ii, self.elevators):
                res += str(e)
                res += '// '
            res += '\n'
        return res
