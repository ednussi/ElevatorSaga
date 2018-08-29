from typing import List, Dict

import numpy as np

from elevator import Elevator
from floor import Floor


class State(object):
    def __init__(self, elevators: List[Elevator], floors: List[Floor], world: Dict):
        self.elevators = [e.__copy__() for e in elevators]
        self.floors = [f.__copy__() for f in floors]
        self.world = world
        self.repr = State.representation(elevators, floors, world)

    @staticmethod
    def representation(elevators: List[Elevator], floors: List[Floor], world: Dict):
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

        state.append(np.power(2, list(set(up_pressed + down_pressed))).sum())

        return tuple(state)

    def __eq__(self, other):
        return self.repr == other.repr

    def __ne__(self, other):
        return self.repr != other.repr

    def __hash__(self):
        return self.repr.__hash__()

    def __str__(self):
        return self.repr.__str__()

    def __repr__(self):
        res = ''
        for ii in reversed(range(len(self.floors))):
            res += str(self.floors[ii])
            for e in filter(lambda ee: ee.currentFloor == ii, self.elevators):
                res += str(e)
                res += '// '
            res += '\n'
        return res

