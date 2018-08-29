

class User(object):
    def __init__(self, weight=100, currentFloor=0, destinationFloor=2, waitTime=0):
        self.weight = weight
        self.currentFloor = currentFloor
        self.destinationFloor = destinationFloor
        self.done = False
        self.removeMe = False
        self.waitTime = waitTime

    def appearOnFloor(self, floor, destinationFloorNum):
        self.currentFloor = floor.floorNum
        self.destinationFloor = destinationFloorNum
        self.pressFloorButton(floor)
        floor.users.append(self)

    def pressFloorButton(self, floor):
        if self.destinationFloor < self.currentFloor:
            floor.buttonStates['down'] = 'activated'
        else:
            floor.buttonStates['up'] = 'activated'

    def increaseWaitTime(self):
        self.waitTime += 1

    def __str__(self):
        a = "({} -> {})".format(self.currentFloor, self.destinationFloor)
        return a

    def __eq__(self, other):
        return self.currentFloor == other.currentFloor and \
               self.destinationFloor == other.destinationFloor and \
               self.weight == other.weight

    def __hash__(self):
        return self.currentFloor.__hash__() * 17 + self.destinationFloor.__hash__() * 19