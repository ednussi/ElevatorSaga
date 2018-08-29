class ElevatorEvents:
    idle = 'idle'
    floor_button_pressed = 'floor_button_pressed'
    passing_floor = 'passing_floor'
    stopped_at_floor = 'stopped_at_floor'


class Elevator:
    def __init__(self,
                 eid=0,
                 currentFloor=0,
                 maxPassengerCount=4,
                 loadFactor=0,
                 destinationDirection=None,
                 destinationQueue=(),
                 pressedFloors=(),
                 userSlots=None,
                 numFloors=3):
        self.id = eid
        self.actions = []
        self.eventHandles = {}

        self.currentFloor = currentFloor
        self.maxPassengerCount = maxPassengerCount
        self.loadFactor = loadFactor
        self.destinationDirection = destinationDirection
        self.destinationQueue = list(destinationQueue)
        self.pressedFloors = list(pressedFloors)
        if userSlots is not None:
            self.userSlots = userSlots
        else:
            self.userSlots = [{'user': None} for _ in range(self.maxPassengerCount)]
        self.numFloors = numFloors

    def getUserList(self):
        users = []
        for u in self.userSlots:
            if u['user'] is not None:
                users.append(u['user'])

        return users

    def getUsersIn(self):
        return len(list(filter(lambda u: u is not None and u['user'] is not None, self.userSlots)))

    def unload(self, slotId):
        self.userSlots[slotId]['user'] = None
        if self.currentFloor in self.pressedFloors:
            self.pressedFloors.remove(self.currentFloor)

    def load(self, user):
        if user is not None:
            self.pressFloorButton(user.destinationFloor)
            for i, s in enumerate(self.userSlots):
                if s['user'] is None:
                    self.userSlots[i]['user'] = user
                    break

    def pressFloorButton(self, floorNum):
        if floorNum not in self.pressedFloors:
            self.pressedFloors.append(floorNum)

    # enqueueable actions
    def checkDestinationQueue(self):
        return 'checkDestinationQueue()'

    def goToFloor(self, floor, now=False):
        return 'goToFloor({}, {})'.format(floor, 'true' if now else 'false')

    def stop(self):
        return 'stop()'

    # conditions
    def ifFloor(self, floorNum, handler):
        return 'if (floorNum === {}) \{{}\}'.format(floorNum, handler)

    # enqueuers
    def on(self, event, handler):
        if event not in self.eventHandles:
            self.eventHandles[event] = []
        self.eventHandles[event].append(handler)

    def now(self, handler):
        self.actions.append(handler)

    def __str__(self):
        a = "{} [{}] | {} |".format(self.id, self.currentFloor, ''.join(list(map(lambda x: str(x) if x in self.pressedFloors else '_', range(self.numFloors)))))
        a += " {}".format(''.join(list(map(lambda x: str(x['user'].destinationFloor) if x['user'] is not None else '_', self.userSlots))))
        return a

    def __copy__(self):
        return Elevator(
            eid=self.id,
            currentFloor=self.currentFloor,
            maxPassengerCount=self.maxPassengerCount,
            loadFactor=self.loadFactor,
            destinationDirection=None,
            destinationQueue=[],
            pressedFloors=self.pressedFloors[:],
            userSlots=[{'user':u['user']} if u['user'] is not None else {'user': None} for u in self.userSlots],
            numFloors=self.numFloors
        )