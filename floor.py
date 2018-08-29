class FloorEvents:
    up_button_pressed = 'up_button_pressed'
    down_button_pressed = 'down_button_pressed'


class Floor:
    def __init__(self, floorNum=0, buttonStates=None, users=()):
        if buttonStates is None:
            buttonStates = {'up':'', 'down':''}
        self.floorNum = floorNum
        self.buttonStates = buttonStates
        self.eventHandles = {}
        self.users = list(users)

    def up_requested(self):
        return self.buttonStates['up'] != ''

    def down_requested(self):
        return self.buttonStates['down'] != ''

    def on(self, event, handler):
        if event not in self.eventHandles:
            self.eventHandles[event] = []
        self.eventHandles[event].append(handler)

    def clear_buttons(self):
        self.buttonStates['up'] = ''
        self.buttonStates['down'] = ''

    def __str__(self):
        a = "{} | {} {} |".format(self.floorNum, 'up' if self.up_requested() else '  ', 'dn' if self.down_requested() else '  ')
        a += " {:<7} >>> ".format(''.join(list(map(lambda x: str(x.destinationFloor) if x is not None else '_', self.users))))
        return a

    def __copy__(self):
        return Floor(
            floorNum=self.floorNum,
            buttonStates=self.buttonStates.copy(),
            users=self.users.copy()
        )