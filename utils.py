from cv2 import inRange
import pandas as pd
import numpy as np
from string import ascii_uppercase


class Drone:
    def __init__(
        self,
        type,
        battery_capacity,
        base_weight,
        payload_weight,
        payload_volume,
        slots,
        max_speed,
        id,
    ):

        self.type = type
        self.battery_capacity = battery_capacity
        self.base_weight = base_weight
        self.payload_weight = payload_weight
        self.payload_volume = payload_volume
        self.slots = slots
        self.max_speed = max_speed
        self.id = 'D' + str(self.type) + ascii_uppercase[id]

        self.current_charge = battery_capacity
        self.current_weight = base_weight
        self.current_volume = 0
        self.available_slots = slots
        self.current_speed = 0
        self.x = 0
        self.y = 0
        self.z = 0

        self.occupied_slots = []

    def occupy_update(self, starting_time, ending_time):

        self.occupied_slots.append([starting_time, ending_time])

    def check_occupy(self, timestamp):

        for i in range(len(self.occupied_slots)):
            occupied = (
                1
                if (
                    timestamp >= self.occupied_slots[i][0]
                    and timestamp <= self.occupied_slots[i][1]
                )
                else 0
            )
            if occupied:
                return 1

        return 0


class Item:
    def __init__(self, id, weight, l, b, h):
        self.id = id
        self.weight = weight
        self.l = l
        self.b = b
        self.h = h

        self.volume = self.l * self.b * self.h


class Demand:
    def __init__(self, wh, demand_id, item, day, x, y, z, del_from, del_to, failure):

        self.wh = wh
        self.demand_id = demand_id
        self.item = item
        self.day = day
        self.x = x
        self.y = y
        self.z = z
        self.del_from = del_from
        self.del_to = del_to
        self.failure = failure
        self.is_completed = False


class NoFlyZone:
    def __init__(self, points) -> None:
        self.points = points
        self.mn = np.min(np.array(points), axis=0)
        self.mx = np.max(np.array(points), axis=0)
        print(self.mx)
        print(self.mn)

    def inRange(self, a, b, c):
        return c >= a and c <= b

    def doesIntersect(self, a, b):
        if a[0] == b[0]:
            return inRange(self.mn[0], self.mx[0], a[0])
        if a[1] == b[1]:
            return inRange(self.mn[1], self.mn[1], a[1])

        # Put x
        y = a[1] + ((b[1] - a[1]) * (self.mn[0] - a[0])) / (b[0] - a[0])
        if inRange(self.mn[1], self.mx[1], y):
            return True

        y = a[1] + ((b[1] - a[1]) * (self.mx[0] - a[0])) / (b[0] - a[0])
        if inRange(self.mn[1], self.mx[1], y):
            return True

        # Put y
        x = a[0] + ((b[0] - a[0]) * (self.mn[1] - a[1])) / (b[1] - a[1])
        if inRange(self.mn[0], self.mx[0], x):
            return True

        x = a[0] + ((b[0] - a[0]) * (self.mx[1] - a[1])) / (b[1] - a[1])
        if inRange(self.mn[0], self.mx[0], x):
            return True

        return False
