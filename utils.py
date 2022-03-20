import pandas as pd
import numpy as np


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
    ):

        self.type = type
        self.battery_capacity = battery_capacity
        self.base_weight = base_weight
        self.payload_weight = payload_weight
        self.payload_volume = payload_volume
        self.slots = slots
        self.max_speed = max_speed

        self.current_charge = battery_capacity
        self.current_weight = base_weight
        self.current_volume = 0
        self.available_slots = slots
        self.current_speed = 0
        self.x = 0
        self.y = 0
        self.z = 0


class Item:
    def __init__(self, id, weight, l, b, h):
        self.id = id
        self.weight = weight
        self.l = l
        self.b = b
        self.h = h

        self.volume = self.l * self.b * self.h



class Demand:
    def __init__(self,wh,demand_id,item,day,x,y,z,del_from,del_to,failure):

        self.wh=wh
        self.demand_id=demand_id
        self.item=item
        self.day=day
        self.x=x
        self.y=y
        self.z=z
        self.del_from=del_from
        self.del_to=del_to
        self.failure=failure





