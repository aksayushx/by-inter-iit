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


def read_data(demands_path="Demand.csv", param_path="Parameters.csv"):
    demand = pd.read_csv(demands_path)
    parameters = pd.read_csv(param_path)
    print(demand.head())
    print(parameters.head())
