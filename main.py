import pandas as pd
import numpy as np
from utils import Drone, Item


def read_drone_details(path="./data/drone.xlsx", max_speed=10):
    drone_df = pd.read_excel(path)
    drone_objects = []
    for i in range(drone_df.shape[0]):
        drone_object = Drone(
            i + 1,
            drone_df["Battery Capacity\n(mAh)"][i],
            drone_df["Base Weight (kg)"][i],
            drone_df["Payload Capacity (KG)"][i],
            drone_df["Payload Capacity (cu.cm)"][i],
            drone_df["Max Slots"],
            max_speed,
        )
        drone_objects.append(drone_object)
    return drone_objects


def read_item_details(path="./data/items.xlsx"):
    items_df = pd.read_excel(path)
    item_objects = {}
    for i in range(items_df.shape[0]):
        item_object = Item(
            items_df["Item Id"][i],
            items_df["Weight (KG)"][i],
            items_df["Length"][i],
            items_df["Breadth"][i],
            items_df["Height"][i],
        )
        item_objects[items_df["Item Id"][i]] = item_object
    return item_objects


def read_data(demands_path="Demand.csv"):
    demand = pd.read_csv(demands_path)
    print(demand.head())


if __name__ == "__main__":
    # max_speed and drone counts will come from parameters
    """
    dronetype_objects = read_drone_details(max_speed=10)
    drone_counts = [5, 6, 2, 3, 4, 1]
    drones = []
    for i in range(len(drone_counts)):
        for j in range(drone_counts[i]):
            drone_object = Drone(
                dronetype_objects[i].type,
                dronetype_objects[i].battery_capacity,
                dronetype_objects[i].base_weight,
                dronetype_objects[i].payload_weight,
                dronetype_objects[i].payload_volume,
                dronetype_objects[i].slots,
                dronetype_objects[i].max_speed,
            )
            drones.append(drone_object)
    print(len(drones))
    """
    itemtype_objects = read_item_details()
