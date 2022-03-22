from utils import Drone, Item, Demand, NoFlyZone
import pandas as pd
import numpy as np
from string import ascii_uppercase

# Max Speed
M = 0.0
# Cost
C = 0.0
# drone count
drone_count = [0, 0, 0, 0, 0, 0]
# Warehouse Coordinates(x,y,z)
wh = []
# Recharge Stations(x,y,z)
rhg = []
# Drone params P
P = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Drone Params Q
Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Energy params A
DA = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Energy parmas B
DB = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Energy params C
DC = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Demands
demands = []
# No fly zones
noflyzones = []

itemtype_objects = []

dummy_demand = Demand(0, 0, Item(0, 0, 0, 0, 0), 0, 0, 0, 0, 0, 0, 0)
# number of no fly zones
n = 0


def get_seconds(hhmmss):
    hh = int(hhmmss.split(":")[0])
    mm = int(hhmmss.split(":")[1])
    ss = int(hhmmss.split(":")[2])
    return (hh - 8) * 3600 + mm * 60 + ss


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
            0,
        )
        drone_objects.append(drone_object)
    return drone_objects


def read_item_details(path="./data/items.xlsx"):
    items_df = pd.read_excel(path)
    for i in range(items_df.shape[0]):
        item_object = Item(
            i + 1,
            items_df["Weight (KG)"][i],
            items_df["Length"][i],
            items_df["Breadth"][i],
            items_df["Height"][i],
        )
        itemtype_objects.append(item_object)


def read_demands(demands_path="data/Scenario1/Demand.csv"):
    demand = pd.read_csv(demands_path)
    for index, row in demand.iterrows():
        warehouse = int(row["WH"][-1])
        demand_id = row["Demand ID"]
        item = itemtype_objects[int(row["Item"][-1]) - 1]
        day = row["Day"]
        x = row["X"]
        y = row["Y"]
        z = row["Z"]
        del_from = get_seconds(row["DeliveryFrom"])
        del_to = get_seconds(row["DeliveryTo"])
        failure = row["DeliveryFailure"]
        demands.append(
            Demand(warehouse, demand_id, item, day, x, y, z, del_from, del_to, failure)
        )


def check_weight_volume(drone, item):
    if item.volume <= drone.payload_volume and item.weight <= drone.payload_weight:
        return 1
    else:
        return 0


def calculate_starting_time_energy(drone, path, demand=dummy_demand):
    fraction_payload = demand.item.weight / drone.payload_weight
    max_xy_speed = M - P[drone.type - 1] * fraction_payload
    max_upward_speed = M - Q[drone.type - 1] * fraction_payload
    max_downward_speed = M + Q[drone.type - 1] * fraction_payload
    total_time = 0
    total_energy = 0
    for i in range(len(path) - 1):
        initial_pos = path[i]
        final_pos = path[i + 1]
        if final_pos[2] != initial_pos[2]:
            distance = final_pos[2] - initial_pos[2]
            if distance < 0:
                time_taken = np.ceil(abs(distance) / max_downward_speed)
                energy_consumed = (
                    drone.current_weight
                    * (DA[drone.type - 1] + DB[drone.type - 1] * max_downward_speed)
                    * (time_taken - 1)
                )
                distance_left = abs(distance) - (time_taken - 1) * max_downward_speed
                energy_consumed += drone.current_weight * (
                    DA[drone.type - 1] + DB[drone.type - 1] * distance_left
                )
            else:
                time_taken = np.ceil(distance / max_upward_speed)
                distance_left = abs(distance) - (time_taken - 1) * max_upward_speed
                energy_consumed = (
                    drone.current_weight
                    * (
                        DA[drone.type - 1]
                        + DB[drone.type - 1] * max_upward_speed
                        + DC[drone.type - 1] * max_upward_speed
                    )
                    * (time_taken - 1)
                )
                energy_consumed += drone.current_weight * (
                    DA[drone.type - 1]
                    + DB[drone.type - 1] * distance_left
                    + DC[drone.type - 1] * distance_left
                )
        else:
            distance = np.sqrt(
                (final_pos[0] - initial_pos[0]) ** 2
                + (final_pos[1] - initial_pos[1]) ** 2
            )
            time_taken = np.ceil(distance / max_xy_speed)
            distance_left = abs(distance) - (time_taken - 1) * max_xy_speed
            energy_consumed = (
                drone.current_weight
                * (DA[drone.type - 1] + DB[drone.type - 1] * max_xy_speed)
                * (time_taken - 1)
            )
            energy_consumed += drone.current_weight * (
                DA[drone.type - 1] + DB[drone.type - 1] * distance_left
            )

        total_time += time_taken
        total_energy += energy_consumed

    reaching_time = demand.del_to - 180
    starting_time = reaching_time - total_time
    return starting_time, total_energy, total_time


def check_drone_availibility(drone, timestamp):
    return not drone.check_occupy(timestamp)


def process_params(param_path="data/Scenario1/Parameters.csv"):

    parameters = pd.read_csv(param_path)

    M = parameters.loc[parameters["Parameter_ID"] == "MaxSpeed (M)", "Value"].iloc[0]
    C = parameters.loc[parameters["Parameter_ID"] == "Cost(C)", "Value"].iloc[0]

    for i in range(1, 4):
        df = parameters.loc[parameters["Parameter_ID"] == "WH" + str(i) + "X", "Value"]
        if df.empty:
            break
        vx = parameters.loc[
            parameters["Parameter_ID"] == "WH" + str(i) + "X", "Value"
        ].iloc[0]
        vy = parameters.loc[
            parameters["Parameter_ID"] == "WH" + str(i) + "Y", "Value"
        ].iloc[0]
        vz = parameters.loc[
            parameters["Parameter_ID"] == "WH" + str(i) + "Z", "Value"
        ].iloc[0]
        wh.append([vx, vy, vz])

    for i in ascii_uppercase:
        df = parameters.loc[parameters["Parameter_ID"] == i + "X1", "Value"]
        if df.empty:
            break
        vx = parameters.loc[parameters["Parameter_ID"] == i + "X1", "Value"].iloc[0]
        vy = parameters.loc[parameters["Parameter_ID"] == i + "Y1", "Value"].iloc[0]
        rhg.append([vx, vy, 0.0])

    for i in range(1, 7):
        p = parameters.loc[parameters["Parameter_ID"] == "P" + str(i), "Value"].iloc[0]
        q = parameters.loc[parameters["Parameter_ID"] == "Q" + str(i), "Value"].iloc[0]
        a = parameters.loc[parameters["Parameter_ID"] == "A" + str(i), "Value"].iloc[0]
        b = parameters.loc[parameters["Parameter_ID"] == "B" + str(i), "Value"].iloc[0]
        c = parameters.loc[parameters["Parameter_ID"] == "C" + str(i), "Value"].iloc[0]
        count = parameters.loc[
            parameters["Parameter_ID"] == "DT" + str(i) + "Count", "Value"
        ].iloc[0]
        P[i - 1] = p
        Q[i - 1] = q
        DA[i - 1] = a
        DB[i - 1] = b
        DC[i - 1] = c
        drone_count[i - 1] = count

    for idx in range(1, 3):
        zone = []
        for i in range(1, 9):
            df = parameters.loc[
                parameters["Parameter_ID"] == "X" + str(idx) + str(i), "Value"
            ]
            if df.empty:
                break
            x = parameters.loc[
                parameters["Parameter_ID"] == "X" + str(idx) + str(i), "Value"
            ].iloc[0]
            y = parameters.loc[
                parameters["Parameter_ID"] == "Y" + str(idx) + str(i), "Value"
            ].iloc[0]
            z = parameters.loc[
                parameters["Parameter_ID"] == "Z" + str(idx) + str(i), "Value"
            ].iloc[0]
            zone.append([x, y, z])
        if len(zone) > 0:
            noflyzones.append(NoFlyZone(zone))

        n = len(noflyzones)


def pathCoordinates(src, dest):
    """
    Return exact drone coornates for goinf from src to dest on straight line
    """
    pass


def getPath(src, dest):

    best_path = []
    min_cost = 1e18
    for z in range(0, 201):
        pres_cost = 0.0
        cor = src
        ok = True
        path = [src]
        if src[2] != z:
            cor = [src[0], src[1], z]
            path.append(cor)
        for i in range(0, n):
            if not noflyzones[i].doesIntersect(cor, [dest[0], dest[1], z]):
                pass
            else:
                ok = False
                break
        if not ok:
            continue
        path.append([dest[0], dest[1], z])
        if z != dest[2]:
            path.append(dest)

        # This is a valid path, check for best path

    if n == 0:
        return best_path

    if n == 1:
        if not noflyzones[0].doesIntersect([src[0], src[1], 0], [dest[0], dest[1], 0]):
            return best_path

        coordinates = [
            [noflyzones[0].mn[0], noflyzones[0].mn[1]],
            [noflyzones[0].mn[0], noflyzones[0].mx[1]],
            [noflyzones[0].mx[0], noflyzones[0].mn[1]],
            [noflyzones[0].mx[0], noflyzones[0].mx[1]],
        ]

        for c in coordinates:

            if noflyzones[0].doesIntersect(src, [c[0], c[1], src[2]]) or noflyzones[
                0
            ].doesIntersect([c[0], c[1], dest[2]], dest):
                continue

            path.append([c[0], c[1], src[2]])
            path.append([dest[0], dest[1], src[2]])

            if src[2] != dest[2]:
                path.append(dest)

            # Check for best path

        for c1 in coordinates:
            for c2 in coordinates:
                if c1 == c2:
                    continue
                if c1[0] != c2[0] and c1[1] != c2[1]:
                    continue

                if noflyzones[0].doesIntersect(
                    src, [c1[0], c1[1], src[2]]
                ) or noflyzones[0].doesIntersect([c2[0], c2[1], dest[2]], dest):
                    continue

                path.append([c1[0], c1[1], src[2]])
                path.append([c2[0], c2[1], src[2]])
                path.append([dest[0], dest[1], src[2]])

                if src[2] != dest[2]:
                    path.append(dest)

                # Check for best path

        return best_path

    # TO be complted
    """
  if n == 2:

    coordinates1=[[noflyzones[0].mn[0],noflyzones[0].mn[1]],[noflyzones[0].mn[0],noflyzones[0].mx[1]],[noflyzones[0].mx[0],noflyzones[0].mn[1]],[noflyzones[0].mx[0],noflyzones[0].mx[1]]]
    coordinates2=[[noflyzones[1].mn[0],noflyzones[1].mn[1]],[noflyzones[1].mn[0],noflyzones[1].mx[1]],[noflyzones[1].mx[0],noflyzones[1].mn[1]],[noflyzones[1].mx[0],noflyzones[1].mx[1]]]


    for c in coordinates1:

      if noflyzones[0].doesIntersect(src,[c[0],c[1],src[2]]) or noflyzones[0].doesIntersect([c[0],c[1],dest[2]],dest):
        continue

      path.append([c[0],c[1],src[2]])
      path.append([dest[0],dest[1],src[2]])
      
      if src[2] != dest[2]:
        path.append(dest)
      
      #Check for best path
  """


if __name__ == "__main__":
    read_item_details()
    process_params()
    read_demands()
    dronetype_objects = read_drone_details(max_speed=M)
    drones = []
    for i in range(len(drone_count)):
        for j in range(int(drone_count[i])):
            drone_object = Drone(
                dronetype_objects[i].type,
                dronetype_objects[i].battery_capacity,
                dronetype_objects[i].base_weight,
                dronetype_objects[i].payload_weight,
                dronetype_objects[i].payload_volume,
                dronetype_objects[i].slots,
                dronetype_objects[i].max_speed,
                j,
            )
            drones.append(drone_object)

    demands.sort(key=lambda x: x.del_to)
    for demand in demands:
        # processing each demand
        startpoint = wh[demand.wh - 1]
        endpoint = [demand.x, demand.y, demand.z]
        demand_item = demand.item
        path = getPath(startpoint, endpoint)
        return_path = getPath(endpoint, startpoint)
        for drone in drones:
            possible = check_weight_volume(drone, demand_item)
            if not possible:
                continue
            # wrong
            # drone.current_charge = drone.battery_capacity
            timestamp, energy, time_taken = calculate_starting_time_energy(
                drone, path, demand
            )
            (
                timestamp_return,
                return_energy,
                return_time_taken,
            ) = calculate_starting_time_energy(drone, return_path)
            if energy + return_energy > drone.current_charge:
                continue
            possible = check_drone_availibility(drone, timestamp)
            if not possible:
                continue
            drone.occupy_update(timestamp, demand.del_to + return_time_taken)
            demand.is_completed = True
            break

        if demand.is_completed:
            print(f"Demand {demand.demand_id} Met")
        else:
            print(f"Demand {demand.demand_id} not met.")
