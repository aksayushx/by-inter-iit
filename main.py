from utils import Drone, Item, Demand, NoFlyZone
import pandas as pd
import numpy as np
from string import ascii_uppercase
from copy import deepcopy
import sys
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
    cost_df = pd.read_excel('./data/cost_components.xlsx')
    drone_objects = []
    print(cost_df.columns)

    for i in range(drone_df.shape[0]):
        drone_object = Drone(
            i + 1,
            drone_df["Battery Capacity\n(mAh)"][i],
            drone_df["Base Weight (kg)"][i],
            drone_df["Payload Capacity (KG)"][i],
            drone_df["Payload Capacity (cu.cm)"][i],
            drone_df["Max Slots"][i],
            max_speed,
            0,
            cost_df["Maintenance Fixed Cost (per day)"][i],
            cost_df["Maintenance Variable Cost (per hour of flight time)"][i],
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

if sys.argv[1]=='1':
    demand_file=f"data/Scenario{sys.argv[1]}/Demand.csv"
else:
    demand_file=f"data/Scenario{sys.argv[1]}/Demand_Day{sys.argv[1]}.csv"

def read_demands(demands_path=demand_file):

    global demands
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
    fraction_payload = (demand.item.weight + demand.extra_weight) / drone.payload_weight
    max_xy_speed = drone.max_speed - P[drone.type - 1] * fraction_payload
    max_upward_speed = drone.max_speed - Q[drone.type - 1] * fraction_payload
    max_downward_speed = drone.max_speed + Q[drone.type - 1] * fraction_payload
    if max_xy_speed == 0:
        print(drone.max_speed)
        print(P[drone.type - 1])
        print(fraction_payload)
        print("yay")
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

def output_path(path2,drone,demand):
    fraction_payload = demand.item.weight / drone.payload_weight
    max_xy_speed = drone.max_speed - P[drone.type - 1] * fraction_payload
    max_upward_speed = drone.max_speed - Q[drone.type - 1] * fraction_payload
    max_downward_speed = drone.max_speed + Q[drone.type - 1] * fraction_payload
    total_time = 0
    total_energy = 0
    timewise_path = [deepcopy(path[0])]
    timewise_energy = [0]
    timewise_speed = [0]
    for i in range(len(path) - 1):
        initial_pos = deepcopy(path[i])
        final_pos = deepcopy(path[i + 1])
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
                curr_position = deepcopy(initial_pos)
                for j in range(int(time_taken) - 1):
                    curr_position[2] -= max_downward_speed
                    timewise_path.append(deepcopy(curr_position))
                    timewise_energy.append(
                        drone.current_weight
                        * (DA[drone.type - 1] + DB[drone.type - 1] * max_downward_speed)
                    )
                    timewise_speed.append(deepcopy(max_downward_speed))
                timewise_path.append(final_pos)
                timewise_energy.append(
                    drone.current_weight
                    * (DA[drone.type - 1] + DB[drone.type - 1] * distance_left)
                )
                timewise_speed.append(deepcopy(distance_left))
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
                curr_position = initial_pos
                for j in range(int(time_taken) - 1):
                    curr_position[2] += max_upward_speed
                    timewise_path.append(deepcopy(curr_position))
                    timewise_energy.append(
                        drone.current_weight
                        * (
                            DA[drone.type - 1]
                            + DB[drone.type - 1] * max_upward_speed
                            + DC[drone.type - 1] * max_upward_speed
                        )
                    )
                    timewise_speed.append(deepcopy(max_upward_speed))
                timewise_path.append(deepcopy(final_pos))
                timewise_energy.append(
                    drone.current_weight
                    * (
                        DA[drone.type - 1]
                        + DB[drone.type - 1] * distance_left
                        + DC[drone.type - 1] * distance_left
                    )
                )
                timewise_speed.append(distance_left)
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
            curr_position = initial_pos
            y_dist = final_pos[1] - initial_pos[1]
            x_dist = final_pos[0] - initial_pos[0]
            hyp = np.sqrt((x_dist) ** 2 + y_dist ** 2)
            cos_theta = x_dist / hyp
            sin_theta = y_dist / hyp
            for j in range(int(time_taken) - 1):
                curr_position[0] += max_xy_speed * cos_theta
                curr_position[1] += max_xy_speed * sin_theta
                timewise_path.append(deepcopy(curr_position))
                timewise_energy.append(
                    drone.current_weight
                    * (DA[drone.type - 1] + DB[drone.type - 1] * max_xy_speed)
                )
                timewise_speed.append(deepcopy(max_xy_speed))
            timewise_path.append(deepcopy(final_pos))
            timewise_energy.append(
                drone.current_weight
                * (DA[drone.type - 1] + DB[drone.type - 1] * distance_left)
            )
            timewise_speed.append(deepcopy(distance_left))

        total_time += time_taken
        total_energy += energy_consumed

    reaching_time = demand.del_to - 180
    starting_time = reaching_time - total_time
    return (
        starting_time,
        total_energy,
        total_time,
        timewise_path,
        timewise_energy,
        timewise_speed,
    )


def process_params(param_path=f"data/Scenario{sys.argv[1]}/Parameters.csv"):

    global M 
    # Cost
    global C 
    # drone count
    global drone_count 
    # Warehouse Coordinates(x,y,z)
    global wh 
    # Recharge Stations(x,y,z)
    global rhg 
    # Drone params P
    global P
    # Drone Params Q
    global Q
    # Energy params A
    global DA
    # Energy parmas B
    global DB
    # Energy params C
    global DC
    # Demands
    global demands
    # No fly zones
    global noflyzones

    global n

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


def getPath(src, dest, drone, demand):

    best_path = []
    min_cost = 1e18
    for z in range(0, 201):
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

        starting_time, energy_cost, time_taken = calculate_starting_time_energy(
            drone, path, demand
        )
        if energy_cost < min_cost:
            best_path = path
            min_cost=energy_cost

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

            path = [src]

            if noflyzones[0].doesIntersect(src, [c[0], c[1], src[2]]) or noflyzones[
                0
            ].doesIntersect([c[0], c[1], dest[2]], dest):
                continue

            path.append([c[0], c[1], src[2]])
            path.append([dest[0], dest[1], src[2]])

            if src[2] != dest[2]:
                path.append(dest)

            # Check for best path
            starting_time, energy_cost, time_taken = calculate_starting_time_energy(
                drone, path, demand
            )
            if energy_cost < min_cost:
                best_path = path
                min_cost=energy_cost

        for c1 in coordinates:
            for c2 in coordinates:
                if c1 == c2:
                    continue
                if c1[0] != c2[0] and c1[1] != c2[1]:
                    continue
                path = [src]

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
                starting_time, energy_cost, time_taken = calculate_starting_time_energy(
                    drone, path, demand
                )
                if energy_cost < min_cost:
                    best_path = path
                    min_cost=energy_cost

        return best_path
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


def simulate_travel(start,end,drone,demand,extra_demand=dummy_demand):
    demand.item.weight += extra_demand.item.weight
    demand.item.volume += extra_demand.item.volume
    possible = check_weight_volume(drone, demand.item)
    
    if not possible:
        demand.item.weight -= extra_demand.item.weight
        demand.item.volume -= extra_demand.item.volume
        return False, 0, 0, 0, []
    drone.current_weight += demand.item.weight 
    path = getPath(start, end, drone, demand)
    timestamp, energy, time_taken = calculate_starting_time_energy(
        drone, path, demand
    )
    drone.current_weight -= demand_item.weight 
    demand.item.weight -= extra_demand.item.weight
    demand.item.volume -= extra_demand.item.volume
    return drone.current_charge >= energy, energy, timestamp, time_taken, path


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
            dronetype_objects[i].maintenance_fixed_cost,
            dronetype_objects[i].maintenance_variable_cost,
        )
        drones.append(drone_object)

demands.sort(key=lambda x: x.del_to)
print(len(demands))
for index,demand in enumerate(demands):
    # processing each demand
    startpoint = wh[demand.wh - 1]
    endpoint = [demand.x, demand.y, demand.z]
    demand_item= demand.item
    if demand.is_completed:
        continue

    print(demand.demand_id)
    print(index)

    found=False

    for i in range(index+1,len(demands)):
        if demands[i].is_completed:
            continue
        for drone in drones:
            if drone.slots == 1:
                continue
            battery_used = 0.0
            done, energy, timestamp, time_taken, path = simulate_travel(startpoint, endpoint, drone, demand, demands[i])
            
            battery_used += energy
            if not done:
                continue
            sec_dest=[demands[i].x,demands[i].y,demands[i].z]
            sec_done, sec_energy, sec_timestamp, sec_time_taken, sec_path = simulate_travel(endpoint, sec_dest, drone, demands[i])

            if not sec_done or sec_timestamp < demand.del_to:
                continue

            battery_used += sec_energy

            last_done, last_energy, last_timestamp, last_time_taken, last_path = simulate_travel(sec_dest, startpoint, drone, dummy_demand)

            if not last_done:
                continue

            battery_used += last_energy

            if battery_used > drone.current_charge:
                continue

            possible = check_drone_availibility(drone, timestamp)
            if not possible:
                continue
            
            demand.is_completed = True
            demands[i].is_completed = True
            found=True
            
            drone.current_charge = drone.current_charge-battery_used
            time_for_full_recharge = np.ceil( ((drone.battery_capacity-drone.current_charge)/5000)*3600)
            drone.battery_charged = drone.battery_capacity-drone.current_charge
            drone.occupy_update(timestamp, demands[i].del_to + last_time_taken + time_for_full_recharge)
            drone.flight_time = drone.flight_time + time_taken + sec_time_taken + last_time_taken
            drone.charge_time = drone.charge_time + time_for_full_recharge
            drone.current_charge = drone.battery_capacity

            charge_start_time = int(demands[i].del_to+last_time_taken)
            time_for_full_recharge = int(time_for_full_recharge)
            for index in range(time_for_full_recharge):
                drone.x_s[charge_start_time+index] = wh[demand.wh-1][0]
                drone.y_s[charge_start_time+index] = wh[demand.wh-1][1]
                drone.z_s[charge_start_time+index] = wh[demand.wh-1][2]
                drone.speed_s[charge_start_time+index] = 0
                drone.energy_mah[charge_start_time+index] = 0
                drone.weights[charge_start_time+index] = drone.base_weight
                drone.activity[charge_start_time+index] = 'C-WH' + str(demand.wh)

            prev_weight = demand.item.weight
            demand.item.weight += demands[i].item.weight
            (starting_time, energy, time_taken, timewise_path, timewise_energy, timewise_speed) = output_path(path, drone, demand)
            demand.item.weight = demands[i].item.weight
            (starting_sec_time, sec_energy, sec_time_taken, timewise_sec_path, timewise_sec_energy, timewise_sec_speed) = output_path(sec_path, drone, demand)
            (starting_last_time, last_energy, last_time_taken, timewise_last_path, timewise_last_energy, timewise_last_speed) = output_path(last_path, drone, dummy_demand)
            
            starting_time = int(starting_time)
            starting_sec_time = int(starting_sec_time)
            starting_last_time = int(starting_last_time)
            time_taken = int(time_taken)
            sec_time_taken = int(sec_time_taken)
            last_time_taken = int(last_time_taken)
            time_counter = starting_time

            for index in range(180):
                if time_counter - index < 0:
                    break
                drone.x_s[time_counter-index] = wh[demand.wh-1][0]
                drone.y_s[time_counter-index] = wh[demand.wh-1][1]
                drone.z_s[time_counter-index] = wh[demand.wh-1][2]
                drone.speed_s[time_counter-index] = 0
                drone.energy_mah[time_counter-index] = 0
                drone.weights[time_counter-index] = drone.base_weight
                drone.activity[time_counter-index] = "PU-WH"+str(demand.wh)


            for index in range(len(timewise_path)):
                drone.x_s[time_counter] = timewise_path[index][0]
                drone.y_s[time_counter] = timewise_path[index][1]
                drone.z_s[time_counter] = timewise_path[index][2]
                drone.speed_s[time_counter] = timewise_speed[index]
                drone.energy_mah[time_counter] = timewise_energy[index]
                drone.weights[time_counter] = drone.base_weight + demand.item.weight + prev_weight
                drone.activity[time_counter] = "T-L"
                time_counter += 1

            for index in range(180):
                drone.x_s[time_counter] = timewise_path[-1][0]
                drone.y_s[time_counter] = timewise_path[-1][1]
                drone.z_s[time_counter] = timewise_path[-1][2]
                drone.speed_s[time_counter] = 0
                drone.energy_mah[time_counter] = 0
                drone.weights[time_counter] = drone.base_weight + demand.item.weight + prev_weight
                drone.activity[time_counter] = demand.demand_id
                time_counter += 1

            for index in range(len(timewise_sec_path)):
                drone.x_s[time_counter] = timewise_sec_path[index][0]
                drone.y_s[time_counter] = timewise_sec_path[index][1]
                drone.z_s[time_counter] = timewise_sec_path[index][2]
                drone.speed_s[time_counter] = timewise_sec_speed[index]
                drone.energy_mah[time_counter] = timewise_sec_energy[index]
                drone.weights[time_counter] = drone.base_weight + demand.item.weight
                drone.activity[time_counter] = "T-L"
                time_counter += 1

            for j in range(180):
                drone.x_s[time_counter] = timewise_sec_path[-1][0]
                drone.y_s[time_counter] = timewise_sec_path[-1][1]
                drone.z_s[time_counter] = timewise_sec_path[-1][2]
                drone.speed_s[time_counter] = 0
                drone.energy_mah[time_counter] = 0
                drone.weights[time_counter] = drone.base_weight + demand.item.weight
                drone.activity[time_counter] = demands[i].demand_id
                time_counter += 1
            
            for index in range(len(timewise_last_path)):
                drone.x_s[time_counter] = timewise_last_path[index][0]
                drone.y_s[time_counter] = timewise_last_path[index][1]
                drone.z_s[time_counter] = timewise_last_path[index][2]
                drone.speed_s[time_counter] = timewise_last_speed[index]
                drone.energy_mah[time_counter] = timewise_last_energy[index]
                drone.weights[time_counter] = drone.base_weight + demand.item.weight
                drone.activity[time_counter] = "T-L"
                time_counter += 1

            demands[i].weight = demand.item.weight
            demand.item.weight = prev_weight
            
            print(f"Demand {demand.demand_id} Met")
            print(f"Demand {demands[i].demand_id} Met")
            break
        if found: 
            break





    if found:
        continue
    
    for drone in drones:
        battery_used=0.0
        done, energy, timestamp, time_taken, path=simulate_travel(startpoint,endpoint,drone,demand)
        battery_used += energy
        
        if not done:
            continue

        return_done, return_energy, return_timestamp, return_time_taken, return_path = simulate_travel(endpoint,startpoint,drone,dummy_demand)

        battery_used += return_energy
        
        if not return_done or battery_used > drone.current_charge:
            continue  

        possible = check_drone_availibility(drone, timestamp)
        if not possible:
            continue
        drone.occupy_update(timestamp, demand.del_to + return_time_taken)
        demand.is_completed = True

        
        drone.current_charge = drone.current_charge-energy-return_energy
        time_for_full_recharge = np.ceil( ((drone.battery_capacity-drone.current_charge)/5000)*3600)
        drone.battery_charged = drone.battery_capacity-drone.current_charge
        drone.occupy_update(demand.del_to + return_time_taken,demand.del_to + return_time_taken+time_for_full_recharge)
        drone.flight_time = drone.flight_time + time_taken + return_time_taken
        drone.charge_time = drone.charge_time + time_for_full_recharge
        drone.current_charge = drone.battery_capacity

        charge_start_time = int(demand.del_to+return_time_taken)
        time_for_full_recharge = int(time_for_full_recharge)

        for index in range(time_for_full_recharge):
            drone.x_s[charge_start_time+index] = wh[demand.wh-1][0]
            drone.y_s[charge_start_time+index] = wh[demand.wh-1][1]
            drone.z_s[charge_start_time+index] = wh[demand.wh-1][2]
            drone.speed_s[charge_start_time+index] = 0
            drone.energy_mah[charge_start_time+index] = 0
            drone.weights[charge_start_time+index] = drone.base_weight
            drone.activity[charge_start_time+index] = 'C-WH' + str(demand.wh)


        print(path)
        (starting_time, energy, time_taken, timewise_path, timewise_energy, timewise_speed) = output_path(path, drone, demand)
        (starting_return_time, return_energy, return_time_taken, return_timewise_path, return_timewise_energy, return_timewise_speed) = output_path(path, drone, demand)

        starting_time = int(starting_time)
        starting_return_time = int(starting_return_time)
        time_taken = int(time_taken)
        return_time_taken = int(return_time_taken)
        time_counter = starting_time

        for index in range(180):
            if time_counter - index < 0:
                break
            drone.x_s[time_counter-index] = wh[demand.wh-1][0]
            drone.y_s[time_counter-index] = wh[demand.wh-1][1]
            drone.z_s[time_counter-index] = wh[demand.wh-1][2]
            drone.speed_s[time_counter-index] = 0
            drone.energy_mah[time_counter-index] = 0
            drone.weights[time_counter-index] = drone.base_weight
            drone.activity[time_counter-index] = "PU-WH"+str(demand.wh)

        for index in range(len(timewise_path)):
            drone.x_s[time_counter] = timewise_path[index][0]
            drone.y_s[time_counter] = timewise_path[index][1]
            drone.z_s[time_counter] = timewise_path[index][2]
            drone.speed_s[time_counter] = timewise_speed[index]
            drone.energy_mah[time_counter] = timewise_energy[index]
            drone.weights[time_counter] = drone.base_weight + demand.item.weight
            drone.activity[time_counter] = "T-L"
            time_counter += 1

        for index in range(180):
            drone.x_s[time_counter] = timewise_path[-1][0]
            drone.y_s[time_counter] = timewise_path[-1][1]
            drone.z_s[time_counter] = timewise_path[-1][2]
            drone.speed_s[time_counter] = 0
            drone.energy_mah[time_counter] = 0
            drone.weights[time_counter] = drone.base_weight + demand.item.weight
            drone.activity[time_counter] = demand.demand_id
            time_counter += 1

        for index in range(len(return_timewise_path)):
            drone.x_s[time_counter] = return_timewise_path[index][0]
            drone.y_s[time_counter] = return_timewise_path[index][1]
            drone.z_s[time_counter] = return_timewise_path[index][2]
            drone.speed_s[time_counter] = return_timewise_speed[index]
            drone.energy_mah[time_counter] = return_timewise_energy[index]
            drone.weights[time_counter] = drone.base_weight
            drone.activity[time_counter] = "T-E"
        break
    
    
    #print(f"{demand.demand_id} and {demand.is_completed}")
    
    if demand.is_completed:
        print(f"Demand {demand.demand_id} Met")
    else:
        print(f"Demand {demand.demand_id} not met.")

def output_costs(day):
    
    Drone_Id=[]
    Day=[]
    Charging_Time = []
    Resting_Time = []
    Maintenance_Cost = []
    Energy_Cost = []
    for drone in drones:
        # costs[drone.id][day]['flight_time']=drone.flight_time
        Drone_Id.append(drone.id)
        Day.append(day)
        Charging_Time.append(drone.charge_time)
        Resting_Time.append(14400-drone.flight_time)
        Maintenance_Cost.append(drone.maintenance_fixed_cost+(drone.maintenance_variable_cost*drone.flight_time)/3600)
        Energy_Cost.append((C*drone.battery_charged*drone.charge_time)/1000)
    
    df = pd.DataFrame({'DroneID':Drone_Id,'Day':Day,'Resting Time (s)':Resting_Time,'Charging time (s)':Charging_Time,'Maintenance Cost ($)':Maintenance_Cost,'Energy Cost ($)':Energy_Cost})
    return df

def create_path_df(day):
    
    drone_ids = []
    days = []
    timestamps = []
    x_coord = []
    y_coord = []
    z_coord = []
    speed_vals = []
    energy_vals = []
    activities = []
    weights = []
    for drone in drones:
        last_activity_index = 17999
        while drone.activity[last_activity_index]=="":
            last_activity_index -= 1                
            if last_activity_index == -1:
                drone_ids.append(drone.id)
                days.append(day)
                timestamps.append(j)
                x_coord.append(drone.x_s[j])
                y_coord.append(drone.y_s[j])
                z_coord.append(drone.z_s[j])
                speed_vals.append(drone.speed_s[j])
                energy_vals.append(drone.energy_mah[j])
                activities.append("END")
                weights.append(drone.weights[i])
                break
    
        if last_activity_index != -1:
            for j in range(0, last_activity_index+1):
                drone_ids.append(drone.id)
                days.append(day)
                timestamps.append(j)
                x_coord.append(drone.x_s[j])
                y_coord.append(drone.y_s[j])
                z_coord.append(drone.z_s[j])
                speed_vals.append(drone.speed_s[j])
                energy_vals.append(drone.energy_mah[j])
                if drone.activity[i] == "":
                    drone.activity[i] = "R-WH1"
                activities.append(drone.activity[i])
                weights.append(drone.weights[i])

            drone_ids.append(drone.id)
            days.append(day)
            timestamps.append(j)
            x_coord.append(drone.x_s[j])
            y_coord.append(drone.y_s[j])
            z_coord.append(drone.z_s[j])
            speed_vals.append(drone.speed_s[j])
            energy_vals.append(drone.energy_mah[j])
            activities.append("END")
            weights.append(drone.weights[i])

    cost_energy = [C*energy for energy in energy_vals]
    df = pd.DataFrame({"DroneID":drone_ids, "Day":days, "Time (In Seconds)":timestamps, "X":x_coord, "Y":y_coord, "Z":z_coord, "Activity":activities, "Speed (m/s)":speed_vals, "mAh Consumed":energy_vals, "Energy Cost (c x mAh)":cost_energy, "Total Weight (kgs)":weights})
    df.to_csv("./DronePath_Output.csv", index=False)




day1_costs = output_costs(1)


day1_costs.to_csv(f"data/Scenario{sys.argv[1]}/DroneCost_Output.csv",index=False)
create_path_df(1)




