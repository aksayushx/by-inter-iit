from email.mime import base
from typing import final
from xml.dom.minidom import Attr
import pandas as pd
import numpy as np
from sniffio import AsyncLibraryNotFoundError

class Drone:
  def __init__(self,id, type,battery_capacity,base_weight,payload_weight,payload_volume,slots,max_speed, p, q, a, b, c):

    self.id = id
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
    self.p, self.q, self.a, self.b, self.c = p, q, a, b, c


class Item:
  def __init__(self,id,weight,l,b,h):
    self.id = id
    self.weight = weight
    self.l = l
    self.b = b
    self.h = h
    
    self.volume = self.l * self.b * self.h 

class Demand:
  def __init__(self, id, x, y, z, start_time, end_time, item_type, is_completed):
    self.id = id
    self.x, self.y, self.z = x, y, z
    self.start_time = start_time
    self.end_time = end_time
    self.item_type = item_type
    self.is_completed = is_completed

class RechargePoints:
  def __init__(self, id, x, y, z, slots, current):
      self.id = id
      self.x, self.y, self.z = x, y, z
      self.slots = slots
      self.current = current


class Path:
  def __init__(self, day, start_time, end_time, demand, activity, current_speed, drone):
    self.day = day
    self.start_time = start_time
    self.end_time = end_time
    self.demand = demand
    self.activity = activity
    self.current_speed = current_speed
    self.drone = drone

def read_data(demands_path='Demand.csv',param_path='Parameters.csv'):
  demand=pd.read_csv(demands_path)
  parameters=pd.read_csv(param_path)
  # print(demand.head())
  # print(parameters.head())

def get_nearest_recharge_station(location, warehouse_locations, recharge_points):
  pass

def get_next_point(drone, current_location, demand_points, warehouse_locations, recharge_points):
  for point in demand_points:
    pass


def get_next_drone(drone_available_time, drones):
  next_time = 10**9
  final_drone = None
  for drone in drones:
    if(next_time>drone_available_time[drone]):
      final_drone = drone
      next_time=drone_available_time[drone]
  
  return final_drone


if __name__=='__main__':
  read_data()
  d1 = Drone(1, 1, 2000, 2, 5, 200, 1, 10, 1, 1, 1, 1, 1)
  d2 = Drone(2, 2, 5000, 4, 6, 500, 4, 15, 1, 1, 1, 1, 1)
  drones = [d1, d2]

  demand_point1 = Demand(1,100,200,300,100000,100600, 1, False)
  demand_point2 = Demand(2,-100,-200,300,110000,100600, 1, False)
  demand_point3 = Demand(3,100,200,250,110000,105600, 1, False)
  demand_points = [demand_point1, demand_point2, demand_point3]
  for i in demand_points:
    print(i.__dict__)