from utils import Drone,Item,Demand
import pandas as pd
import numpy as np
from string import ascii_uppercase

# Max Speed
M=0.0
#Cost
C=0.0
#drone count
drone_count=[0,0,0,0,0,0]
#Warehouse Coordinates(x,y,z)
wh=[]
#Recharge Stations(x,y,z)
rhg=[]
#Drone params P
P=[0.,0.,0.,0.,0.,0.]
#Drone Params Q
Q=[0.,0.,0.,0.,0.,0.]
#Energy params A
DA=[0.,0.,0.,0.,0.,0.]
#Energy parmas B
DB=[0.,0.,0.,0.,0.,0.]
#Energy params C
DC=[0.,0.,0.,0.,0.,0.]
#Demands
demands=[]


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


def read_demands(demands_path="data/Demand.csv"):
  demand = pd.read_csv(demands_path)
  print(demand.head())
  for index,row in demand.iterrows():
    warehouse=row['WH']
    demand_id=row['Demand ID']
    item=row['Item']
    day=row['Day']
    x=row['X']
    y=row['Y']
    z=row['Z']
    del_from=row['DeliveryFrom']
    del_to=row['DeliveryTo']
    failure=row['DeliveryFailure']
    demands.append(Demand(warehouse,demand_id,item,day,x,y,z,del_from,del_to,failure))









def process_params(param_path="data/Parameters.csv"):
  
  parameters=pd.read_csv(param_path)
  print(parameters.head())
  
  M=parameters.loc[parameters['Parameter_ID']=='MaxSpeed (M)','Value'].iloc[0]
  C=parameters.loc[parameters['Parameter_ID']=='Cost(C)','Value'].iloc[0]
  
  for i in range(1,4):
    df=parameters.loc[parameters['Parameter_ID']=='WH'+str(i)+'X','Value']
    if(df.empty):
      break
    vx=parameters.loc[parameters['Parameter_ID']=='WH'+str(i)+'X','Value'].iloc[0]
    vy=parameters.loc[parameters['Parameter_ID']=='WH'+str(i)+'Y','Value'].iloc[0]
    vz=parameters.loc[parameters['Parameter_ID']=='WH'+str(i)+'Z','Value'].iloc[0]
    wh.append([vx,vy,vz])
  
  for i in ascii_uppercase:
    df=parameters.loc[parameters['Parameter_ID']==i+'X1','Value']
    if(df.empty):
      break
    vx=parameters.loc[parameters['Parameter_ID']==i+'X1','Value'].iloc[0]
    vy=parameters.loc[parameters['Parameter_ID']==i+'Y1','Value'].iloc[0]
    rhg.append([vx,vy,0.0])

  for i in range(1,7):
    p=parameters.loc[parameters['Parameter_ID']=='P'+str(i),'Value'].iloc[0]
    q=parameters.loc[parameters['Parameter_ID']=='Q'+str(i),'Value'].iloc[0]
    a=parameters.loc[parameters['Parameter_ID']=='A'+str(i),'Value'].iloc[0]
    b=parameters.loc[parameters['Parameter_ID']=='B'+str(i),'Value'].iloc[0]
    c=parameters.loc[parameters['Parameter_ID']=='C'+str(i),'Value'].iloc[0]
    count=parameters.loc[parameters['Parameter_ID']=='DT'+str(i)+'Count','Value'].iloc[0]
    P[i-1]=p
    Q[i-1]=q
    DA[i-1]=a
    DB[i-1]=b
    DC[i-1]=c
    drone_count[i-1]=count

if __name__ == "__main__":
  process_params()
  read_demands()
    # max_speed and drone counts will come from parameters
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
          dronetype_objects[i].max_speed
        )
      drones.append(drone_object)
  print(len(drones))
  itemtype_objects = read_item_details()