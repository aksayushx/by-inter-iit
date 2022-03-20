from utils import Drone,Item
import pandas as pd
import numpy as np

def read_data(demands_path="Demand.csv"):
  demand = pd.read_csv(demands_path)
  print(demand.head())


if __name__ == "__main__":
  read_data()
