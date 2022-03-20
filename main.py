import pandas as pd
import numpy as np

def read_data(demands_path="Demand.csv", param_path="Parameters.csv"):
  demand = pd.read_csv(demands_path)
  parameters = pd.read_csv(param_path)
  print(demand.head())
  print(parameters.head())


if __name__ == "__main__":
  read_data()
