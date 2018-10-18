import pandas as pd
import numpy as np


dictionary1 = np.array(pd.read_csv('data/out/dictionary4.csv', sep=" ", header=None))
print(len(dictionary1))