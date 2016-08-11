import pandas as pd

csv = pd.read_csv('photos.csv')

for i in range(0, 240000):
    csv.loc[i,]
