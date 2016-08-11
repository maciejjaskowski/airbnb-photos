import lmdb
import os
from PIL import Image
from scipy import misc
import numpy as np
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath("src"))
from datum import Datum, DatumJPEG

path = 'photos-all/'
files = [path + f for f in os.listdir(path) if f.endswith(".jpg")]
files = files[:3000]
csv = pd.read_csv('photos.csv')
csv = csv[['hosting_id']]
data = pd.read_csv('data-cleaned.csv')
data = data[['hostingId', 'airEventData.price']].drop_duplicates()
prices = pd.merge(csv, data, how='inner', left_on='hosting_id', right_on='hostingId')[['airEventData.price']]

def id(file):
    return int(file.split("/")[-1][:-4])

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
map_size = 240 * 240 * 3 * 4 * len(files)

import time
start = time.time()

env = lmdb.open('mylmdb1', map_size=map_size, writemap=False, max_dbs=1)


with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i, file in enumerate(files):
        price=prices[i,'airEventData.price']

        f = open(file,"wb")
        from StringIO import StringIO
        output = StringIO()
        X = Image.open(file)
        X.save(output, format="JPEG")
        X = output.getvalue()
        output.close()
        datum = DatumJPEG(X, 0)
        datum_price = DatumFloat()


    @staticmethod
    def deserialize(bytes):
        data, label = pickle.loads(bytes)
        return DatumJPEG(Image.open(data), label)
        str_id = '{:08}'.format(id(file))

        txn.delete(str_id.encode('ascii'))
        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.serialize())
end = time.time()
print(end-start)

env = lmdb.open('mylmdb1', readonly=True)
with env.begin() as txn:
    datum = Datum.deserialize(txn.get(b'00000001'))

start = time.time()
env = lmdb.open('mylmdb1', readonly=True, max_dbs=1)
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:

end = time.time()
print(end-start)

start = time.time()
for file in files[:10000]:
    #X = misc.imread(file)
    X = np.array(Image.open(file))
end = time.time()
print(end-start)
