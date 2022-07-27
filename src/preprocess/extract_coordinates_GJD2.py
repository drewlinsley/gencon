"""Precompute coordinates for processing a volume."""
import numpy as np
from db import db


vol_shape = [25696, 32278, 959]
proc_shape =[4740, 4740, 240]
overlap = [384, 384, 120]

coords = []
for x in range(0, vol_shape[0], proc_shape[0] - overlap[0]):
    for y in range(0, vol_shape[1], proc_shape[1] - overlap[1]):
        for z in range(0, vol_shape[2], proc_shape[2] - overlap[2]):
            coords.append([x, y, z])
print("Adding coordinates to DB.")
db.populate_db(coords)

