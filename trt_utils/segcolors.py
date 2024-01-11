import numpy as np

classes=['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

colors=np.array([
 [31,   120,   180], # rl lane
 [227,  26,    28], # mid line
 [106, 61, 154], # car
 [0,   0,   0],
], dtype=np.uint8)

lanecolor=np.array([
 [31,   120,   180], # rl lane
 [31,   120,   100], # mid line
 [0,   0,   0],
 [0,   0,   0],
 [0,   0,   0],
], dtype=np.uint8)

midcolor=np.array([
 [0,   0,   0],
 [31,   120,   190], # mid line
 [0,   0,   0],
 [0,   0,   0],
 [0,   0,   0],
], dtype=np.uint8)

obstacle=np.array([
 [0,   0,   0], # rl lane
 [0,   0,    0], # mid line
 [255, 255, 255], # car
 [255, 255, 255], # person
 [0,   0,   0],
], dtype=np.uint8)