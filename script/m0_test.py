
import numpy as np

if 0:
    # result = 1
    print((1 == 1, 0 == 1) and 1)

if 1:
    trainTimes_poseInShape_array = [1800, 1500, 1200, 1200, 1200, 1200]
    sum = 0
    for i in range(1, trainTimes_poseInShape_array.__len__()):
        sum = sum + trainTimes_poseInShape_array[i]
    print("sum: %d" % sum)
