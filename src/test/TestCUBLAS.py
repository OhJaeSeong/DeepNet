"""
Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)
"""

import numpy

aa = [[1, 2, 3],
      [4, 5, 6]]

bb = [[1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12]]

aa = numpy.array(aa)
bb = numpy.array(bb)

cc = numpy.dot(aa, bb)

print(cc)

# Result:
# [[ 38  44  50  56]
#  [ 83  98 113 128]]
