"""
Rachel and Julie

Project 2
"""

import cv2
import numpy


########################

def pyr_build(Gi):

  lp = []

  win = 'Image Window'
  cv2.namedWindow(win)

  for i in range (4):
    #scale it down
    Gi1 = cv2.pyrDown(Gi)
    print 'Gi1 has shape', Gi1.shape

    #just specifying size as (w, h) works
    uGi1 = cv2.pyrUp(Gi1, dstsize=(w, h))
    print 'unsmall has shape', uGi1.shape

    Gi.astype(numpy.float32)
    uGi1.astype(numpy.float32)
    Li = Gi - uGi1

    lp.append(Li)

    cv2.imshow(win, 0.5 + 0.5 * (Li / numpy.abs(Li).max()))

    while cv2.waitKey(15) < 0: pass

    Gi = uGi1
  return lp

#########################

G0 = cv2.imread('Golden_Retriever.JPG')

h = G0.shape[0]
w = G0.shape[1]

pyr_build(G0)
