"""
Rachel and Julie

Project 2
"""

import cv2
import numpy

NUM_PYR = 15 #number of iterartions in the pyramid

########################

def pyr_build(Gi):

  lp = []
  h = Gi.shape[0]
  w = Gi.shape[1]

  win = 'Image Window'
  cv2.namedWindow(win)

  for i in range (NUM_PYR):
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
def pyr_reconstruct(lp):

    win = 'Image Window'
    cv2.namedWindow(win)

    Rn = lp[-1]
    h,w,d = Rn.shape
    image = Rn

    Ri = Rn
    for i in range(len(lp)-1,0,-1):
        #uRi = cv2.pyrUp(Ri, dstsize=(w, h))

        uRi = cv2.pyrUp(Ri)
        uRi = cv2.resize(uRi,(w,h))

        Ri1 = uRi + lp[i]
        image += Ri1
        Ri = Ri1

    image.astype(numpy.uint8)
    cv2.imshow(win, image)
    while cv2.waitKey(15) < 0: pass

#########################
def main():
    G0 = cv2.imread('Golden_Retriever.JPG')

    h = G0.shape[0]
    w = G0.shape[1]

    lp = pyr_build(G0)
    pyr_reconstruct(lp)

main()
