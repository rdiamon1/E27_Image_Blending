"""
Rachel Diamond and Julie Harris

Project 2
"""

import cv2
import numpy

NUM_PYR = 3 #number of iterartions in the pyramid

win = 'Image Window'
cv2.namedWindow(win)
########################

def pyr_build(Gi):

  lp = []


  for i in range (NUM_PYR):
    #scale it down
    h = Gi.shape[0]
    w = Gi.shape[1]
    Gi1 = cv2.pyrDown(Gi)
    print 'Gi1 has shape', Gi1.shape

    #just specifying size as (w, h) works
    uGi1 = cv2.pyrUp(Gi1, dstsize=(w, h))
    print 'unsmall has shape', uGi1.shape

    Gi = Gi.astype(numpy.float32)
    uGi1 = uGi1.astype(numpy.float32)
    Li = Gi - uGi1
    lp.append(Li)

    cv2.imshow(win, 0.5 + 0.5 * (Li / numpy.abs(Li).max()))

    while cv2.waitKey(15) < 0: pass

    Gi = Gi1

  lp.append(Gi)
  cv2.imshow(win, Gi/255.0)
  while cv2.waitKey(15) < 0: pass
  return lp

#########################
def pyr_reconstruct(lp):

    Rn = lp[-1]
    h = Rn.shape[0]
    w = Rn.shape[1]
    image = Rn

    Ri = Rn
    for i in range(len(lp)-1,0,-1):
        #uRi = cv2.pyrUp(Ri, dstsize=(w, h))

        uRi = cv2.pyrUp(Ri)
        uRi = cv2.resize(uRi,(w,h))

        Ri1 = uRi + lp[i]
        image += Ri1
        Ri = Ri1

    image = image.astype(numpy.uint8)
    cv2.imshow(win, image)
    while cv2.waitKey(15) < 0: pass

#########################
def main():
    G0 = cv2.imread('Golden_Retriever.JPG')
    #G0 = cv2.cvtColor(G0, cv2.COLOR_RGB2GRAY)

    h = G0.shape[0]
    w = G0.shape[1]

    lp = pyr_build(G0)
    pyr_reconstruct(lp)

main()
