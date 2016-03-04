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

  Gi = Gi.astype(numpy.float32)
  lp.append(Gi)
  cv2.imshow(win, Gi/255.0)
  while cv2.waitKey(15) < 0: pass

  return lp

#########################
def pyr_reconstruct(lp):

    Ri = lp[-1]

    for i in range(len(lp)-1,0,-1):

        h = Ri.shape[0]
        w = Ri.shape[1]

        print "h = ", h
        print "w = ", w

        uRi = cv2.pyrUp(Ri, dstsize=(2*w, 2*h))

        print "uRi h = ", uRi.shape[0]
        print "uRi w = ", uRi.shape[1]

        print "lp[i-1] h = ", lp[i-1].shape[0]
        print "lp[i-1] w = ", lp[i-1].shape[1]

        Ri1 = uRi + lp[i-1]

        Ri = Ri1

        print "iteration = ", i

    Ri = Ri.astype(numpy.uint8)
    cv2.imshow(win, Ri)
    while cv2.waitKey(15) < 0: pass

#########################
def alpha_blend(A, B, alpha):
    A = A.astype(alpha.dtype)
    B = B.astype(alpha.dtype)
    # if A and B are RGB images, we must pad
    # out alpha to be the right shape
    if len(A.shape) == 3:
        alpha = numpy.expand_dims(alpha, 2)
    return A + alpha*(B-A)

#########################
def main():
    G0 = cv2.imread('Golden_Retreiver02.JPG')

    h = G0.shape[0]
    w = G0.shape[1]

    lp = pyr_build(G0)
    pyr_reconstruct(lp)

    imgA = cv2.imread('cat_edit02.jpg')
    imgB = cv2.imread('dog_edit02.jpg')

main()
