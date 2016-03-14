"""
Rachel Diamond and Julie Harris
Project 2: Image blending
"""

import cv2
import numpy

NUM_PYR = 3 #number of iterartions in the pyramid

#########################
def pyr_build(Gi):

  lp = []

  for i in range (NUM_PYR):
    #scale it down
    h = Gi.shape[0]
    w = Gi.shape[1]
    Gi1 = cv2.pyrDown(Gi)

    #just specifying size as (w, h) works
    uGi1 = cv2.pyrUp(Gi1, dstsize=(w, h))

    Gi = Gi.astype(numpy.float32)
    uGi1 = uGi1.astype(numpy.float32)
    Li = Gi - uGi1
    lp.append(Li)

    Gi = Gi1

  Gi = Gi.astype(numpy.float32)
  lp.append(Gi)

  return lp

#########################
def pyr_reconstruct(lp):

    Ri = lp[-1]

    for i in range(len(lp)-1,0,-1):

        h = Ri.shape[0]
        w = Ri.shape[1]

        uRi = cv2.pyrUp(Ri, dstsize=(2*w, 2*h))

        Ri1 = uRi + lp[i-1]

        Ri = Ri1

    Ri = Ri.astype(numpy.uint8)
    return Ri

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
def blend(A,B):
        width = A.shape[0]

        steepness = 50.0/width #increase numerator to decrease blend width
        print 'steepness = %f'%(steepness)
        x = range(width)
        midpt = width*0.5*numpy.ones(width)
        alpha = 1/(1+numpy.exp(-steepness*(x-midpt)))
        blended_img = alpha_blend(A,B,alpha)

        return blended_img

#########################
def Hybrid(A, B):
    kA = 1
    kB = 2
    sigmaA = 7
    sigmaB = 4

    

    # lo-pass filter
    Anew = cv2.GaussianBlur(A, (0,0), sigmaA)

    # high-pass filter
    Bnew = B - cv2.GaussianBlur(B, (0,0), sigmaB)

    hybridimg = (kA * Anew) + (kB * Bnew)

    return hybridimg

#########################
def main():
    imgA = cv2.imread('cat_square.JPG')
    imgB = cv2.imread('dog_square.JPG')

    imgC = cv2.imread('puppy_face.jpg')
    imgD = cv2.imread('Julie_face.jpg')

    #make two Lapacian pyramids
    lpA = pyr_build(imgA)
    lpB = pyr_build(imgB)
    lpAB = []

    #blend the pyramids
    for i in range(len(lpA)):
        lpAB.append(blend(lpA[i],lpB[i]))
        #cv2.imshow("blend", lpA[i])
        #while cv2.waitKey(15) < 0: pass

    #reconstruct and show hybrid image
    blended_img = pyr_reconstruct(lpAB)
    cv2.imshow("blend", blended_img)
    while cv2.waitKey(15) < 0: pass

    naive_blend = blend(imgA,imgB)*.004
    cv2.imshow("blend", naive_blend)
    while cv2.waitKey(15) < 0: pass

    hybrid_img = Hybrid(imgC, imgD)
    cv2.imshow("hybrid", hybrid_img)
    while cv2.waitKey(15) < 0: pass

main()
