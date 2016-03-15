"""
Rachel Diamond and Julie Harris
Project 2: Image blending
"""

import cv2
import numpy

PYR_DEPTH = 4  #number of iterartions (depth) for the Lapacian pyramid

#########################
def pyr_build(Gi):
  '''
  Parameters: uint8 image and int representing pyramid depth
  Returns: list of float32 images
  '''

  lp = [] #list to accumulate pyramid

  for i in range(PYR_DEPTH):
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

    print '...pyramid level',i+1,'...'
    cv2.imshow('image', 0.5 + 0.5 * (Li / numpy.abs(Li).max()))
    while cv2.waitKey(15) < 0: pass

    Gi = Gi1

  Gi = Gi.astype(numpy.float32)
  lp.append(Gi)

  return lp

#########################
def pyr_reconstruct(lp):
    '''takes list of float32 images and returns single uint8 image'''

    Ri = lp[-1]

    for i in range(len(lp)-1,0,-1):

        h = Ri.shape[0]
        w = Ri.shape[1]

        uRi = cv2.pyrUp(Ri, dstsize=(2*w, 2*h))

        #threshold to prevent overflow by setting all negative values to zero
        uRi= uRi.astype(numpy.float32)
        xx,uRi = cv2.threshold(uRi,0,0,cv2.THRESH_TOZERO)

        Ri1 = uRi + lp[i-1]

        Ri = Ri1

    #threshold to prevent overflow by setting values above 255 to 255
    Ri= Ri.astype(numpy.float32)
    xx,Ri = cv2.threshold(Ri,255,0,cv2.THRESH_TRUNC)

    Ri = Ri.astype(numpy.uint8)
    print 'Ri range',Ri.min(),Ri.max()
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
def makeAlpha(A):
    '''make an alpha mask of the same dimesions as the image A'''
    height = A.shape[0]
    width = A.shape[1]

    #make vector form of alpha using a logistic equation with sloped region
    #centered within the image
    steepness = 300.0/width #increase numerator to decrease blend width
    x = numpy.arange(width)
    alpha = 1/(1+numpy.exp(-steepness*(x-0.5*width)))

    #extend vector to be the same size as the image
    alpha = numpy.tile(alpha,(height,1))

    #cv2.imshow('alpha', alpha)
    #while cv2.waitKey(5) < 0: pass

    return alpha

#########################
def hybrid(A, B):
    #define parameters
    kA = 1
    kB = 1
    sigmaA = 9
    sigmaB = 6

    #make sure images are float32 before performing operations
    A = A.astype(numpy.float32)
    B = B.astype(numpy.float32)

    # low-pass filter on image A
    Anew = cv2.GaussianBlur(A, (0,0), sigmaA)

    # high-pass filter on image B
    Bintr = cv2.GaussianBlur(B, (0,0), sigmaB)
    Bnew = B - Bintr

    #threshold to prevent overflow by setting all negative values to zero
    xx,Bnew = cv2.threshold(Bnew,0,0,cv2.THRESH_TOZERO)

    hybrid_img = (kA * Anew) + (kB * Bnew)

    return hybrid_img
#########################
def main():
    #read in images
    imgA = cv2.imread('cat_square.JPG')
    imgB = cv2.imread('dog_square.JPG')
    #imgA = cv2.imread('sun.jpg')
    #imgB = cv2.imread('moon.jpg')
    imgC = cv2.imread('puppy_face.jpg')
    imgD = cv2.imread('Julie_face.jpg')

    #make the images for the hybrid images greyscale
    imgC = cv2.cvtColor(imgC, cv2.COLOR_RGB2GRAY)
    imgD = cv2.cvtColor(imgD, cv2.COLOR_RGB2GRAY)

    #make two Lapacian pyramids
    lpA = pyr_build(imgA)
    lpB = pyr_build(imgB)
    lpAB = []

    #blend the pyramids
    alpha = makeAlpha(imgA)
    for i in range(len(lpA)):
        lpAB.append(alpha_blend(lpA[i],lpB[i],alpha))
        aheight = alpha.shape[0]
        awidth = alpha.shape[1]
        alpha = cv2.resize(alpha,(awidth/2,aheight/2),interpolation=cv2.INTER_AREA)

    #reconstruct and show blended image
    print '...reconstructed image...'
    blended_img = pyr_reconstruct(lpAB)
    cv2.imshow("image", blended_img)
    while cv2.waitKey(15) < 0: pass

    #make and show the naive/traditional blend (without Lapacian pyramid)
    print '...naive blend...'
    alpha = makeAlpha(imgA)
    naive_blend = alpha_blend(imgA,imgB,alpha)
    cv2.imshow("image", naive_blend/255) #divide by 255 since it's a float32
    while cv2.waitKey(15) < 0: pass

    #make and show hybrid image
    print '...hybrid...'
    hybrid_img = hybrid(imgC, imgD)
    cv2.imshow("image", hybrid_img/255) #divide by 255 since it's a float32
    while cv2.waitKey(15) < 0: pass

    #build pyramid of hybrid, just doing this to see what the pyramid looks
    #like, so we don't need to save the result
    pyr_build(hybrid_img)

#######################
main()
