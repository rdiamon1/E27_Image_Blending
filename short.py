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
    print 'alpha is', alpha.shape
    print 'image a is',A.shape
    print 'image b is',B.shape

    #cv2.imshow('alpha', alpha)
    #while cv2.waitKey(5) < 0: pass

    return A + alpha*(B-A)

#########################
'''
def blend(A,B):
        width = A.shape[1]
        height = A.shape[0]

        steepness = 50.0/width #increase numerator to decrease blend width
        print 'steepness = %f'%(steepness)
        x = numpy.arange(width)
        #midpt = width*0.5*numpy.ones(width)
        alpha = 1/(1+numpy.exp(-steepness*(x-0.5*width))) #make logistic vector

        #extend vector to be the same size as the image
        alpha = numpy.tile(alpha,(height,1))

        blended_img = alpha_blend(A,B,alpha)

        return blended_img
'''
#########################
def makeAlpha(A):
    '''make an alpha mask of the same dimesions as the image A'''
    height = A.shape[0]
    width = A.shape[1]

    steepness = 200.0/width #increase numerator to decrease blend width
    print 'steepness = %f'%(steepness)
    x = numpy.arange(width)
    #midpt = width*0.5*numpy.ones(width)
    alpha = 1/(1+numpy.exp(-steepness*(x-0.5*width))) #make logistic vector

    #extend vector to be the same size as the image
    alpha = numpy.tile(alpha,(height,1))
    #cv2.imshow('alpha', alpha)
    #while cv2.waitKey(5) < 0: pass

    return alpha

#########################
def Hybrid(A, B):
    kA = 2
    kB = 1.8
    sigmaA = 7
    sigmaB = 4

    A = A.astype(numpy.float32)
    B = B.astype(numpy.float32)
    print 'A range:', A.min(), A.max()
    print 'B range:', B.min(), B.max()

    # low-pass filter
    Anew = cv2.GaussianBlur(A, (0,0), sigmaA)

    cv2.imshow('Anew', Anew)
    while cv2.waitKey(15) < 0: pass

    # high-pass filter
    Bnew = B - cv2.GaussianBlur(B, (0,0), sigmaB)
    xx,Bnew = cv2.threshold(Bnew,0,0,cv2.THRESH_TOZERO)
    print 'Anew range:', Anew.min(), Anew.max()
    print 'Bnew range:', Bnew.min(), Bnew.max()

    cv2.imshow('Bnew', Bnew)
    while cv2.waitKey(15) < 0: pass

    hybridimg = (kA * Anew) + (kB * Bnew)

    return hybridimg

#########################
def main():
    imgA = cv2.imread('cat_square.JPG')
    imgB = cv2.imread('dog_square.JPG')
    imgA = cv2.imread('angryface.jpg')
    imgB = cv2.imread('happyface.jpg')

    imgC = cv2.imread('puppy_face.jpg')
    imgD = cv2.imread('Julie_face.jpg')
    imgC = cv2.cvtColor(imgC, cv2.COLOR_RGB2GRAY)
    imgD = cv2.cvtColor(imgD, cv2.COLOR_RGB2GRAY)

    #make two Lapacian pyramids
    lpA = pyr_build(imgA)
    lpB = pyr_build(imgB)
    lpAB = []

    alpha = makeAlpha(imgA)
    #blend the pyramids
    for i in range(len(lpA)):
        #lpAB.append(blend(lpA[i],lpB[i]))
        #cv2.imshow("blend", lpA[i])
        #while cv2.waitKey(15) < 0: pass
        lpAB.append(alpha_blend(lpA[i],lpB[i],alpha))
        print 'i in main',i
        aheight = alpha.shape[0]
        awidth = alpha.shape[1]
        alpha = cv2.resize(alpha,(awidth/2,aheight/2),interpolation=cv2.INTER_AREA)
    print '...reconstructed image...'

    #reconstruct and show hybrid image
    blended_img = pyr_reconstruct(lpAB)
    cv2.imshow("blend", blended_img)
    while cv2.waitKey(15) < 0: pass

    print 'imgA range:', imgA.min(), imgA.max()
    print 'imgB range:', imgB.min(), imgB.max()
    #naive_blend = blend(imgA,imgB)
    alpha = makeAlpha(imgA)
    naive_blend = alpha_blend(imgA,imgB,alpha)
    print '...naive blend...'
    cv2.imshow("blend", naive_blend/255)
    while cv2.waitKey(15) < 0: pass

    print '...hybrid...'
    hybrid_img = Hybrid(imgC, imgD)
    hybrid_img = hybrid_img.astype(numpy.uint8)
    print 'hybrid_img range:', hybrid_img.min(), hybrid_img.max()
    cv2.imshow("hybrid", hybrid_img)
    #cv2.imshow("hybrid", 0.5 + 0.5 * (hybrid_img / numpy.abs(hybrid_img).max()))
    while cv2.waitKey(15) < 0: pass

main()
