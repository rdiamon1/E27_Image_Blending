import cv2
import numpy

# generate a test image with odd dimensions
h = 47
w = 63
img = numpy.random.randint(0, 256, size=(h, w, 3)).astype('uint8')
print 'img has shape', img.shape

# scale it down
small = cv2.pyrDown(img)
print 'small has shape', small.shape

# just specifying size as (w, h) works
unsmall = cv2.pyrUp(small, dstsize=(w, h))
print 'unsmall has shape', unsmall.shape

# specifying None as the second positional argument works
unsmall = cv2.pyrUp(small, None, (w, h))
print 'unsmall has shape', unsmall.shape

# specifying a destination image works
dst = numpy.empty_like(img)
unsmall = cv2.pyrUp(small, dst, (w, h))
print 'unsmall has shape', unsmall.shape

# specifying a 3-tuple as the dstsize seems to cause the error Josh
# encountered.
unsmall = cv2.pyrUp(small, dstsize=img.shape)
print 'unsmall has shape', unsmall.shape
