from PIL import Image
import numpy as np
import cv2


image = Image.open('bg_images/andrew-ridley-76547-unsplash.jpg')
image = image.resize((512,512))
cv2.imshow('test', np.array(image))

_width, _height = image.size
pts1 = np.float32([[0,0],
                    [_width-1,0],
                    [0,_height-1],
                    [_width-1,_height-1]])

pts2 = np.float32([[0,0],
                    [(_width-1)//2,0],
                    [0,(_height-1)//2],
                    [(_width-1)//2,(_height-1)//2]])
# pts2 = np.float32([ch['quad_roi'][0],
#                         ch['quad_roi'][1],
#                         ch['quad_roi'][3],
#                         ch['quad_roi'][2]])

print(pts1)
print(pts2)
# image.show()
M = cv2.getPerspectiveTransform(pts1,pts2)
img_result = cv2.warpPerspective(np.array(image), M, (_width, _height))
print(M)
out = image.transform(image.size, Image.PERSPECTIVE, M.flatten(),fillcolor=(0,0,0))#, resample.Image.BILINEAR)
print(out.size)
cv2.imshow('test1', np.array(out+image))
cv2.imshow('test2', img_result)
cv2.waitKey()