import queue
import threading
import time


class Mth:
    def __init__(self):
        self.qu = queue.Queue()


class A:
    def __init__(self, x):
        self.x = x

    def increase_x(self, qu):
        while True:
            self.x += 1
            qu.put(self.x)
            time.sleep(0.1)



class B:
    def __init__(self, value):
        self.y = value

    def print_p(self, qu):
        while True:
            self.y = qu.get()
            print(self.y)

import cv2
import numpy as np

def main():

    bgr_image = cv2.imread('img2.akspic.ru-priroda-pustynya-nebo-lug-rastitelnost-4321x2882.jpg')

    im = cv2.resize(bgr_image, (80, 80))

    _, k = cv2.threshold(bgr_image[:, :, 1], 170, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('k', k)

    cv2.imwrite("mask_inv_nat.png", k)






    #b, g, r = cv2.split(bgr_image)


    #_, k = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY_INV)
    #img_ = cv2.merge((b, g, r, k))
#
    #b, g ,r = cv2.split(bgr_image)
    #b = cv2.bitwise_not(b)
    #_, k = cv2.threshold(b, 170, 255, cv2.THRESH_BINARY_INV)
    #3cv2.imshow('y', k)
    #cv2.waitKey(0)
    #cv2.imwrite("mask.png", k)

    #l = cv2.bitwise_and(b, b, k)
    #img = cv2.merge((l, l, l, k))
    ##
    #print(img.shape)
    #cv2.imshow('y', img)
    #cv2.waitKey(0)
#
    #cv2.imwrite("new_image.png", img)
#
def analiz():
    img = cv2.imread('new_image.jpg')
    img = cv2.resize(img, (80, 80))

    a, b = img.shape[:-1]
    print(a, b)
    pst_point = np.array(((0, 0), (0, 80), (80, 0), (80, 80)), dtype=np.float32)

    dst_poiny = np.array(((477, 293), (531, 303), (455, 324), (513, 335)), np.float32)

    M = cv2.getPerspectiveTransform(pst_point, dst_poiny)

    img_ = cv2.warpPerspective(img, M, (640, 480))


    cv2.imshow("img", img_)
    cv2.waitKey(0)

if __name__ == "__main__":
    analiz()



#    a = A(1)
#    b = B(a.x)
#    mth = Mth()
#
#    th1 = threading.Thread(target=a.increase_x, args=(mth.qu, ))
#    th2 = threading.Thread(target=b.print_p, args=(mth.qu,))
#
#    th1.start()
#    th2.start()




