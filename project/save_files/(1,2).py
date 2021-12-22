import cv2
import math
import numpy as np
import config as cf
import function as f


#1) 영상에서 첫번째 프레임 추출하는 알고리즘 #이지만 초당 10개씩 16개 프레임 추출했음 / 후에 동영상 제작 

OUTPUT_PATH='/Users/mba13/RB_distance_estimation/FrameExtraction/output/'
DATA_PATH='/Users/mba13/RB_distance_estimation/FrameExtraction/vehicle_video1.mov'


def method(dataPath,savePath,n): #method 함수 정의 -프레임 첫번째 값 추출 

    cap = cv2.VideoCapture(dataPath)
    if cap.isOpened():
        rval=True
    else:
        rval=False

    length=cap.get(7)
    gap=n
    currentF = 0
    tmp=0

    while rval:


        rval,frame = cap.read()

        if tmp == gap:
            tmp=0

        if tmp == 0:
            cv2.imwrite(savePath+str(currentF)+'.jpg',frame)
            print(currentF)

        currentF+=1
        tmp+=1
        #cv2.waitKey(1)

    return

def test_three():
        f.method(DATA_PATH,OUTPUT_PATH, 10)
        return

if __name__ == '__main__':
        test_three()


#sub_img 추출 알고리즘 #sub-img ㄹㅡㄹ 추출해 저장

def sub_img_ext():
    img1 = cv2.imread(OUTPUT_PATH + '0.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = img1[200:400, 270:430]
    img2 += 20

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imwrite('/Users/mba13/RB_distance_estimation/FrameExtraction/sub_image.jpg',img2)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sub_img_ext()

