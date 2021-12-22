import numpy as np
import cv2
import PIL
from PIL import Image,ImageDraw,ImageFont
import os
import math
 

#추출한 sub-img와 프레임 이미지로 orb visual featuring 후 차량 detection, detection 한 차량의 하단부 좌표 구함 
#세이브 파일 만들어서 각 프레임 저장하기 

frame_no = 0 
while frame_no <= 150:

    def find_homography():
        src1 = cv2.imread('/Users/mba13/RB_distance_estimation/FrameExtraction/sub_image.png', cv2.IMREAD_GRAYSCALE)
        src2 = cv2.imread('/Users/mba13/RB_distance_estimation/FrameExtraction/output/' +str(frame_no) +'.jpg', cv2.IMREAD_GRAYSCALE)

        if src1 is None or src2 is None:
            print('Image load failed!')
            return

        orb = cv2.ORB_create()

        keypoints1, desc1 = orb.detectAndCompute(src1, None)
        keypoints2, desc2 = orb.detectAndCompute(src2, None)

        matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        matches = matcher.match(desc1, desc2)

        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]



        pts1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
        pts2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)

        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

        (h, w) = src1.shape[:2]
        corners1 = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2).astype(np.float32)
        corners2 = cv2.perspectiveTransform(corners1, H)

        

        dst = cv2.polylines(src2, [np.int32(corners2)], True, (0, 255, 0), 2, cv2.LINE_AA)
        
        min = np.min(corners2)
        min1 = str(min)
        min2 = int(min)
        focal_lenth = 0.00026 #초점거리 26m 가정 
        h = 476/2 # 사진 픽셀 크기가 700 * 476 임 
        C = 1.5 #카메라 설치 높이 (직접측정)   
        y=h-min2
        setai=math.atan(y/focal_lenth)
        setac = 89 * 3.14 /180
        setax= setac-setai
        X=math.tan(setax)
        lenth=C*math.tan(X)
        l=str(lenth)
        

        # 폰트 색상 지정
        black = (0, 0, 0)
        # 폰트 지정
        font =  cv2.FONT_HERSHEY_PLAIN
 
        # 이미지에 글자 합성하기
        dst = cv2.putText(dst,l, (350, 40), font, 2, black, 1, cv2.LINE_AA)
 
        # 이미지 보여주고 창 끄기
        cv2.imshow('dst' , dst)
        cv2.imwrite('/Users/mba13/RB_distance_estimation/FrameExtraction/output_video/dst' +str(frame_no)+'.jpg',dst)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    
    


    if __name__ == '__main__':
        find_homography()


    frame_no = frame_no+10



 
