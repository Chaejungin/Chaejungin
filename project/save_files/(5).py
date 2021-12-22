
import imageio
import os
from PIL import Image
import natsort

directory = 'style-transfer'
image_type = '.jpg'
gif_name = 'output'
speed_sec = { 'duration': 0.1 } #사진 넘기는 시간

images = []

# 이미지 이름 순서대로 만들려고 sorting
file_list = natsort.natsorted(os.listdir(directory))

for file in file_list:
    if file.endswith(image_type) :
        file_path = os.path.join(directory, file)
        images.append(imageio.imread(directory))
    
imageio.mimsave('{0}/{1}.gif'.format(directory, gif_name), images, **speed_sec)

cv2.imshow('/Users/mba13/RB_distance_estimation/FrameExtraction/output_video/output.gif', img2)