# # Importing libraries
# import cv2
# from cvzone.SelfiSegmentationModule import SelfiSegmentation
# # Reading the green screen and background image
# green_screen_img = cv2.imread("Green Screen Image Path")
# bg_img = cv2.imread("Background Image Path")
#
# # Checking dimensions
# print(green_screen_img.shape)
# print(bg_img.shape)
# # In my case both are having different dimension.
# # So, I have to resize my images to same dimensions.
# w, h = 640, 480
# green_screen_img = cv2.resize(green_screen_img, (w, h))
# bg_img = cv2.resize(bg_img, (w, h))
#
# output_1 = segmentor.removeBG(green_screen_img, bg_img)
# cv2.imshow("Output-1", output)
# cv2.waitKey(0)
#
# output_2 = segmentor.removeBG(green_screen_img, bg_img, threshold=0.4)
# cv2.imshow("Output-2", output_2)
# cv2.waitKey(0)
#
#
# output_3 = segmentor.removeBG(green_screen_img, bg_img, threshold=1)
# cv2.imshow("Output-3", output_3)
# cv2.waitKey(0)
#
#
# # In my case, I am choosing Red color as background
# # RED - (0, 0, 255)
# output = segmentor.removeBG(green_screen_img, (0, 0, 255))
# cv2.imshow("Output", output)
# cv2.waitKey(0)


import cv2
import numpy as np
import os

# bg_folder = '/home/yygx/sun_dataset/images/b/bedroom/'
bg_folder = '/home/yygx/sun_dataset/images/total/'
hands_folder = "dataset_10000_256_left_right_no_wrists/rendered_hand/"
dst_folder = "dataset_10000_256_left_right_no_wrists/dataset_change_bg/"

bg_imgs = os.listdir(bg_folder)
hands_imgs = os.listdir(hands_folder)


cnt_hands = 0
cnt_bg = 0
while cnt_hands < len(hands_imgs):
# for i in range(865, len(hands_imgs)):
    if cnt_hands % 1000 == 0:
        print(cnt_hands)
    frame = cv2.imread(hands_folder + hands_imgs[cnt_hands])
    image = cv2.imread(bg_folder + bg_imgs[cnt_bg])

    cnt_bg += 1
    if image is None:
        print("image is None")
        continue

    image_size_shortest = np.min([image.shape[0], image.shape[1]])
    image = image[:image_size_shortest, :image_size_shortest, :]
    image = cv2.resize(image, (256, 256))

    u_green = np.array([19, 235, 34])
    l_green = np.array([17, 233, 32])

    mask = cv2.inRange(frame, l_green, u_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    f = frame - res
    f = np.where(f == 0, image, f)

    cv2.imwrite(dst_folder + hands_imgs[cnt_hands], f)

    cnt_hands += 1

#
#
# frame = cv2.imread("dataset_10000_more_angles_green/rendered_hand/000.png")
# image = cv2.imread("/home/yygx/sun_dataset/images/b/bedroom/labelme_dqiqpvxsosokefd.jpg")
#
#
# image_size_shortest = np.min([image.shape[0], image.shape[1]])
# image = image[:image_size_shortest, :image_size_shortest, :]
# image = cv2.resize(image, (512, 512))
#
# u_green = np.array([19, 235, 34])
# l_green = np.array([17, 233, 32])
#
# mask = cv2.inRange(frame, l_green, u_green)
# res = cv2.bitwise_and(frame, frame, mask=mask)
#
# f = frame - res
# f = np.where(f == 0, image, f)
#
# cv2.imwrite("dataset_change_bg/img.png", frame)
# cv2.imwrite("dataset_change_bg/mask.png", f)
