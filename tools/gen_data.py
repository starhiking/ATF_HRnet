import os
import argparse
import cv2
import numpy as np
import pandas as pd

import sys
sys.path.append('..')

nose_rect_index = {
    'aflw':[8,9,12,13,14],
    'wflw':[51,52,53,54,55,56,57,58,58,64,68]
}

def get_rect(index_array,landmarks,img_shape):
    """
    args:
        landmarks: (-1,2)
        index_array: [int,]

    return:
        left_top: (int,int)
        right_bottom : (int,int)
    """

    point_landmarks = np.array([landmarks[i] for i in index_array],dtype=np.float32)
    a0_max, a1_max = np.argmax(point_landmarks,axis=0)
    a0_min, a1_min = np.argmin(point_landmarks,axis=0)

    left_top_x = int(max(0,point_landmarks[a0_min][0]-np.random.randint(0,15)/1000.0 * img_shape[1]))
    left_top_y = int(max(0,point_landmarks[a1_min][1]-np.random.randint(0,15)/1000.0 * img_shape[0]))

    right_bottom_x = int(min(img_shape[1],point_landmarks[a0_max][0]+np.random.randint(0,15)/1000.0 * img_shape[1]))
    right_bottom_y = int(min(img_shape[0],point_landmarks[a1_max][1]+np.random.randint(0,15)/1000.0 * img_shape[0]))

    return (left_top_x,left_top_y),(right_bottom_x,right_bottom_y)
    # return (point_landmarks[a0_min][0],point_landmarks[a1_min][1]),(point_landmarks[a0_max][0],point_landmarks[a1_max][1])

def create_folder(str_path):
    paths = str_path.split('/')
    temp_folder = paths[0]
    for i in range (len(paths)-2):
        temp_folder = os.path.join(temp_folder,paths[i+1])
        if not os.path.exists(temp_folder):
            print("{} not exist , created.".format(temp_folder))
            os.mkdir(temp_folder)
    if os.path.exists(str_path):
        print("{} exist more than one face.".format(str_path))   

def parse_args():
    parser = argparse.ArgumentParser(description='Gen Nose image dataset')

    parser.add_argument('--Dataset', help='experiment dataset name',
                        type=str, default="wflw")
    parser.add_argument('--Train_type', help='experiment dataset type',
                        type=str, default="test")
    parser.add_argument('--Gen_folder', help='generate dataset folder',
                        type=str, default="nose_images")

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    csv_file = os.path.join('data',args.Dataset,"face_landmarks_{}_{}.csv".format(args.Dataset,args.Train_type))
    image_folder = os.path.join('data',args.Dataset,"images")
    Gen_folder = os.path.join('data',args.Dataset,args.Gen_folder)
    if not os.path.exists(Gen_folder):
        os.mkdir(Gen_folder)
        print("Create folder : {}".format(Gen_folder))


    landmark_frame = pd.read_csv(csv_file)
    for i in range(landmark_frame.shape[0]):
        image_name = landmark_frame.iloc[i, 0]
        image_path = os.path.join(image_folder,image_name)
        gen_image_path = os.path.join(Gen_folder,image_name)
        # print(image_path)
        # print("Process image to {}".format(gen_image_path))
        img = cv2.imread(image_path)

        if os.path.exists(gen_image_path):
            print("{} have processed one face, add face inplace.".format(gen_image_path))
            nose_img = cv2.imread(gen_image_path)
        else:
            nose_img = np.zeros(img.shape,dtype=np.int)

        if args.Dataset == 'wflw':
            landmarks = landmark_frame.iloc[i,4:].values
        else :
            landmarks = landmark_frame.iloc[i,5:].values

        landmarks = landmarks.astype('int').reshape(-1, 2)
        
        left_top, right_bottom = get_rect(nose_rect_index[args.Dataset],landmarks,img.shape)
        
        nose_img[left_top[1]:right_bottom[1],left_top[0]:right_bottom[0]] = img[left_top[1]:right_bottom[1],left_top[0]:right_bottom[0]]

        # cv2.rectangle(nose_img,left_top,right_bottom,(0,255,255),1)
        # for l_i in range(landmarks.shape[0]) : 
        #     cv2.circle(nose_img,(landmarks[l_i][0],landmarks[l_i][1]),1,(0,255,255),1,1)
        # #     cv2.putText(nose_img,str(l_i),(landmarks[l_i][0],landmarks[l_i][1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        
        create_folder(gen_image_path)
        cv2.imwrite(gen_image_path,nose_img)
        # exit()
        
    # print(image_folder)


if __name__=="__main__":
    main()

