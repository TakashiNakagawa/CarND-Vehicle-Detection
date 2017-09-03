import learning_image
import hog_subsample
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import *
import heat_map
import labeling

def learning():
    svc, X_scaler = learning_image.svm_fit()
    learning_image.save_svc(svc, X_scaler)

heat_image_total = []
def process_image(image):
    dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    color_space = dist_pickle["color_space"]



    ystart = 350
    ystop = 656
    scale = 1.5

    out_img, boundingbox = hog_subsample.find_cars(image, ystart, ystop, scale, svc,
                            X_scaler, orient, pix_per_cell,
                            cell_per_block, spatial_size, hist_bins)

    # return out_img

    heat_image = np.zeros((out_img.shape[0], out_img.shape[1]))
    heat_map.add_heat(heat_image, boundingbox)

    heat_image_total.append(heat_image)
    if len(heat_image_total)>10:
        total = np.sum(heat_image_total, axis=0) / 10.0
        total = heat_map.apply_threshold(total, 4)
        # total = total.astype(np.uint8) * 10
        heat_image_total.pop(0)
        return labeling.draw_labeled_bboxes(image, total)


        # binary = np.zeros((total.shape[0], total.shape[1]))
        # result = np.dstack((total, binary, binary))
        # heat_image_total.pop(0)
        #
        #
        #
        # return result
    else:
        heat_image = heat_map.apply_threshold(heat_image, 4)
        heat_image = heat_image.astype(np.uint8)*10
        binary = np.zeros((heat_image.shape[0], heat_image.shape[1]))
        result = np.dstack((heat_image, binary, binary))
        # bb = cv2.cvtColor(aa, cv2.COLOR_GRAY2RGB)
        # plt.imshow(result)
        # plt.show()
        return result
        # return out_img


def video():
    from moviepy.editor import VideoFileClip
    white_output = 'project_video_output.mp4'
    # clip1 = VideoFileClip('test_video.mp4')
    clip1 = VideoFileClip('project_video.mp4')
    # white_clip = clip1.fl_image(process_image).subclip(49,50)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)

if __name__ == '__main__':
    # learning()
    # img = cv2.imread('./test_images/test6.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # process_image(img)
    video()
    # 事前に学習を行う

    # 画像を一枚取得する

    # 車領域を見つける

    # 過去複数フレームでも一致するかを確認

    # 確定したら実際に画像に書き出して動画を作成