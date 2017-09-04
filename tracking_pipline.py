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
# scale = 1.5
# scale = 2
scale = 1

heat_image_total = []
def process_image(image):
    out_img = np.copy(image)
    boundingbox = []
    for s, color in [(1, (0,0,255)), (1.5, (0,255,0)), (2, (255,0,0))]:
        _, boundingbox_tmp = hog_subsample.find_cars(image, ystart, ystop, s, svc,
                                X_scaler, orient, pix_per_cell,
                                cell_per_block, spatial_size, hist_bins)
        boundingbox.extend(boundingbox_tmp)
        for b in boundingbox_tmp:
            cv2.rectangle(out_img, b[0], b[1], color, 6)

    heat_image = np.zeros((out_img.shape[0], out_img.shape[1]))
    heat_map.add_heat(heat_image, boundingbox)

    binary = np.zeros((image.shape[0], image.shape[1]))
    image_tl = np.copy(image)
    image_tr = out_img
    image_bl = np.dstack((binary, binary, binary))
    image_br = np.dstack((binary, binary, binary))

    heat_image_total.append(heat_image)
    if len(heat_image_total)>10:
        total = np.sum(heat_image_total, axis=0) / 10.0
        image_bl = np.dstack((np.copy(total), binary, binary)).astype(np.uint8)*5
        total = heat_map.apply_threshold(total, 8)
        image_br = np.dstack((total, binary, binary)).astype(np.uint8)*5
        heat_image_total.pop(0)
        image_tl = labeling.draw_labeled_bboxes(image, total).astype(np.uint8)

    top = cv2.hconcat([image_tl, image_tr]).astype(np.uint8)
    bottom = cv2.hconcat([image_bl, image_br]).astype(np.uint8)
    top_bottom = cv2.vconcat([top, bottom])
    return cv2.resize(top_bottom, None, fx=0.5, fy=0.5)


def video():
    from moviepy.editor import VideoFileClip
    white_output = 'project_video_output.mp4'
    # clip1 = VideoFileClip('test_video.mp4')
    clip1 = VideoFileClip('project_video.mp4')
    # white_clip = clip1.fl_image(process_image).subclip(0.9)
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