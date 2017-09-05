import matplotlib.pyplot as plt
from skimage.feature import hog
import cv2
import glob
import numpy as np


def extrac_hog(img_file, orient=9,
               pix_per_cell=8, cell_per_block=2):
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    image = image.astype(np.float32) / 255
    hog_images = []
    for channel in range(image.shape[2]):
        gray = image[:, :, channel]
        features, hog_image = hog(gray, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), visualise=True,
                                  feature_vector=False)
        hog_images.append((gray, hog_image))
    return hog_images


cars = glob.glob("../vehicles/*/*.png")
notcars = glob.glob("../non-vehicles/*/*.png")

np.random.seed(100)
car = np.random.choice(cars)
notcar = np.random.choice(notcars)


for i, gh in enumerate(extrac_hog(car)):
    plt.subplot(3,2,i*2+1)
    plt.imshow(gh[0], cmap='gray')
    plt.subplot(3,2,i*2+2)
    plt.imshow(gh[1], cmap='gray')

plt.show()

for i, gh in enumerate(extrac_hog(notcar)):
    plt.subplot(3,2,i*2+1)
    plt.imshow(gh[0], cmap='gray')
    plt.subplot(3,2,i*2+2)
    plt.imshow(gh[1], cmap='gray')

plt.show()