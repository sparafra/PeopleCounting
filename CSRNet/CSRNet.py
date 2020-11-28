import os
import cv2
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as c
from keras.models import model_from_json




class CSRNet:
    def __init__(self, ROOT_DIR):
        self.ROOT_DIR = ROOT_DIR
        self.model = self.load_model()



    def get_prediction(self, img):
        image = self.create_img(img)
        self.ans = self.model.predict(image)
        count = np.sum(self.ans)
        #plt.autoscale(True)
        plt.clf()
        #plt.imshow(image.reshape(image.shape[1], image.shape[2], image.shape[3]))

        # Set whitespace to 0
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        plt.imshow(self.ans.reshape(self.ans.shape[1], self.ans.shape[2]), cmap=c.jet)

        plt.axis('tight')
        plt.axis('off')
        #plt.draw()
        plt.savefig("testP.png")

        return int(count), image, self.ans


    def load_model(self):
        # Function to load and return neural network model
        json_file = open(os.path.join(self.ROOT_DIR,'models/Model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # loaded_model.load_weights("weights/model_A_weights.h5")
        # loaded_model.load_weights("weights/CSRNet_MAE8.31_MSE14.361_SFN0.0_MAPE0.066_epoch135-400.0.hdf5")
        loaded_model.load_weights(os.path.join(self.ROOT_DIR, "weights/CSRNet_MAE67.984_RMSE103.25_SFN0.838_MAPE0.173_epoch127-150.0.hdf5"))

        return loaded_model

    def create_img(self, path):
        # Function to load,normalize and return image
        print(path)
        im = Image.fromarray(path).convert('RGB')

        im = np.array(im)

        im = im / 255.0

        im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
        im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225

        im = np.expand_dims(im, axis=0)
        return im

