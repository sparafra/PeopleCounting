import cv2
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as c
from keras.models import model_from_json


def load_model():
    # Function to load and return neural network model
    json_file = open('models/Model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #loaded_model.load_weights("weights/model_A_weights.h5")
    #loaded_model.load_weights("weights/CSRNet_MAE8.31_MSE14.361_SFN0.0_MAPE0.066_epoch135-400.0.hdf5")
    loaded_model.load_weights("weights/CSRNet_MAE67.984_RMSE103.25_SFN0.838_MAPE0.173_epoch127-150.0.hdf5")

    return loaded_model


def create_img(path):
    # Function to load,normalize and return image
    print(path)
    im = Image.open(path).convert('RGB')

    im = np.array(im)

    im = im / 255.0

    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225

    im = np.expand_dims(im, axis=0)
    return im

def predict(path):
    #Function to load image,predict heat map, generate count and return (count , image , heat map)
    model = load_model()
    image = create_img(path)
    ans = model.predict(image)
    count = np.sum(ans)

    return count,image,ans


ans,img,hmap = predict('test_images/B/IMG_309.jpg')

print(ans)
#Print count, image, heat map
plt.imshow(img.reshape(img.shape[1],img.shape[2],img.shape[3]))
plt.show()

plt.imshow(hmap.reshape(hmap.shape[1],hmap.shape[2]) , cmap = c.jet )
plt.savefig("testP.png")
plt.show()

temp = h5py.File('data/ShanghaiTech/part_B/test_data/ground/IMG_309.h5' , 'r')
temp_1 = np.asarray(temp['density'])
#plt.imshow(temp_1,cmap = c.jet)
print("Original Count : ",int(np.sum(temp_1)) + 1)