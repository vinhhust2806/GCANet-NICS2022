from utils.model import model
from utils.data import *
from train import IMAGE_SIZE
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from PIL import Image

model.load_weights('a_seg.hdf5')
test_gen = train_generator(data_test,1,
                                dict(),
                                target_size=IMAGE_SIZE)
results = model.evaluate(test_gen, steps=len(data_test))

print("Test lost: ",results[0])
print("Test Dice Coefficent: ",results[1])
print("Test IOU: ",results[2])

for i in range(len(data_test)):
    img = Image.open(data_test['image_path'].iloc[i])
    img = img.resize((256,256))
    img = np.array(img) / 255.
    img = img.reshape(1,256,256,3)
    pred=model.predict(img)
    plt.axis('off')
    plt.figure(figsize=(12,12))
    plt.subplot(1,3,1)
    plt.imshow(np.squeeze(img))
    plt.title('Original Image')
    plt.subplot(1,3,2)
    plt.imshow(np.squeeze(cv2.resize(cv2.imread(data_test['mask_path'].iloc[i]),(256,256))))
    plt.title('Original Mask')
    plt.subplot(1,3,3)
    plt.imshow(np.squeeze(pred) > .5,'gray')
    plt.title('Prediction')
    plt.show()
