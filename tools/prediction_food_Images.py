"""
@author: Robert Kamunde
"""
from keras import utils
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model

# 전체 데이터셋 폴더에 저장된 음식이름(음식폴더명)을 리스트에 저장 => 현재, ramen, pho, bibimbap, sushi, sashimi 5개 폴더 존재
def create_foodlist(path):
    list_ = list()
    for root, dirs, files in os.walk(path, topdown=False):
      for name in dirs:
        list_.append(name)
    return list_    

# 학습시킨 모델과 모든 음식이름리스트 불러오기      
my_model = load_model('model_trained.h5', compile = False)
food_list = create_foodlist("archive/images")

# 내 컴퓨터에 저장된 새로운 이미지를 불러오고 어떤 이미지인지 예측하는 함수 
def predict_class(model, images, show = True):
  for img in images:
    img = utils.load_img(img, grayscale=False, color_mode='rgb', target_size=(299, 299))

    img = utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      

    pred = model.predict(img)
    index = np.argmax(pred)    #Returns the indices of the maximum values along an axis, In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
    food_list.sort()
    pred_value = food_list[index]
    if show:
        plt.imshow(img[0])                           
        plt.axis('off')
        plt.title(pred_value)
        plt.show()

# 예측할 이미지를 root파일 바로 밑에 저장 후, images.append('예측할 이미지 파일명.확장자')
images = []
images.append('aa.jpg')
images.append('bb.jpg')
images.append('cc.jpg')
images.append('dd.jpg')
images.append('ee.jpg')



print("PREDICTIONS BASED ON PICTURES UPLOADED")
predict_class(my_model, images, True)

