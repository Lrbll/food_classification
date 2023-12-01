"""
@author: Robert Kamunde
"""
# train : 학습을 위한 데이터셋
# test : 검증을 위한 데이터셋

# 전체 데이터셋을 train 폴더(학습용)와 test 폴더(검증용)로 분리하는 메서드
from shutil import copy
from collections import defaultdict
import os

# 데이터셋을 train셋과 test셋으로 전처리하는 함수
# 이미지를 복사하여 진행되므로 한번만 실행해주어야함 => 맨 밑 두줄에서 함수가 두번 처리되므로 출력메세지는 2번 나올 것임!!
def prepare_data(filepath, src,dest):
  classes_images = defaultdict(list)
  with open(filepath, 'r') as txt:
      paths = [read.strip() for read in txt.readlines()]
      for p in paths:
        food = p.split('/')
        classes_images[food[0]].append(food[1] + '.jpg')

  for food in classes_images.keys():
    print("\nCopying images into ",food)
    if not os.path.exists(os.path.join(dest,food)):
      os.makedirs(os.path.join(dest,food))
    for i in classes_images[food]:
      copy(os.path.join(src,food,i), os.path.join(dest,food,i))
  print("Copying Done!")

# 전처리할 데이터셋의 이름을 txt 파일로 관리하면,
# 학습용(train)과 검증용(test) 데이터셋으로 나누기 용이하고 데이터셋 폴더의 경로이동, 복사 등 관리가 쉬워짐

# prepare_data('전처리할 데이터셋 이름 txt 파일', '전체 데이터셋 경로', '전처리 후 저장될 데이터셋 경로')
prepare_data('C:/food_classification/archive/meta/meta/train2.txt', 'C:/food_classification/archive/images', 'C:/food_classification/archive/train')
prepare_data('C:/food_classification/archive/meta/meta/test2.txt', 'C:/food_classification/archive/images', 'C:/food_classification/archive/test')
   
