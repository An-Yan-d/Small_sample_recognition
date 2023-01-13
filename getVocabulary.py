import cv2
import os
import numpy
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import pickle

K = 1024
features = []
sift = cv2.xfeatures2d.SIFT_create()

obj_path='objects'
img_path='images'

print(os.listdir(obj_path))
for file in os.listdir(obj_path):
    print(file)
    if not file.endswith('png'):
        continue
    img = cv2.imread(os.path.join(obj_path, file))
    # SIFT
    kp, desc = sift.detectAndCompute(img, None)
    features += list(desc)

kmeans = KMeans(n_clusters=K)
kmeans.fit(features)
with open('kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
print('vocabulary OK')


