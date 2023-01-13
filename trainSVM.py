import cv2
import os
import numpy
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import pickle


K = 1024
obj_path='objects'
img_path='images'

sift = cv2.xfeatures2d.SIFT_create()
with open('kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)
datas = []
labels = []
for file in os.listdir(obj_path):
    if not file.endswith('png'):
        continue
    img = cv2.imread(os.path.join(obj_path, file))
    # SIFT
    kp, desc = sift.detectAndCompute(img, None)
    predictions = kmeans.predict(list(desc))
    data = numpy.zeros(K)
    for i in predictions:
        data[i] += 1
    datas.append(data)
    if file.startswith('主板'):
        labels.append(0)
    else:
        labels.append(1)

clf = SVC(kernel='linear')
clf.fit(datas, labels)

with open('clf.pkl', 'wb') as f:
    pickle.dump(clf, f)
print('SVM OK')
