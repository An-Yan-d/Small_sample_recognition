import cv2
import os
import numpy
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import pickle
import time


K = 1024
lx = 384
ly = 180
dx = int(lx / 4)
dy = int(ly / 4)
obj_path='objects'
img_path='images'

sift = cv2.xfeatures2d.SIFT_create()


with open('kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

def is_intersect(ax1,ax2,ay1,ay2,bx1,bx2,by1,by2):
    if ax1 >= bx2 or ax2 <= bx1 or ay1 >= by2 or ay2 <= by1:
        return False
    else:
        return True

start_time=time.time()
for file in os.listdir(img_path):
    positions = []
    if not file.endswith('png'):
        continue
    print(file)
    ori = cv2.imread(os.path.join(img_path, file))
    x = 0
    x_ = lx
    while x_ <= 1920:
        y = 0
        y_ = ly
        while y_ <= 1080:
            img = ori[y:y_, x:x_]
            kp, desc = sift.detectAndCompute(img, None)
            if kp:
                predictions = kmeans.predict(list(desc))
                data = numpy.zeros(K)
                for i in predictions:
                    data[i] += 1
                if clf.predict([data])[0] == 0:
                    positions.append([x, x_, y, y_])

            y += dy
            y_ += dy
        x += dx
        x_ += dx

    print(len(positions))
    i = 0
    while i < len(positions):
        j = i + 1
        while j < len(positions):
            if is_intersect(*positions[i], *positions[j]):
                positions[i][1] = positions[j][1]
                positions[i][3] = positions[j][3]
                positions.pop(j)
            else:
                j+=1
        i+=1
    for p in positions:
        cv2.rectangle(ori, (p[0], p[2]), (p[1], p[3]), (0, 0, 255), 2)

    cv2.imwrite(os.path.join('output', file),ori)

end_time=time.time()
print((end_time-start_time)/13)