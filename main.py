from random import randint
import cv2
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from math import sqrt

#trenowanie
original_img = cv2.imread("healthy/01_h.jpg")
print (original_img.shape)

original_img = original_img[700:1400,700:1400,:]

manual_img_mask = cv2.imread("healthy_manualsegm/01_h.tif")
manual_img_mask = manual_img_mask[700:1400,700:1400,:]
print (np.sum(manual_img_mask))

offset = 20
samples_number = 100
train_x = np.zeros((samples_number, (2*offset)**2))
train_y = np.zeros((samples_number))

for i in range(0,samples_number):
    x = randint(0+offset, original_img.shape[0]-offset)
    y = randint(0+offset, original_img.shape[1]-offset)
    crop_img = original_img[x-offset:x+offset, y-offset:y+offset]
    flattened = crop_img[:,:,1].flatten()
    train_x[i,:] = flattened
    train_y[i] = manual_img_mask[x,y,1]/255
    
# clf = svm.SVC()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(offset, 2), random_state=1)
clf.fit(train_x, train_y)

#predykcja na calym obrazie
predict_img = cv2.imread("healthy/02_h.jpg")

manual_predict_img_mask = cv2.imread("healthy_manualsegm/02_h.tif")
manual_predict_img_mask = manual_predict_img_mask[700:1400,700:1400,:]

predict_img = predict_img[700:1400,700:1400,:]

predict_x = np.zeros(((len(predict_img[0])-2*offset)*(len(predict_img[1])-2*offset), (2*offset)**2))
# print("shape of you", ((len(predict_img[0])-2*offset), (len(predict_img[1])-2*offset)))
a = 0
for i in range(offset, predict_img.shape[0]-offset):
    for j in range(offset, predict_img.shape[1]-offset):
        predict_crop_img = predict_img[i-offset:i+offset, j-offset:j+offset]
        predict_x[a,:] = predict_crop_img[:,:,1].flatten()
        a=a+1


print ("predict_x", predict_x)

res = clf.predict(predict_x)
new_res = res.reshape((660, 660))
print ("new res", new_res)

cv2.imshow("cropped", new_res*255)
cv2.imshow("cropped2", manual_predict_img_mask)
cv2.waitKey(0)
