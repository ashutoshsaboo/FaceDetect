import cv2
import sys
import numpy as np
from sklearn import svm
from sklearn import linear_model
# from sklearn.neural_network import MLPClassifier

l = sys.argv
l.pop(0)

# cascPath = l[-1]
# l.pop(-1)

#im = []
# print type(l[0])
with open(l[0]) as f:
    content = f.readlines()
l.pop(0)

# print type(content)
# raw_input()
m = 112
n = 92
mn = m * n
faces_count = 40
train_faces_count = 8
t = train_faces_count * faces_count # training images count
L = np.empty(shape=(mn, t), dtype='float64')

cur_img = 0

# Y = np.empty(shape=(1, t), dtype='int64')
X = np.empty(shape=(t, mn), dtype='float64')
Y= []
# X = []
for imagePath in content:
    imagePath="".join(imagePath.split())

#    faceCascade = cv2.CascadeClassifier(cascPath)
    image = cv2.imread(imagePath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    Y.append(int((imagePath.split("/")[1])[1:]))
#    height, width = gray.shape[:2]
#    print height, width

#    im.append(gray)
    img_col = np.array(gray, dtype='float64').flatten()
    #print type(X[cur_img]) , cur_img
    #print X , X[0] , X[cur_img]
    # input()
    # X.append(img_col)
    # X.append(img_col)
    X[cur_img, :] = img_col[:]
    L[:, cur_img] = img_col[:]
    cur_img += 1
# for i in Y:
#     print i[0]
#     # input()
#     i = int(i[0])
# print Y[0].transpose()
# print "g"
# t = Y[0].transpose()
# t.reshape(Y[0].transpose().shape[0], 1)
# print Y[0].transpose().shape[0]
# print t.shape
# Y=t

#    cv2.imshow('Image', gray)
# X = L.transpose()

#    cv2.waitKey(0)
# print Y , type(Y) , type(Y[0])
# input()

mean_img_column = np.sum(L, axis=1) / t
for i in xrange(0,t):
    L[:,i] -= mean_img_column[:]
    # X.append(list(L[:,i]))
    # input()

#print L
# print "hi"
Covariance_Mat = ( np.matrix(L.transpose()) * np.matrix(L) )/t
eigenvals , eigenvects= np.linalg.eig(Covariance_Mat)

# print type((eigenvals[0]))
# print Covariance_Mat.shape
# print eigenvects.shape
# print L.shape
#print eigenvals[0],eigenvals[1], eigenvals[2]
sorted_indices = eigenvals.argsort()[::-1]
eigenvals = eigenvals[sorted_indices]
eigenvects = eigenvects[sorted_indices]

eigenvects = L * (eigenvects.transpose())

norms = np.linalg.norm(eigenvects, axis=0)
eigenvects = eigenvects / norms
weights = eigenvects.transpose() * L
temp = eigenvects.shape[1]
Y = np.array(Y).reshape((1, -1)).transpose()
# X = np.array(X).reshape((1, -1))
print X.shape
print Y.shape

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X, Y)
clf.predict(X[1])
# clf = linear_model.LogisticRegressionCV()
# clf.fit(X,Y)

# clf = svm.SVC(decision_function_shape='ovo')
# clf.fit(X, Y)
# dec = clf.decision_function([[1]])
# print dec.shape[1]
# clf.decision_function_shape = "ovr"
# dec = clf.decision_function([[1]])
# print dec.shape[1]

# clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X, y)

# input()
# for i in xrange(0,temp):
    # print eigenvects[:,i].shape
    # cv2.imshow(eigenvects[:,i])

# eigenvects1 = np.reshape(eigenvects[:,0], (mn,1))

# cv2.imshow("eigenvector",eigenvects1)
# cv2.waitKey(0)
# input("SDF")

# print l

# with open(l[0]) as f:
#     content = f.readlines()

# corr = 0.0
# incorr = 0.0
# for imagePath in content:
#     imagePath="".join(imagePath.split())
#     image = cv2.imread(imagePath)

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)
#     test_col = np.array(gray, dtype='float64').flatten()
#     # test_col -= mean_img_column
#     test_col = np.reshape(test_col, (mn,1))

#     test_weights = eigenvects.transpose() * test_col

# #    print test_weights.transpose()
#     # input()
#     diff = weights - test_weights
#     test_norms = np.linalg.norm(diff, axis=0)
#     closest_face_id = np.argmin(test_norms)
#     predicted_person_id = int((closest_face_id / train_faces_count ) + 1)
#     actual_person_id = int((imagePath.split("/")[1])[1:])
#     # if predicted_person_id == actual_person_id:
    #     corr +=1
    # else:
    #     incorr+=1
    # print "Predicted Person: " , predicted_person_id , "Actual Person: " , actual_person_id
# accuracy = (corr / (corr + incorr))*100
# print "Correct Cases = " , corr , "Incorrect Cases = " , incorr , "Accuracy in %= ", accuracy
#print eigenvals[0],eigenvals[1], eigenvals[2]
# print 

# for i in xrange(0,eigenvects.shape[1]):
#     cv2.imshow(eigenvects[:,i])
#     cv2.waitKey(0)
#print eigenvects.shape
#    faces = faceCascade.detectMultiScale(gray, 1.01, 5)
    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.05,
    #     minNeighbors=5,
    #     minSize=(70, 70),
    # #    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    #     flags=0
    # )

    # Draw a rectangle around the faces
#     for (x, y, w, h) in faces:
#          #image = image[y-10:y+h+10,x-10:x+w+10]
# #        gray = gray[y-10:y+h+10,x-10:x+w+10]
# #        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     #   
#         height, width = gray.shape[:2]
#     #    print height, width
# #        resized_image = cv2.resize(gray, (80, 80)) 
#         gray = cv2.equalizeHist(gray)
# #        gray = gray[y-10:y+h+10,x-10:x+w+10]        
#         im.append(gray)

    #for image in im:
    #    image = 
    # Display the resulting frame
