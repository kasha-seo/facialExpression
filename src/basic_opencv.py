import cv2
from matplotlib import pyplot as plt

'''
opencv를 이용한 간단한 얼굴 및 몸 인식
참조 링크 : https://pinkwink.kr/1124
'''

# path = '../lib/python3.6/site-packages/cv2/data/'
path = 'dependency/'
face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')

image = cv2.imread('./images/daesom.jpeg')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12,8))
plt.imshow(grayImage, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

faces = face_cascade.detectMultiScale(grayImage, 1.3, 5)

print(faces.shape)
print("Number of faces detected: " + str(faces.shape[0]))

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.rectangle(image, ((0,image.shape[0] -25)), (270, image.shape[0]), (255,255,255), -1);
cv2.putText(image, "KaSha test", (0,image.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1);

plt.figure(figsize=(12,12))
plt.imshow(image, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.rectangle(image, ((0,image.shape[0] -25)), (270, image.shape[0]), (255,255,255), -1);
cv2.putText(image, "KaSha test", (0,image.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1);

plt.figure(figsize=(12,12))
# plt.imshow(image, cmap='gray')
plt.imshow(image[y:y+h, x:x+w], cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()