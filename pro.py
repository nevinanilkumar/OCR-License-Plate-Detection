import cv2
import numpy as np
import imutils
import pytesseract
from matplotlib import pyplot as plt
from PIL import Image
import string
table = str.maketrans('', '', string.ascii_lowercase)

def show_img(img):
  fig = plt.gcf()
  fig.set_size_inches(16, 8)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.show()

img = cv2.imread('/home/nevin/Desktop/project/testimg/7.jpeg')
(H, W) = img.shape[:2]
print(H, W)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(blur, 30, 200) 
conts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
conts = imutils.grab_contours(conts) 
conts = sorted(conts, key=cv2.contourArea, reverse=True)[:8] 
location = None
for c in conts:
    peri = cv2.arcLength(c, True)
    aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
    if cv2.isContourConvex(aprox):
      if len(aprox) == 4:
          location = aprox
          break
mask = np.zeros(gray.shape, np.uint8) 
img_plate = cv2.drawContours(mask, [location], 0, 255, -1)
img_plate = cv2.bitwise_and(img, img, mask=mask)

(y, x) = np.where(mask==255)
(beginX, beginY) = (np.min(x), np.min(y))
(endX, endY) = (np.max(x), np.max(y))

plate = gray[beginY:endY, beginX:endX]
plate=cv2.resize(plate, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
plate=cv2.GaussianBlur(plate, (5, 5), 0)


im=Image.fromarray(plate)

im.show()

config_tesseract = "--tessdata-dir tessdata --psm 6"

text = pytesseract.image_to_string(plate, lang="eng", config=config_tesseract)
print(text)

text="".join(ch for ch in text if ch.isalnum())
print("special characters removed: " + text)
text=text.translate(table)

def correctToText(text):
    for i in range(len(text)):
      if text[i]=="5":
         text=text=text.replace("5","S")
      elif text[i]=="2":
         text=text.replace("2","Z")
      elif text[i]=="4":
         text=text.replace("4","A")
      elif text[i]=="8":
         text=text.replace("8","B")
      elif text[i]=="7":
         text=text.replace("7","Z")
    return text
def correctToNumbers(text):
    for i in range(len(text)):
      if text[i]=="S":
         text=text.replace("S","5")
      elif text[i]=="Z":
         text=text.replace("Z","2")
      elif text[i]=="A":
         text=text.replace("A","4")
      elif text[i]=="B":
         text=text.replace("B","8")
      elif text[i]=="I":
         text=text.replace("I","1")
      elif text[i]=="O":
         text=text.replace("O","0")
      elif text[i]=="T":
         text=text.replace("T","1")
    return text
def checkState(text):
   list_of_states=["AN","AP","AR","AS","BR","CH","DN","DD","DL","GA","GJ","HR","HP","JK","KA","KL","LD","MP","MH","MN","ML","MZ","NL","OR","PY","PN","RJ","SK","TN","TR","UP","WB"]
   if text[0:2] in list_of_states:
      return text
   else:
      return correctToText(text[0:2])+text[2:]
def checkRTO(text):
   if not(text[2].isnumeric() and text[3].isnumeric()):
      return text[0:2]+correctToNumbers(text[2:4])+text[4:]
   else:
      return text
def checkLastFour(text):
   if not ((text[-1].isnumeric() and text[-2].isnumeric()) and (text[-3].isnumeric() and text[-4].isnumeric())):
      return text[:-4]+correctToNumbers(text[-4:])
   else:
      return text
def checkSeries(text):
    isthereanum=False
    for i in text[4:][:-4]:
        if i.isnumeric():
            isthereanum=True
    if isthereanum:
       return text[0:4]+correctToText(text[4:][:-4])+text[-4:]
    else:
       return text

print(checkSeries(checkLastFour(checkRTO(checkState(text)))))
