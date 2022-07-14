import cv2
import numpy as np
from matplotlib import pyplot as plt
# from myFun import HSV

#### Global variable
showPoint = False   ## Points to fit image with each other
draw = False     
seeToEdit = False  ## Intermediate figures 
Shift =60  ## shift to increase accuracy
Flip =0  ## if image need to flip first

### Intialize
i=0
lowerA =np.array([])
upperA =np.array([])
def nothing(x):  ## For trackBar
    pass

###############################                Functions                     #######################################
####################################################################################################################
### 1- filters
def errosion(img):
    src=img
    dilatation_size =2
    dilation_shape = cv2.MORPH_RECT
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    out = cv2.erode(src, element)
    return out

def dilation(img,n):
    src=img
    dilatation_size =n
    dilation_shape = cv2.MORPH_RECT
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    out = cv2.dilate(src, element)
    #cv.imshow("her", dilatation_dst)
    #cv.imwrite("dial2.jpg",dilatation_dst)
    return out
def morph(img):
    op= cv2.MORPH_OPEN
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, None)
    return opening


### 2- Get color in image
def HSV(img,waitTime=30):
    global i,lowerA,upperA
    # create trackbars for color change
    print("i= ",i)
    if i==0:
        cv2.namedWindow('image')
        cv2.createTrackbar('HMin', 'image', 0, 179, nothing)  # Hue is from 0-179 for Opencv
        cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
        cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
        cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
        cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
        cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

        cv2.setTrackbarPos('HMax', 'image', 179)
        cv2.setTrackbarPos('SMax', 'image', 255)
        cv2.setTrackbarPos('VMax', 'image', 255)

        img = cv2.resize(img, (640, 480))
        output = img

        while (1):
            # get current positions of all trackbars
            hMin = cv2.getTrackbarPos('HMin', 'image')
            sMin = cv2.getTrackbarPos('SMin', 'image')
            vMin = cv2.getTrackbarPos('VMin', 'image')

            hMax = cv2.getTrackbarPos('HMax', 'image')
            sMax = cv2.getTrackbarPos('SMax', 'image')
            vMax = cv2.getTrackbarPos('VMax', 'image')

            # Set minimum and max HSV values to display

            lower = np.array([hMin, sMin, vMin], dtype="uint8")
            upper = np.array([hMax, sMax, vMax], dtype="uint8")

            print(lower,upper)
            lowerA = lower
            upperA = upper

            # Create HSV Image and threshold into a range.
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove noice

            # cv2.imshow("MaskinHsv",hsv)
            # cv2.waitKey()

            output = cv2.bitwise_and(img, img, mask=mask)

            # Display output image
            cv2.imshow('image1111', output)
            cv2.imshow('image', img)
            # Wait longer to prevent freeze for videos.
            if cv2.waitKey(waitTime) & 0xFF == 27:
                break
    else:
        lower = lowerA
        upper = upperA
    i+=1
    return lower,upper


####   Find area of contour and take the max one
def findArea(image_):
    param =1
    original = image_.copy()

    # if param != 0:
    image = cv2.cvtColor(image_, cv2.COLOR_BGR2HSV)
    lowerArray = [0, 16, 157]
    upperArray = [179, 255, 255]


    if param !=0:
        lower = np.array(lowerArray, dtype="uint8")
        upper = np.array(upperArray, dtype="uint8")
    else:
        lower=lowerArray
        upper=upperArray
    mask = cv2.inRange(image, lower, upper)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove noice

    output = cv2.bitwise_and(image, image, mask=mask)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    M = 0
    arrayArea = []
    for c in cnts:
        area = cv2.contourArea(c)
        arrayArea.append(area)
        M += 1

    if M == 0:
        print("NO")
        return 0, 0, original,original
    else:
        maxArea = np.max(arrayArea)
        print("Max Area= ", maxArea)
        output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
        return maxArea, arrayArea.index(maxArea), cnts,output


######## 3-Fit operation
def operation(image):
    maxArea, index, cnts ,Maskreq= findArea(image)
    cntMax = cnts[index]
    if draw ==True:
        cv2.drawContours(image, cntMax, -1, (0, 0, 255), 3)
    maxPoint = cntMax.max(axis=0)[0]
    minPoint = cntMax.min(axis=0)[0]
    xMin,yMin = minPoint[0] , minPoint[1]+Shift
    xMax ,yMax =maxPoint[0] , maxPoint[1]

    points = [[xMin, yMin], [xMin, yMax], [xMax, yMax], [xMax, yMin]]
    if showPoint ==True:
        cv2.circle(image,(xMin,yMin),10,(0,0,255),-1)
        cv2.circle(image,(xMin, yMax),10,(0,255,255),-1)
        cv2.circle(image,(xMax, yMin),10,(0,0,155),-1)
        cv2.circle(image,(xMax,yMax),10,(0,0,255),-1)

    return points , maxArea,Maskreq

def getWarp(img1,pts1,pts2):
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    # imgHeight,imgWidth,Colo = img2.shape
    imgWidth , imgHeight = 640,480
    matrix =cv2.getPerspectiveTransform(pts1,pts2)

    imgOutput = cv2.warpPerspective(img1,matrix,(imgWidth,imgHeight))

    return imgOutput


### 4- Resizing
def resizing(image1,image2):
    points1,_,trainMask= operation(image1)
    points2,_,QueryMask= operation(image2)

    train = np.zeros((640, 480))
    train = image1

    outPut = cv2.resize(image2, None, fx=(points1[2][0] - points1[0][0]) / (points2[2][0] - points2[0][0])
                        , fy=(points1[2][1] - points1[0][1]) / (points2[2][1] - points2[0][1]))
    points3, _ ,QueryMask= operation(outPut)
    # points3 = [[128, 346], [546, 344], [546, 58], [128, 58]]
    if points1[0][1] < points3[0][1]:
        print("here1")
        points3[0][1] = points3[0][1]-Shift
        points3[3][1] = points3[3][1]-Shift
    imgWarped = getWarp(outPut, points3, points1)
    return train,imgWarped,trainMask,QueryMask

####################################################################################################################


############    Read images
nameOfSaveimg1 ="AfterFit1"
nameOfSaveimg2 ="AfterFit2"

image1= cv2.imread("reference.jpg")
image1 = cv2.resize(image1,(640,480))

image2 = cv2.imread("image.jpg")
image2 = cv2.resize(image2,(640,480))


########### To filp image to fit reffernce
if Flip ==1:
    image2 = cv2.flip(image2, 1)

##########  To fit image with Reff
train , Query,trainMask2,QueryMask2 = resizing(image1,image2)

_,_,_,trainMask =findArea(train)
_,_,_,QueryMask =findArea(Query)

cv2.imwrite(nameOfSaveimg1+".jpg", train)
cv2.imwrite(nameOfSaveimg2+".jpg", Query)

####Multi show
fig = plt.figure(figsize=(10, 7))
rows ,columns =2,2

###First
fig.add_subplot(rows, columns, 1)

plt.imshow(train)
plt.title("Query")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)

plt.imshow(image1)
plt.title("First")

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)

plt.imshow(Query)
plt.title("Reference")
# Adds a subplot at the 4th position

fig.add_subplot(rows, columns, 4)

plt.imshow(image2)
plt.title("Second")

plt.show()

# cv2.waitKey()
# #################################################
# #################################################
# #################################################



main = image1
main_c = image1

stnd_pic = trainMask
live_pic = QueryMask
cv2.imwrite("Train.jpg",trainMask)
cv2.imwrite("Query.jpg",QueryMask)

stnd_c = stnd_pic.copy()
live_c = live_pic.copy()

# smoothing
##1st smoother
stnd_pic = cv2.bilateralFilter(stnd_pic, 9, 75, 75)
live_pic = cv2.bilateralFilter(live_pic, 9, 75, 75)
# 2nd smoother
stnd_pic = cv2.GaussianBlur(stnd_pic, (5, 5), 1)
live_pic = cv2.GaussianBlur(live_pic, (5, 5), 1)

# splitting live_pic to b,g,r
Bs, Gs, Rs = cv2.split(stnd_pic)
Bl, Gl, Rl = cv2.split(live_pic)

# convert from BGR to HSV
stnd_pic = cv2.cvtColor(stnd_pic, cv2.COLOR_BGR2HSV)
live_pic = cv2.cvtColor(live_pic, cv2.COLOR_BGR2HSV)

# splitting live_pic to h,s,v
Hs, Ss, Vs = cv2.split(stnd_pic)
Hl, Sl, Vl = cv2.split(live_pic)

# tracbar creation
## empty func used with tracbar

##creat tracbar window
cv2.namedWindow('Color Track Bar')
##create tracbar
cv2.createTrackbar("MaxG", "Color Track Bar", 0, 255, nothing)
cv2.createTrackbar("MinG", "Color Track Bar", 0, 255, nothing)
cv2.createTrackbar("MaxS", "Color Track Bar", 0, 255, nothing)
cv2.createTrackbar("MinS", "Color Track Bar", 0, 255, nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('MinG', 'Color Track Bar', 255)
cv2.setTrackbarPos('MinS', 'Color Track Bar', 255)

##getting tracbar position to contron thresh values
while (True):
    G2_l = cv2.getTrackbarPos("MaxG", "Color Track Bar")
    G2_H = cv2.getTrackbarPos("MinG", "Color Track Bar")
    S2_L = cv2.getTrackbarPos("MaxS", "Color Track Bar")
    S2_H = cv2.getTrackbarPos("MinS", "Color Track Bar")

    # threshhloding G & S
    ret, fulls = cv2.threshold(Gs, G2_l, G2_H, cv2.THRESH_BINARY_INV)
    ret, fulll = cv2.threshold(Gl, G2_l, G2_H, cv2.THRESH_BINARY_INV)

    ret, whitel = cv2.threshold(Gs, S2_L, S2_H, cv2.THRESH_BINARY_INV)
    ret, whites = cv2.threshold(Gl, S2_L, S2_H, cv2.THRESH_BINARY_INV)

    whites= dilation(whites,2)
    whites= dilation(whites,2)

    pinks = ~(cv2.subtract(whites, fulls))
    pinkl = ~(cv2.subtract(whitel, fulll))

    pinks = errosion(pinks)
    pinkl = errosion(pinkl)

    cv2.imshow("fulls", fulls)
    cv2.imshow("fulll", fulll)

    cv2.imshow("whitel", whitel)
    cv2.imshow("whites", whites)

    # cv2.imshow("pinkl", pinkl)
    # cv2.imshow("pinks", pinks)

    # break when esc is pressed and save the images
    if cv2.waitKey(1) == 27:
        # Thresh_S1=thresh4s.copy()
        # Thresh_S2=thresh4l.copy()
        Thresh_S1 = pinks.copy()
        Thresh_S2 = pinkl.copy()
        Thresh_g1 = whites.copy()
        Thresh_g2 = whitel.copy()
        break

#####################################################################################
# extra blur
Thresh_S1 = cv2.GaussianBlur(Thresh_S1, (5, 5), 1)
Thresh_S2 = cv2.GaussianBlur(Thresh_S2, (5, 5), 1)
Thresh_g1 = cv2.GaussianBlur(Thresh_g1, (5, 5), 1)
Thresh_g2 = cv2.GaussianBlur(Thresh_g1, (5, 5), 1)
# Layers To Use
#Layers To Use
#growth_1     = cv2.subtract(Thresh_S2 , Thresh_S1)
#growth_2     = cv2.subtract(growth_1 , Thresh_g1)
growth_2     = cv2.subtract (fulls  ,fulll)
growth_3     =cv2.subtract(growth_2,whites)
growth_4     =cv2.subtract(growth_3,whitel)
if seeToEdit :
    cv2.imshow("wishes",growth_4)
damage_2      =cv2.subtract(fulll , fulls)
damage_2      =cv2.subtract(damage_2,pinks)
bleach_0     = cv2.subtract( whitel,whites)

if seeToEdit :
    cv2.imshow("secondwisgh",bleach_0)

cured_0     = cv2.subtract(whites,whitel)
cured_0     = cv2.subtract(cured_0,damage_2)

if seeToEdit :
    cv2.imshow("growth_2",growth_2)
    cv2.imshow("damage_2",damage_2)
    #cv2.imshow("bleach_2",bleach_2)
    cv2.imshow("cured_0" ,cured_0)
#Noise Removal
growth_2 = cv2.medianBlur(growth_2,13)
damage_2 = cv2.medianBlur(damage_2,13)
bleach_2 = cv2.medianBlur(bleach_0,13)#
cured_2  = cv2.medianBlur(cured_0,13)#

Pinkgrow    , hierarchy  = cv2.findContours(growth_2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
Pinkdamage      , hierarchy  = cv2.findContours(damage_2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
bleached , hierarchy  = cv2.findContours(bleach_2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
Pinkheal      , hierarchy  = cv2.findContours(cured_2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

############################Conditions###########################
bleach=[]
damage=[]
heal=[]
growth=[]

minArea =1000
maxArea =50000
for cnt in bleached:
    if maxArea > cv2.contourArea(cnt) > minArea:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        M_W = cv2.moments(approx)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.drawContours(main, cnt, -1, (0,0,255), 3)
        cv2.rectangle(live_c,(x-10,y-10),(x+w+10,y+h+10),(0,0,255),2)
        b=[x,y,h,w]
        bleach.append(b)
        if seeToEdit:
            print("here is bleach ",bleach)
for cnt in Pinkgrow:
    if maxArea > cv2.contourArea(cnt) >minArea:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        M_P = cv2.moments(approx)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(live_c,(x-10,y-10),(x+w+10,y+h+10),(0,255,0),2)
        cv2.drawContours(main, cnt, -1, (0,255,0), 3)
        g=[x,y,h,w]
        growth.append(g)
        if seeToEdit:
            print("here is growth ",growth)

for cnt in Pinkheal:
    if maxArea > cv2.contourArea(cnt) > minArea:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        M_P = cv2.moments(approx)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(ive_c,(x-10,y-10),(x+w+10,y+h+10),(255,0,0),2)
        cv2.drawContours(main, cnt, -1, (255,0,0), 3)
        h=[x,y,h,w]
        heal.append(h)
        if seeToEdit:
            print("done3")
            print("here is heal ",heal)

for cnt in Pinkdamage:
    if maxArea > cv2.contourArea(cnt) >minArea:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        M_P = cv2.moments(approx)
        #calculate x,y coordinate of center
        if M_P["m00"] ==0:
            M_P["m00"]=1
        cX = int(M_P["m10"] / M_P["m00"])
        cY = int(M_P["m01"] / M_P["m00"])
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.drawContours(live_c, cnt, -1, (0,255,255), 3)
        cv2.rectangle(main,(x-10,y-10),(x+w+10,y+h+10),(0,255,255),2)
        d=[x,y,h,w]
        damage.append(d)
        if seeToEdit:
            print("here is damage ",damage)


cv2.imshow("IMG2",live_c)
cv2.imshow("img",stnd_c)
cv2.imshow("main",main)

cv2.imwrite('Result1.jpg',main)
cv2.imwrite('Result2.jpg',live_c)
cv2.waitKey(0)
