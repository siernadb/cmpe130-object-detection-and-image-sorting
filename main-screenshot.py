import cv2
import glob
import time
import os

# Set counters and other global variables
is_inFrame = False
prev = time.time()
appleCount = bananaCount = orangeCount = 0
path = "G:\My Drive\CMPE 130 - adv alg des\project versions\h02-cmpe130-object-detection-and-image-sorting-main\cmpe130-object-detection-and-image-sorting"


# ------------------------- OBJECT DETECTION STARTS HERE -------------------------
# Setup for camera detection8
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap.set(3,640)
cap.set(4,480)

# Read list of item names from coco file
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Specify config path and weights path
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Build the model for object detection
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)                              # Set size of the input box
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Send image to the model
while True:
    success, img = cap.read()
    classID_list, confidence, bbox = net.detect(img, confThreshold = 0.65)
    print(classID_list, bbox)

    if len(classID_list) != 0:
        for classID, confidence_element, box in zip(classID_list.flatten(), confidence.flatten(), bbox):
            cv2.rectangle(img, box, color=(255, 216, 1), thickness = 3)
            cv2.putText(img, classNames[classID-1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 216, 1), 2)

        # Screenshot if apple/orange/banana is detected
        #   Capture every 3 seconds -- otherwise: opencv will keep exporting a screenshot for each frame
        # BANANA
        if classID == 52:
            is_inFrame = True
            curr = time.time()
            if curr - prev >= 3 and is_inFrame:
                img_name = "banana{}.png".format(bananaCount)
                bananadir = 'Bananas'
                if os.path.isdir(os.path.join(path,bananadir)):
                    cv2.imwrite(os.path.join(path, bananadir, img_name), img)
                    print("{} written!".format(img_name))
                    prev = curr
                    bananaCount += 1
                else:
                    os.mkdir(os.path.join(path,bananadir))
                    cv2.imwrite(os.path.join(path, bananadir, img_name), img)
                    print("{} written!".format(img_name))
                    prev = curr
                    bananaCount += 1
        # APPLE
        elif classID == 53:
            is_inFrame = True
            curr = time.time()
            if curr - prev >= 3 and is_inFrame:
                img_name = "apple{}.png".format(appleCount)
               
                appledir = 'Apples'
                if os.path.isdir(os.path.join(path,appledir)):
                    cv2.imwrite(os.path.join(path, appledir, img_name), img)
                    print("{} written!".format(img_name))
                    prev = curr
                    appleCount += 1
                else:
                    os.mkdir(os.path.join(path,appledir))
                    cv2.imwrite(os.path.join(path, appledir, img_name), img)
                    print("{} written!".format(img_name))
                    prev = curr
                    appleCount += 1
        # ORANGE
        elif classID == 55:
            is_inFrame = True
            curr = time.time()
            if curr - prev >= 3 and is_inFrame:
                img_name = "orange{}.png".format(orangeCount)
                orangedir = 'Oranges'
                if os.path.isdir(os.path.join(path,orangedir)):
                    cv2.imwrite(os.path.join(path, orangedir, img_name), img)
                    print("{} written!".format(img_name))
                    prev = curr
                    print("{} written!".format(img_name))
                    prev = curr
                    orangeCount += 1
                else:
                    os.mkdir(os.path.join(path,orangedir))
                    cv2.imwrite(os.path.join(path, orangedir, img_name), img)
                    print("{} written!".format(img_name))
                    prev = curr

        # NOT A FRUIT / OFF-FRAME
        else:
            is_inFrame = False
            prev -= 5

    if(cv2.waitKey(1) == 27):
        break

    # Show display
    cv2.imshow("Output", img)
    cv2.waitKey(1)

# ------------------------- OBJECT DETECTION ENDS HERE -------------------------
# ------------------------- SORTING STARTS HERE --------------------------------





# ------------------------------------------------------------------------------

print("Apple count: " + str(appleCount))
print("Banana count: " + str(bananaCount))
print("Orange count: " + str(orangeCount))
