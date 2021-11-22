import cv2
import glob

i=0
img_counter=0
path = "C:/Users/siern/github/cmpe130-object-detection-and-image-sorting"

#For camera detection
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap.set(3,640)
cap.set(4,480)

#Set counters
appleCount = bananaCount = orangeCount = 0

# Read list of item names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Specify config path and weights path
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Build the model
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
            if classID == 53:
                if img_counter%(3*fps) == 0:
                    img_name = "opencv_frame_{}.png".format(img_counter)
                    cv2.imwrite(img_name, img)
                    print("{} written!".format(img_name))
                    appleCount += 1
                    img_counter += 1
            elif classID == 52:
                bananaCount += 1


    # Show the image that was read earlier
    cv2.imshow("Output", img)
    cv2.waitKey(1)

    # Ends session when 'b' is pressed
    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break

    # Screenshot if apple/orange/banana is detected

print("Apple count: " + str(appleCount))
print("Banana count: " + str(bananaCount))
