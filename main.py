import cv2
import glob

# Read an image and store it in a variable
#img = cv2.imread('rotten-apple.jpg')

path = "C:/Users/siern/github/cmpe130-object-detection-and-image-sorting"
images = [cv2.imread(file) for file in glob.glob(path + "/*.jpg")]
i=0

appleCount = bananaCount = orangeCount = 0

for img in images:
    print("image " + str(i+1))
    i +=1
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
    classID_list, confidence, bbox = net.detect(img, confThreshold = 0.65)
    print(classID_list, bbox)

    for classID, confidence_element, box in zip(classID_list.flatten(), confidence.flatten(), bbox):
        cv2.rectangle(img, box, color=(255, 216, 1), thickness = 3)
        cv2.putText(img, classNames[classID-1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 216, 1), 2)

        if classID == 53:
            appleCount += 1
        elif classID == 52:
            bananaCount += 1

    # Show the image that was read earlier
    #cv2.imshow("Output", img)
    cv2.imshow("Output", img)
    cv2.waitKey(500)

print("Apple count: " + str(appleCount))
print("Banana count: " + str(bananaCount))
