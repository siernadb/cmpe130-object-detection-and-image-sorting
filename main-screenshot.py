import cv2
import glob
import time
import os
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog



# Set counters and other global variables
global is_inFrame
global prev
global appleCount, bananaCount, orangeCount
global fruits_scanned
global path
global path_string
global endTask_isEnabled

root = tk.Tk()
path = tk.StringVar()
path_string = ""
endTask_isEnabled = False


#path = "C:/Users/siern/github/cmpe130-object-detection-and-image-sorting"
# ------------------------- GUI STARTS HERE --------------------------------
def browse_directory():
    filename = filedialog.askdirectory()
    path.set(filename)
    path_string = str(filename)
    print("File name set: ", path_string)

def endTask():
    endTask_isEnabled = True

def objectDetection_enable():
    # ------------------------- OBJECT DETECTION STARTS HERE -------------------------
    # Setup for camera detection8
    is_inFrame = False
    prev = time.time()
    appleCount = bananaCount = orangeCount = 0
    fruits_scanned = []
    print("Path string: ", path_string)


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
                    fruits_scanned += img_name

                    bananadir = 'Bananas'
                    if os.path.isdir(os.path.join(path_string,bananadir)):
                        cv2.imwrite(os.path.join(path_string, bananadir, img_name), img)
                        print("{} written!".format(img_name))
                        prev = curr
                        bananaCount += 1
                    else:
                        os.mkdir(os.path.join(path_string,bananadir))
                        cv2.imwrite(os.path.join(path_string, bananadir, img_name), img)
                        print("{} written!".format(img_name))
                        prev = curr
                        bananaCount += 1

            # APPLE
            elif classID == 53:
                is_inFrame = True
                curr = time.time()
                if curr - prev >= 3 and is_inFrame:
                    img_name = "apple{}.png".format(appleCount)
                    fruits_scanned += img_name

                    appledir = 'Apples'
                    if os.path.isdir(os.path.join(path_string,appledir)):
                        cv2.imwrite(os.path.join(path_string, appledir, img_name), img)
                        print("{} written!".format(img_name))
                        prev = curr
                        appleCount += 1
                    else:
                        os.mkdir(os.path.join(path_string,appledir))
                        cv2.imwrite(os.path.join(path_string, appledir, img_name), img)
                        print("{} written!".format(img_name))
                        prev = curr
                        appleCount += 1

            # ORANGE
            elif classID == 55:
                is_inFrame = True
                curr = time.time()
                if curr - prev >= 3 and is_inFrame:
                    img_name = "orange{}.png".format(orangeCount)
                    fruits_scanned += img_name

                    orangedir = 'Oranges'
                    if os.path.isdir(os.path.join(path_string,orangedir)):
                        cv2.imwrite(os.path.join(path_string, orangedir, img_name), img)
                        print("{} written!".format(img_name))
                        prev = curr
                        print("{} written!".format(img_name))
                        prev = curr
                        orangeCount += 1
                    else:
                        os.mkdir(os.path.join(path_string,orangedir))
                        cv2.imwrite(os.path.join(path_string, orangedir, img_name), img)
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
# ------------------------- MAIN STARTS HERE -------------------------
def main():
    # Interface variables
    directoryFound = False





    # Set up window
    canvas = tk.Canvas(root, width=600, height=300)
    canvas.grid(columnspan=3, rowspan=5)   # Splits canvas into three invisible sections

    # Titles and labels
    mainTitle = tk.Label(root, text="CMPE 133\nOBJECT DETECTION AND IMAGE SORTING", font=("Consolas", 18, 'bold'))
    mainTitle.grid(columnspan=3, column=0, row=0)

    # Insert BROWSE button
    browse_label = tk.StringVar()
    browse_button = tk.Button(root, textvariable=browse_label, command=lambda:browse_directory(), font="Raleway", bg="#81a55f", fg="white", height=2, width=15)
    browse_label.set("Set Directory")
    browse_button.grid(column=1, row=2)


    # Insert END TASK button
    end_label = tk.StringVar()
    end_button = tk.Button(root, textvariable=end_label, command=lambda: endTask(), font="Raleway", bg="#81a55f", fg="white", height=2, width=15)
    end_label.set("End Session")
    end_button.grid(column=1, row=4)

    # Insert RUN button
    run_label = tk.StringVar()

    run_button = tk.Button(root, textvariable=run_label, command=lambda:objectDetection_enable(), font="Raleway", bg="#81a55f", fg="white", height=2, width=15)
    run_label.set("Run")
    run_button.grid(column=1, row=3)



    root.mainloop()

    print("Apple count: " + str(appleCount))
    print("Banana count: " + str(bananaCount))
    print("Orange count: " + str(orangeCount))


main()



# ------------------------------------------------------------------------------
