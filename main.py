import cv2
import glob
import time
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt         #pip install matplotlib
import pathlib
from pathlib import Path


# Set counters and other global variables
global is_inFrame
global prev

global path
global tpath
global tdir
global id_list


tdir = 'temp'
tpath = os.path.dirname(os.path.abspath(__file__))

root = tk.Tk()
path = tk.StringVar()
path_string = ""

# ------------------------- GUI STARTS HERE --------------------------------
def displaySummary():
    run_label.set("Run")
    x_fruits = ["Apples", "Bananas", "Oranges"]
    x_values = [appleCount, bananaCount, orangeCount]
    plt.bar(x_fruits,x_values)
    plt.xlabel("Fruits")
    plt.ylabel("No. of Fruits Detected")
    plt.title("Fruits Detected and Sorted")
    plt.show()


def browse_directory():
    global path_string
    filename = filedialog.askdirectory()
    path.set(filename)
    path_string = str(filename)
    print("File name set: ", path_string)



def runDetection():
    run_label.set("Running...")
    objectDetection_enable()

def objectDetection_enable():
    # ------------------------- OBJECT DETECTION STARTS HERE -------------------------
    # Setup for camera detection8
    is_inFrame = False
    prev = time.time()
    global appleCount,bananaCount,orangeCount
    appleCount = bananaCount = orangeCount = 0
    global appleC,bananaC,orangeC
    appleC = 10000
    bananaC = 20000
    orangeC = 30000
    global fruits_scanned
    fruits_scanned = []
    print("Path string: ", path_string)
    if not os.path.isdir(os.path.join(tpath,tdir)):
        os.mkdir(os.path.join(tpath,tdir))

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
        classID_list, confidence, bbox = net.detect(img, confThreshold = 0.70)
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
                    img_name = "{}.png".format(bananaC)

                    cv2.imwrite(os.path.join(tpath,tdir,img_name), img)
                    print("{} written!".format(img_name))
                    fruits_scanned.append(img_name[0:5])
                    prev = curr
                    bananaCount += 1
                    bananaC+=1



            # APPLE
            elif classID == 53:
                is_inFrame = True
                curr = time.time()
                if curr - prev >= 3 and is_inFrame:
                    img_name = "{}.png".format(appleC)

                    cv2.imwrite(os.path.join(tpath, tdir, img_name), img)
                    print("{} written!".format(img_name))
                    fruits_scanned.append(img_name[0:5])
                    prev = curr
                    appleCount += 1
                    appleC += 1



            # ORANGE
            elif classID == 55:
                is_inFrame = True
                curr = time.time()
                if curr - prev >= 3 and is_inFrame:
                    img_name = "{}.png".format(orangeC)

                    cv2.imwrite(os.path.join(tpath, tdir, img_name), img)
                    fruits_scanned.append(img_name[0:5])
                    print("{} written!".format(img_name))
                    prev = curr
                    orangeCount += 1
                    orangeC += 1


            # NOT A FRUIT / OFF-FRAME
            else:
                is_inFrame = False
                prev -= 5

        if(cv2.waitKey(1) == 27):
            cap.release()
            cv2.destroyAllWindows()
            break

        # Show display
        cv2.imshow("Output", img)
        cv2.waitKey(1)




#..............................SORTING BEGINS HERE ........................
def quick_sort(sequence):
    for x in range(len(sequence)):
        print (sequence[x])
    length = len(sequence)
    if length>1:
        pivot = sequence.pop()
        items_greater = []
        items_lower = []

        for item in sequence:
            if item >  pivot:
                items_greater.append(item)
            else:
                items_lower.append(item)
        return quick_sort(items_lower) + [pivot] + quick_sort(items_greater)
    else:
        return sequence


def sort_list():
    sort_label.set("Sorting")
    global sorted_list
    sorted_list = quick_sort(fruits_scanned)
    for x in range(len(sorted_list)):
        print (sorted_list[x])
    tfile = Path(os.path.join(tpath,tdir))
    appledir = 'Apples'
    bananadir = 'Bananas'
    orangedir = 'Oranges'
    count = 0

    for file in tfile.iterdir():

        for item in range(appleCount):

            if file.name[0:5] == sorted_list[item]:
                fname = "Apple{}.png".format(count)
                new_path =os.path.join(path_string,appledir)
                new_file =os.path.join(new_path,fname)
                if not os.path.isdir(new_path):
                    os.mkdir(new_path)
                file.replace( Path(new_file))
                count+=1

    count = 0
    index = appleCount
    until = appleCount + bananaCount

    for file in tfile.iterdir():
        for item in range(index,until):
            if file.name[0:5] == sorted_list[item]:
                fname = "Banana{}.png".format(count)
                new_path =os.path.join(path_string,bananadir)
                new_file =os.path.join(new_path,fname)
                if not os.path.isdir(new_path):
                    os.mkdir(new_path)
                file.replace( Path(new_file))
                count+=1

    count = 0
    index = appleCount + bananaCount
    until = appleCount +bananaCount + orangeCount

    for file in tfile.iterdir():
        for item in range(index, until):
            if file.name[0:5] == sorted_list[item]:
                fname = "Orange{}.png".format(count)
                new_path =os.path.join(path_string,orangedir)
                new_file =os.path.join(new_path,fname)
                if not os.path.isdir(new_path):
                    os.mkdir(new_path)
                file.replace(Path(new_file))
                count+=1
    print ("sorting and re-organization done")


# ------------------------- OBJECT DETECTION ENDS HERE -------------------------
# ------------------------- MAIN STARTS HERE -------------------------
def main():
    # Interface variables
    # Set up window
    canvas = tk.Canvas(root, width=800, height=400)
    canvas.grid(columnspan=3, rowspan=9)   # Splits canvas into three invisible sections

    # Titles and labels
    mainTitle = tk.Label(root, text="CMPE 130\nOBJECT DETECTION AND IMAGE SORTING", font=("Consolas", 18, 'bold'))
    mainTitle.grid(columnspan=3, column=0, row=0)

    # Insert BROWSE button
    browse_label = tk.StringVar()
    browse_button = tk.Button(root, textvariable=browse_label, command=lambda:browse_directory(), font="Raleway", bg="#81a55f", fg="white", height=2, width=15)
    browse_label.set("Set Directory")
    browse_button.grid(column=1, row=2)


    # Insert Summary button
    global summary_label
    summary_label = tk.StringVar()
    summary_button = tk.Button(root, textvariable=summary_label, command=lambda: displaySummary(), font="Raleway", bg="#81a55f", fg="white", height=2, width=15)
    summary_label.set("Show Summary")
    summary_button.grid(column=1, row=5)

     # Insert END TASK button
    global sort_label
    sort_label = tk.StringVar()
    sort_button = tk.Button(root, textvariable=sort_label, command=lambda: sort_list(), font="Raleway", bg="#81a55f", fg="white", height=2, width=15)
    sort_label.set("Sort list")
    sort_button.grid(column=1, row=4)

    # Insert RUN button
    global run_label
    run_label = tk.StringVar()
    run_button = tk.Button(root, textvariable=run_label, command=lambda:runDetection(), font="Raleway", bg="#81a55f", fg="white", height=2, width=15)
    run_button.grid(column=1, row=3)
    run_label.set("Run")
    root.mainloop()

main()



# ------------------------------------------------------------------------------
