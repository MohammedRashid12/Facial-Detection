#import necessary libraries to use for collecting the images that we can use for training model
import cv2 as CV #importing cv2 library as CV for image processing applications
import os  #importing os to save images

Video_Feed = CV.VideoCapture(0)  #raspberry pi camera's video can be captured from VideoCapture(0) so we can process it later
Video_Feed.set(3, 640) # this command is used to set video's resolution width to 640 pixels
Video_Feed.set(4, 480) # this command is used to set video's resolution height to 480 pixels

Face_Finder = CV.CascadeClassifier('haarcascade_frontalface_default.xml') #we have to load classifier, here we are loading "haarcascade_frontalface_default.xml"

Face_SN = input(" \n enter user id and press enter:  ") # We have to assign every face we want to train to a specific Serial Number
Counter_Max = int(input(" \n enter sample number and press enter:  ")) # asking from user, how many samples to capture for training face
Counter = 0 # we are starting our counter from 0

while(True):  #starting our while loop, it runs until the loops exits from inside
    Return, Image = Video_Feed.read()  #Return is a boolean variable that returns true if the image frame is available. Image is an image array vector captured based on the default frames per second
    ImageToGray = CV.cvtColor(Image, CV.COLOR_BGR2GRAY) #converting normal captured images to grayscale
    Faces = Face_Finder.detectMultiScale(ImageToGray, 1.3, 5) #using grayscale image to find face in the image

    for (x,y,w,h) in Faces: # this loop will only be entered if there is face available in the image
        CV.rectangle(Image, (x,y), (x+w,y+h), (255,0,0), 2)  # If faces are found in the image frame , it returns the positions of detected faces as a rectangle giving coordinates (x,y,w,h)
        Counter += 1 # increasing the counter by 1
        CV.imwrite("dataset/User." + str(Face_SN) + '.' + str(Counter) + ".jpg", ImageToGray[y:y+h,x:x+w]) # Saving the captured image in the dataset folder for training
        print("\n Saved "+ str(Counter)+" image for "+str(Face_SN) +"user") # Displaying the saved image info
        CV.imshow('image', Image) #Displaying the saved image frame in the screen

    KeyValue = CV.waitKey(100) & 0xff # waiting for 'ESC' button press t0 exit video
    if KeyValue == 27: #checking if 'ESC' button was pressed
        break #if pressed then the loops exits here
    elif Counter >= Counter_Max: # checking if counter has reached the value of number of samples as input from user
         break # if counter has reached number of samples the loop exits

print("\n Exiting Program and closing camera and cv2 windows")
Video_Feed.release() #Closing camera feed
CV.destroyAllWindows() #closing all opencv windows
