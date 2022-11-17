#import necessary libraries to use for training model
import cv2 as CV #importing cv2 library as CV for image processing applications
import os  #importing os to save yml files
import numpy as np #import numpy so we can work with arrays
from PIL import Image #import PIL images module to work with image manipulation

Dataset_Path = 'dataset' #The path where dataset folder is locally

Face_Recognizer = CV.face.LBPHFaceRecognizer_create() #We will use recognizer, the LBPH (LOCAL BINARY PATTERNS HISTOGRAMS) Face Recognizer, included on OpenCV package.
Face_Finder = CV.CascadeClassifier("haarcascade_frontalface_default.xml") #we have to load classifier, here we are loading "haarcascade_frontalface_default.xml"


def Fetch_FaceSamples_And_Sn(dataset_path): #function to get the face samples and serial number data

    GrayscalesImagesPaths = [os.path.join(dataset_path,f) for f in os.listdir(dataset_path)]  #getting path for processing images  
    FaceSamples=[] # creating empty array for Face samples and serial number
    Sn = []

    for GrayscalesImagesPath in GrayscalesImagesPaths: #loop for all images in the dataset path

        PILimage = Image.open(GrayscalesImagesPath).convert('L') # converting image to grayscale
        Image_Numpy = np.array(PILimage,'uint8') #converting PIL images into numpy array

        id = int(os.path.split(GrayscalesImagesPath)[-1].split(".")[1]) #getting the serial number
        Face = Face_Finder.detectMultiScale(Image_Numpy) #using numpy array to find face in the image

        for (x,y,w,h) in Face: #loops for all the found faces
            FaceSamples.append(Image_Numpy[y:y+h,x:x+w]) #appending face samples
            Sn.append(id) #appending serial number

    return FaceSamples, Sn #returning the face samples acquired with serial number

print ("\n The face samples are being trained. Please Patiently Wait ...") #printing the message so user can know the work is being done in the background
PersonFace,PersonId = Fetch_FaceSamples_And_Sn(Dataset_Path) #calling function by passing Dataset path and getting Person's face and Person's Id to train model
Face_Recognizer.train(PersonFace, np.array(PersonId)) #training each and every person's model

Face_Recognizer.write('trainer/trainer.yml') #Lets save the trained model in trainer folder

print("\n {0} faces trained. Exiting Program".format(len(np.unique(PersonId)))) #Printing number of faces trained