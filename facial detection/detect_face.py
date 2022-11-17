#import necessary libraries to use for face detection and sending email
import cv2 as CV #importing cv2 library as CV for image processing applications
import numpy as np #import numpy so we can work with arrays
from time import sleep #import sleep from time module so we could add delay in the program
import RPi.GPIO as GPIO #import RPi.GPIO module as GPIO to turn gpio pin on or off for relay control
import smtplib #import smtplib module to send mail
import ssl #import ssl module for using transport layer security
from email.message import EmailMessage #import email.message module to create the container email message
from datetime import datetime #import datetime to attach on the email

email_sender = 'ameenproject42@gmail.com' #adding credentials from which account the email has to be send
email_password = 'mpiiwmbfcwlgpubn' #generated app passwords for using it via python
email_receiver = 'mazgoufadrian@gmail.com' #email receiver address

Relay_Trigger = 26  # declaring the relay pin to turn it on or off
GPIO.setwarnings(False) #turning off any warnings from GPIO module
GPIO.setmode(GPIO.BCM) #configuring GPIO module to use BCM numbering of the 40 header pins
GPIO.setup(Relay_Trigger, GPIO.OUT) #setting relay pin as output pin

Video_Feed = CV.VideoCapture(0)  #raspberry pi camera's video can be captured from VideoCapture(0) so we can process it later
Video_Feed.set(3, 640) # this command is used to set video's resolution width to 640 pixels
Video_Feed.set(4, 480) # this command is used to set video's resolution height to 480 pixels

Face_Recognizer = CV.face.LBPHFaceRecognizer_create() #We will use recognizer, the LBPH (LOCAL BINARY PATTERNS HISTOGRAMS) Face Recognizer, included on OpenCV package.
Face_Finder = CV.CascadeClassifier("haarcascade_frontalface_default.xml") #we have to load classifier, here we are loading "haarcascade_frontalface_default.xml"
Face_Recognizer.read('trainer/trainer.yml') #making recogonizer module use our trained model from trainer folder stored locally after training process

font = CV.FONT_HERSHEY_SIMPLEX #using Hershey Simplex font from open cv package
Users = ["None","Blase"] #list of users starting from 0

MinWidth = 0.1*Video_Feed.get(3) #minimum width size to be recognized as a face
MinHeight = 0.1*Video_Feed.get(4) #minimum height size to be recognized as a face

def EmailSender(subject,body): #Email sender function to send email
    em = EmailMessage() # creating an object to call EmailMessage() function
    em['From'] = email_sender #adding email_sender address
    em['To'] = email_receiver #adding email_receiver address
    em['Subject'] = subject #adding email subject
    em.set_content(body) #adding body of the email

    context = ssl.create_default_context() # Add SSL (layer of security)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password) # Log in to the email
        smtp.sendmail(email_sender, email_receiver, em.as_string()) #sending email

while(True):  #starting our while loop, it runs until the loops exits from inside
    Return, Image = Video_Feed.read()  #Return is a boolean variable that returns true if the image frame is available. Image is an image array vector captured based on the default frames per second
    ImageToGray = CV.cvtColor(Image, CV.COLOR_BGR2GRAY) #converting normal captured images to grayscale
    Faces = Face_Finder.detectMultiScale(
        ImageToGray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(MinWidth), int(MinHeight)),
       )                                            #using grayscale image to find face in the image with minimum size to be detected as a face
   
    for (x,y,w,h) in Faces: # this loop will only be entered if there is face available in the image
        CV.rectangle(Image, (x,y), (x+w,y+h), (255,0,0), 2)  # If faces are found in the image frame , it returns the positions of detected faces as a rectangle giving coordinates (x,y,w,h)
        Sn, MatchPercentage = Face_Recognizer.predict(ImageToGray[y:y+h,x:x+w]) # Face Recognizer modules takes grayscale image and predicts Serial number and Match Percentage level
        if (MatchPercentage <= 50): # MatchPercentage is reverse so the lower the match percentage the better, we are checking if its lower then 30 percentage for if condition
            Sn = Users[Sn] #finding the user based on trained model
            MatchPercentage = "  {0}%".format(round(100 - MatchPercentage)) #Displaying the actual Match Percentage
            GPIO.output(Relay_Trigger, GPIO.LOW) #Setting the relay pin to low to open it
            #sleep(5)
            #GPIO.output(Relay_Trigger, GPIO.HIGH) #setting the relay pin to high to lock it again
            print("Opening Lock") # Displaying the message that Lock has opened
            subject = 'Door Lock has Opened' #subject of the email
            TimeOfUnlocked = datetime.now().strftime("%d/%m/%Y %H:%M:%S") #getting date and time from datetime module and converting it in proper dd/mm/YY H:M:S format
            body = Sn+" has opened the lock at "+str(TimeOfUnlocked) #adding username who has opened the lock in body of an email
            EmailSender(subject,body); #calling email sender function to send email
            print("email has been send") # Displaying the message that email has been send


        else: # if above case is not satisfied then this block of code is executed
            Sn = "Unknown" #displaying unknown message because the face is not matched in the trained model
            MatchPercentage = "100" #Displaying the full Match Percentagen because the person is unknown
            subject = 'Unknown Person trying to get access' #subject of the email
            TimeOfUnlocked = datetime.now().strftime("%d/%m/%Y %H:%M:%S") #getting date and time from datetime module and converting it in proper dd/mm/YY H:M:S format
            body = "Unknown Person is at the Door at "+str(TimeOfUnlocked) #adding unknown person who is trying to open the lock in body of an email
            EmailSender(subject,body); #calling email sender function to send email
            print("email has been send") # Displaying the message that email has been send
            print("Door lock has locked") #display message on the console
            GPIO.output(Relay_Trigger, GPIO.HIGH) #setting the relay pin to high to lock it again
        CV.putText(Image, str(Sn), (x+5,y-5), font, 1, (255,255,255), 2) #overlaying the Serial number information on the Image
        CV.putText(Image, str(MatchPercentage), (x+5,y+h-5), font, 1, (255,255,0), 1) #overlaying the name information on the Image
    CV.imshow('Video',Image) #Displaying the Video frame in the screen
    KeyValue = CV.waitKey(100) & 0xff # waiting for 'ESC' button press t0 exit video
    if KeyValue == 27: #checking if 'ESC' button was pressed
        break #if pressed then the loops exits here

print("\n Exiting Program and closing camera and cv2 windows")
Video_Feed.release() #Closing camera feed
CV.destroyAllWindows() #closing all opencv windows
GPIO.cleanup() #cleanup GPIO pins during exit()