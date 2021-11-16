# ===============================================================================
#    This sample illustrates how to grab and process images using the CInstantCamera class.
#    The images are grabbed and processed asynchronously, i.e.,
#    while the application is processing a buffer, the acquisition of the next buffer is done
#    in parallel.
#
#    The CInstantCamera class uses a pool of buffers to retrieve image data
#    from the camera device. Once a buffer is filled and ready,
#    the buffer can be retrieved from the camera object for processing. The buffer
#    and additional image data are collected in a grab result. The grab result is
#    held by a smart pointer after retrieval. The buffer is automatically reused
#    when explicitly released or when the smart pointer object is destroyed.
# ===============================================================================
from pypylon import pylon
from pypylon import genicam

import cv2
import sys

from imageio import get_writer

import numpy as np
import matplotlib

class Cap():
    def __init__(self):
        self.cap = None

        # converting to opencv bgr format
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        #%% Set video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('C:/Users/USER\Desktop/Baseball VR/Camera/全彩攝影機/Recording Output/Amber-test1-r.avi', fourcc, 60.0, (800,  600))
        self.out = cv2.VideoWriter('D:/Luke-test_new.avi', fourcc, 60.0, (800,  600))

    # def get_cap(self):
        # Number of images to be grabbed.
        countOfImagesToGrab = 100

        # The exit code of the sample application.
        exitCode = 0

        try:
            # Create an instant camera object with the camera device found first.
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            camera.Open()

            # Print the model name of the camera.
            print("Using device ", camera.GetDeviceInfo().GetModelName())

            # demonstrate some feature access
            new_width = camera.Width.GetValue() - camera.Width.GetInc()
            if new_width >= camera.Width.GetMin():
                camera.Width.SetValue(new_width)

            pylon.FeaturePersistence.Load("acA800-510uc_22048202.pfs", camera.GetNodeMap(), True)

            self.cap = camera
            #%%
            # Grabing Continusely (video) with minimal delay
            self.cap.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
   
        except genicam.GenericException as e:
            # Error handling.

            print("An exception occurred.")
            print(e.GetDescription())
            exitCode = 1
        
    def read(self):
        # Wait for an image and then retr
        # ieve it. A timeout of 5000 ms is used.
        grabResult = self.cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            # Access the image data.
            # print("SizeX: ", grabResult.Width)
            # print("SizeY: ", grabResult.Height)
            image = self.converter.Convert(grabResult)
            img = image.GetArray()
            self.out.write(img)
            print(img)

            cv2.line(img, (0,220), (800,220), (0,255,0), 2)
            cv2.line(img, (0,350), (800,350), (0,255,0), 2)
            cv2.line(img, (320,0), (320,600), (0,255,0), 2)
            cv2.line(img, (433,0), (433,600), (0,255,0), 2)

            # cv2.imshow('title', np.array(img, dtype = np.uint8))
            # cv2.waitKey(1)
            # print(img)

        else:
            pass
            # print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        
        return None, img
        grabResult.Release()
    def close(self):
        self.cap.Close()


    # sys.exit(exitCode)
    # out.release()
    # cv2.destroyAllWindows()


if __name__ == "main":
    import time
    cap = Cap()
    while True:
        cap.read()
    cap.close()