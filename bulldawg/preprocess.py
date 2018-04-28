import cv2
import numpy as np
import pandas as pd

def convert_bgr2gray(img):
    '''
    Convert to gray image
    
    Args: 
        img: image to be converted in gray
    Returns:
        gray image
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def resize_img(img, shape):
    '''
    Resize the image to shape provided

    Args:
        img: image to be reshaped

        shape: intended shape e.g. if intended shape is (128, 128) 
               shape value is 128
    Returns:
        resized image
    '''
    img_resize = cv2.resize(img,(shape, shape))
    return img_resize

def modify_imgs(img_data):
    '''
    This method divides the pixels number by 255 for better training and adds axis to it

    Args:
        img_data: numpy array containing image data
    
    Returns:
        numpy array of modified images
    '''
    img_data = img_data.astype('float32')
    img_data /= 255
    img_data= img_data[...,np.newaxis]
    return img_data



