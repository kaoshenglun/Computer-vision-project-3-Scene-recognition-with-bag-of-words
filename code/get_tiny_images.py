from PIL import Image

import pdb
import numpy as np

def get_tiny_images(image_paths):

    '''
    Input : 
        image_paths: a list(N) of string where where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    N = len(image_paths) #1500
    tiny_images = np.zeros((N,256))
    
    for i in range(N):
        img = Image.open(image_paths[i])
        img = img.resize((16,16))
        image_resize = np.asarray(img, dtype='float32').reshape(256,).flatten()
        tiny_images[i,:] = (image_resize-np.mean(image_resize))/np.std(image_resize)
    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images
