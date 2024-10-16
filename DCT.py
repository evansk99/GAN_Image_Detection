from PIL import Image
from scipy import ndimage
import numpy as np
import scipy 
from scipy.fftpack import dct, idct
import PIL
from PIL import Image

###
#This function computes the blockwise DCT of an image in 8x8 blocks, keeps a number of coeffecients (1-64)
#in specific positions and rearranges the result to create the proposed data format
###

def DCT(image):
    dct = np.zeros_like(image,dtype = 'float64')
    block_rows=8 #block width
    block_columns=8 #block height
    #specify positions of the coefficients that will be kept
    pos_x=np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4,4,5,5,6]) #Keep 28 coefficients 
    pos_y=np.array([0,1,2,3,4,5,6,0,1,2,3,4,5,0,1,2,3,4,0,1,2,3,0,1,2,0,1,0]) #Keep 28 coefficients  
    block_number=int(image.shape[0]*image.shape[1]/(block_rows*block_columns)) 
    number_of_coef=len(pos_x)
    coef=np.zeros((block_number,number_of_coef))
    #calculate blockwise DCT
    k=0
    for i in range(0, image.shape[0], block_rows):
        for j in range(0, image.shape[1], block_columns):
            block = image[i:i+block_rows, j:j+block_columns]
            dct_block = np.round(scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho'))
            c=dct_block[pos_x,pos_y].flatten()
            coef[k,:]=c
            k=k+1
    
    rows=int(image.shape[0]/block_rows)
    colls=int(image.shape[1]/block_columns)
    #rearrange coefficients to create homonymous coefficient matrices
    dct_images=coef[:,0].reshape(rows,colls)
    for j in range(1,coef.shape[1]):
      dct_images=np.concatenate((dct_images, coef[:,j].reshape(rows,colls)), axis=1)

    return dct_images

###
#This function applies the previous procedure onto a 3D image
###


def dct_3d(image,block_dimensions):
    #The 3 layers of an RGB image
    R=DCT(image[:,:,0], (block_dimensions[0],block_dimensions[1]) )
    G=DCT(image[:,:,1], (block_dimensions[0],block_dimensions[1]) )
    B=DCT(image[:,:,2], (block_dimensions[0],block_dimensions[1]) )
    
    dct_images= np.stack([R, G, B], axis=2)
    return dct_images 
