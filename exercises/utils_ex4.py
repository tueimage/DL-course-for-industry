# -*- coding: utf-8 -*-
"""
Created in January 2020
@author: mlafarge

[Utilitaty functions for exercice 4]

"""
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import os

#-- Plot Color
colors = ["b", "r", "g", "o", "p"]

def plot_image_list(
        list_images,
        list_titles):
    """ Generate a visualization of the dataset
    """
    batchSize = len(list_images)
    
    spShape = (1, batchSize) #-- h,w
    _figure = plt.figure(figsize=(10,3), facecolor=(1,1,1)) #-- w,h
    splt = lambda y,x,h=1,w=1,**kwargs: plt.subplot2grid(spShape, (y,x), colspan=w, rowspan=h, **kwargs)
    
    for idx in range(batchSize):
        img_c   = list_images[idx]
        title_c = list_titles[idx]
        
        ax = splt(0, idx, frameon=False) #-- y,x
        ax.imshow(img_c)
        ax.set_title(str(title_c),fontsize="small")
        ax.axis("off")
    
    """ FINAL RENDERING
    """  
    #plt.gcf().show()
    plt.gcf().canvas.draw()
    time.sleep(1e-3)   
    
def plot_image_batch(
        batch_images,
        batch_masks=None):
    """ Generate a visualization of the dataset
    """
    batchSize = batch_images.shape[0]
    
    spShape = (2, batchSize) #-- h,w
    _figure = plt.figure(figsize=(10,3), facecolor=(1,1,1)) #-- w,h
    splt = lambda y,x,h=1,w=1,**kwargs: plt.subplot2grid(spShape, (y,x), colspan=w, rowspan=h, **kwargs)
    
    for idx in range(batchSize):
        img_c  = batch_images[idx,]
        
        ax = splt(0, idx, frameon=False) #-- y,x
        ax.imshow(img_c) 
        ax.axis("off")
        if np.any(batch_masks):
            mask_c = batch_masks[idx,]
            ax = splt(1, idx, frameon=False) #-- y,x
            ax.imshow(mask_c)
            ax.axis("off")
        
        #ax.set_title(str(label_c),fontsize="small")
    
    """ FINAL RENDERING
    """  
    #plt.gcf().show()
    plt.gcf().canvas.draw()
    time.sleep(1e-3)   

def plot_image_sequence(
        tensor_images, # Expected shape [N, Himg, Wimg]
        tensor_masks): # Expected shape [M, Hmask, Wmask]

    
    # Prepare the joint images
    img_h, img_w   = tensor_images.shape[1:3]
    mask_h, mask_w = tensor_masks.shape[1:3]
    pad_y = (img_h - mask_h) // 2
    pad_x = (img_w - mask_w) // 2
    
    tensor_masks = np.pad(tensor_masks, ([0,0], [pad_y,pad_y], [pad_x,pad_x]))
    tensor_joined = np.concatenate([tensor_images, tensor_masks], axis=2) # concat along x-axis
          
    
    # First set up the figure, the axis, and the plot element we want to animate
    _figure = plt.figure() #-- w,h
    ax = plt.axes()
    ax.axis("off")
    canvas_anim_images = ax.imshow(tensor_joined[0,])
    
    # initialization function: plot the background of each frame
    def init():
        canvas_anim_images.set_data(tensor_joined[0,])
        return canvas_anim_images,
     
    # animation function.  This is called sequentially
    def animate(i):
        canvas_anim_images.set_data(tensor_joined[i,])
        return canvas_anim_images,
     
    # call the animator.  blit=True means only re-draw the parts that have changed.
    _anim_c = anim.FuncAnimation(_figure, animate, init_func=init,
                                frames=tensor_joined.shape[0], interval=50, blit=True, repeat=True)
    
    plt.tight_layout()
    plt.gcf().canvas.draw()
    
    closeFigure = lambda: plt.close(_figure)
    return closeFigure

def plot_featureMaps(
        inputImage, # expected shape [nb, H, W, 1]
        featureMap, # expected shape [nb, H, W, Channels]
        maxNbChannels = 8):
    """ Generate a visualization of an input image and corresponding feature maps:
    """
    
    # We transform the return tensor to the shape of batch of images for visualization purpose
    featureMap_reshaped = featureMap[0:1,] 
    featureMap_reshaped = np.transpose(featureMap_reshaped, [3,1,2,0])
    featureMap_reshaped = featureMap_reshaped - np.amin(featureMap_reshaped, axis=(1,2,3), keepdims=True)
    featureMap_reshaped = featureMap_reshaped / (1e-3 + np.amax(featureMap_reshaped, axis=(1,2,3), keepdims=True))
    featureMap_reshaped = 255.0 * featureMap_reshaped
    featureMap_reshaped = np.repeat(featureMap_reshaped, 3, axis=3)
    featureMap_reshaped = featureMap_reshaped.astype(np.uint8)
    
    spShape = (1, maxNbChannels + 1) #-- h,w
    plt.figure(figsize=(10,3)) #-- w,h
    
    ax = plt.subplot2grid(spShape, (0,0))
    ax.imshow(inputImage[0,:,:,0])
    plt.axis("off")
    ax.set_title("Input Image",fontsize="small")
    
    for i in range(maxNbChannels):
        ax = plt.subplot2grid(spShape, (0,1+i))
        ax.imshow(featureMap_reshaped[i,])
        plt.axis("off")
        ax.set_title("Feature map #{}".format(1+i),fontsize="small")

def submit_results(result_dictionary):
    """ This function checks the format of the result dictionary
        and store it at the right location to be read later on by the organizers.
    """
    expected_number_of_keys = 1193
    expected_number_of_dims = 2
    expected_shape = [352, 352]
    
    #-- Size check
    if len(result_dictionary) != expected_number_of_keys:
        print("Failure: expected number of masks = {} but received {}.".format(expected_number_of_keys, len(result_dictionary)))
        return None
    
    #-- Dimension check
    for k in result_dictionary:
        mask_shape = result_dictionary[k].shape
        
        if len(mask_shape) != expected_number_of_dims:
            print("Failure: key[{}] is expected to have {} dimensions but received {}.".format(expected_number_of_dims, len(mask_shape)))
            return None
        
        for dim_idx in range(expected_number_of_dims):
            if mask_shape[dim_idx] != expected_shape[dim_idx]:
                print("Failure: key[{}] is expected to be of shape {} but received {}.".format(expected_shape, mask_shape))
                return None
    
    
    #-- Finally to ensure correct formatting, the keys are turned to strings and masks to booleans
    result_dict_formatted = {}
    for k in result_dictionary:
        result_dict_formatted[str(k)] = result_dictionary[k].astype(np.bool)
        
        
    #-- Then submit
    path_root   = os.path.expanduser("~")
    path_target = path_root + os.sep + "results_testSet_submitted.npz"
    np.savez(path_target, **result_dict_formatted)
    
    print("Success: your results were successfully stored at [{}], good luck.".format(path_target))
    return result_dict_formatted
    
from IPython import display
class Monitoring(tf.keras.callbacks.Callback):
    """ Monitoring class herited from keras Callback structure
        Generates a dynamic plotting matplotlib figure to track the training losses.
    """
    def __init__(self,
                 model,
                 layerTracking=None,
                 validImage=None, validMask=None):
        super(tf.keras.callbacks.Callback, self).__init__()
        
        self._model = model
        
        
        self.layerTracking = layerTracking
        self.layerWeights_run = None
        
        self._valImage = None
        self._valMask  = None
        if np.any(validImage) and np.any(validMask):
            self._valImage = validImage
            self._valMask  = validMask
        

    def on_train_begin(self, logs=None):
        self._batch_count = 0
        self._train_loss_tracking = [] # 'batch': 0, 'acc': 0.505, 'loss'
        self._val_loss_tracking   = []
        
        spShape = (4, 4) #-- h,w
        self._figure = plt.figure(figsize=(10,6)) #-- w,h
        splt = lambda y,x,h=1,w=1,**kwargs: plt.subplot2grid(spShape, (y,x), colspan=w, rowspan=h, **kwargs)
        
        """ Loss Tracking
        """
        
        self._lossPlot = splt(0,0,2,2) #-- y,x
        
        """ Output Tracking
        """
        self._canvas_ori      = np.zeros([1,1])
        self._canvas_maskTrue = np.zeros([1,1])
        self._canvas_maskPred = np.zeros([1,1])
        self._trackingPlot_ori      = splt(0,2, frameon=False) #-- y,x
        self._trackingPlot_maskTrue = splt(1,2, frameon=False) #-- y,x
        self._trackingPlot_maskPred = splt(1,3, frameon=False) #-- y,x
        

    def on_train_batch_begin(self, batch, logs=None):
        self._batch_count += 1
    
    
    def on_train_batch_end(self, batch, logs=None):
        train_loss_c = logs["loss"]
        self._train_loss_tracking.append([self._batch_count, train_loss_c])
        
    def on_test_batch_begin(self, batch, logs=None):
        pass
    
    def on_test_batch_end(self, batch, logs=None):
        #-- LOSS TRACKING
        test_loss_c = logs["loss"]
        #test_loss_c = logs["acc"]
        self._val_loss_tracking.append([self._batch_count, test_loss_c])
                
        """ VALIDATION MASK CHECK
        """
        if  np.any(self._valImage) and np.any(self._valMask):
            img_batch = self._valImage / 255.0 #-- [0,1] scaling
            img_batch = np.expand_dims(img_batch, axis=0)
            img_batch = np.expand_dims(img_batch, axis=3)
            
            predictions = self._model.predict(img_batch) #-- Compute prediction
            predicted_mask = predictions[0, :, :, :]
            
            #-- Original Image
            self._canvas_ori = self._valImage
            
            #-- Original Mask
            mask_h, mask_w = self._valMask.shape[:2]
            newCanvas = np.zeros([mask_h, mask_w, 3])
            newCanvas[:, :, 0] = 255.0 * self._valMask
            self._canvas_maskTrue = newCanvas
            
            #-- Predicted Mask
            newCanvas = np.zeros([mask_h, mask_w, 3])
            pred_h, pred_w = predicted_mask.shape[:2]
            pred_pad_y = (mask_h - pred_h) // 2
            pred_pad_x = (mask_w - pred_w) // 2
            newCanvas[slice(pred_pad_y, pred_pad_y + pred_h), slice(pred_pad_x, pred_pad_x + pred_w), 0] = 255.0 * predicted_mask[:,:,0]
            self._canvas_maskPred = newCanvas
            
        self.updateFigure() #-- Update figure
        
        
    def updateFigure(self):        
        """ Clean loss plot
        """
        plt.sca(self._lossPlot)
        plt.cla()
                
        """ Draw Training Loss
        """
        plot_pts = self._train_loss_tracking
        pt_freq = 1
        plot_x = [pt[0] for pt in plot_pts[::pt_freq]]
        plot_y = [pt[1] for pt in plot_pts[::pt_freq]]
        
        plt.plot(plot_x, plot_y, label="Training Loss")
        
        """ Draw Validation Loss
        """
        plot_pts = self._val_loss_tracking
        pt_freq = 1
        plot_x = [pt[0] for pt in plot_pts[::pt_freq]]
        plot_y = [pt[1] for pt in plot_pts[::pt_freq]]
         
        plt.plot(plot_x, plot_y, label="Validation Loss")
        plt.legend()
        
        """ Tracked predictions
        """
        plt.sca(self._trackingPlot_ori); plt.cla()
        canvas_c = self._canvas_ori
        self._trackingPlot_ori.imshow(canvas_c)
        self._trackingPlot_ori.axis("off")
        self._trackingPlot_ori.set_title("Original Image", fontsize="small")
        
        plt.sca(self._trackingPlot_maskTrue); plt.cla()
        canvas_c = self._canvas_maskTrue.astype(np.uint8)
        self._trackingPlot_maskTrue.imshow(canvas_c)
        self._trackingPlot_maskTrue.axis("off")
        self._trackingPlot_maskTrue.set_title("True Mask", fontsize="small")
                
        plt.sca(self._trackingPlot_maskPred); plt.cla()
        canvas_c = self._canvas_maskPred.astype(np.uint8)
        self._trackingPlot_maskPred.imshow(canvas_c)
        self._trackingPlot_maskPred.axis("off")
        self._trackingPlot_maskPred.set_title("Predicted Mask", fontsize="small")
            
        # Update figure rendering   
        plt.gcf().canvas.draw()
        
        display.clear_output(wait=True)        
        display.display(plt.gcf())
        
        time.sleep(1e-3)  
    
    