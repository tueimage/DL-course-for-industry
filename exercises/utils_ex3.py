# -*- coding: utf-8 -*-
"""
Created in January 2020
@author: mlafarge

[Utilitaty functions for exercice 3]

"""
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#-- Plot Color
colors = ["b", "r", "g", "o", "p"]

def plot_image_batch(
        batch_images,
        batch_labels):
    """ Generate a visualization of the dataset
    """
    batchSize = batch_images.shape[0]
    
    spShape = (1, batchSize) #-- h,w
    _figure = plt.figure(figsize=(16,3), facecolor=(1,1,1)) #-- w,h
    splt = lambda y,x,h=1,w=1,**kwargs: plt.subplot2grid(spShape, (y,x), colspan=w, rowspan=h, **kwargs)
    
    for idx in range(batchSize):
        img_c   = batch_images[idx,]
        label_c = batch_labels[idx]
        
        ax = splt(0, idx, frameon=False) #-- y,x
        ax.imshow(img_c) #-- left, right, bottom, top
        
        #-- Axes
        ax.axis("off")
        ax.set_title(str(label_c),fontsize="small")
    
    """ FINAL RENDERING
    """  
    #plt.gcf().show()
    plt.gcf().canvas.draw()
    time.sleep(1e-3)   
    

def plot_featureMaps(
        inputImage, # expected shape [nb, H, W, 3]
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
    plt.figure(figsize=(15,3)) #-- w,h
    
    ax = plt.subplot2grid(spShape, (0,0))
    ax.imshow(inputImage[0,])
    plt.axis("off")
    ax.set_title("Input Image",fontsize="small")
    
    for i in range(maxNbChannels):
        ax = plt.subplot2grid(spShape, (0,1+i))
        ax.imshow(featureMap_reshaped[i,])
        plt.axis("off")
        ax.set_title("Feature map #{}".format(1+i),fontsize="small")
    
    
from IPython import display
class Monitoring(tf.keras.callbacks.Callback):
    """ Monitoring class herited from keras Callback structure
        Generates a dynamic plotting matplotlib figure to track the training losses.
    """
    def __init__(self,
            layerTracking=None,
            
            y_range=[-0.1, 1.0],
            x_range=[0, 10000],
            refresh_steps = 100):
        super(tf.keras.callbacks.Callback, self).__init__()
    
        self.layerTracking = layerTracking
        self.layerWeights_run = None
        
        self._y_range = y_range
        self._x_range = x_range
        self._refresh_steps = refresh_steps
        

    def on_train_begin(self, logs=None):
        self._batch_count = 0
        self._train_loss_tracking = [] # 'batch': 0, 'acc': 0.505, 'loss'
        self._val_loss_tracking   = []
        
        spShape = (1, 2) #-- h,w
        self._figure = plt.figure(figsize=(15,5)) #-- w,h
        splt = lambda y,x,h=1,w=1,**kwargs: plt.subplot2grid(spShape, (y,x), colspan=w, rowspan=h, **kwargs)
        
        """ Loss Tracking
        """
        self._lossPlot = splt(0,0) #-- y,x
        
        """ Weight Tracking
        """
        self.tracking_canvas = None
        self._trackingPlot = splt(0,1, frameon=False) #-- y,x
        
    def on_train_batch_begin(self, batch, logs=None):
        self._batch_count += 1
    
    
    def on_train_batch_end(self, batch, logs=None):
        train_loss_c = logs["loss"]
        self._train_loss_tracking.append([self._batch_count, train_loss_c])
        
    def on_test_batch_begin(self, batch, logs=None):
        pass
    
    def on_test_batch_end(self, batch, logs=None):        
        test_loss_c = logs["loss"]
        #test_loss_c = logs["acc"]
        self._val_loss_tracking.append([self._batch_count, test_loss_c])
        
        if self.layerTracking:
            layerWeights = self.layerTracking.get_weights()
            
            weights_select  = layerWeights[0] #-- # [h, w, in, out]            
            nb_kernels      = weights_select.shape[-1]
            nb_kernels_sqrt = int(np.sqrt(nb_kernels)) + 1
            weight_shape    = weights_select.shape[0:3] #-- h,w,c
            
            margin = 2
            nb_colrows = nb_kernels_sqrt
            tracking_canvas = 255.0 * np.ones([margin + nb_colrows * (weight_shape[0] + margin), margin + nb_colrows * (weight_shape[1] + margin), 3])
            for idx in range(nb_kernels):
                yy = idx // nb_colrows
                xx = idx - nb_colrows * yy
                
                target_slice_y = slice(margin + yy * (weight_shape[0] + margin), margin + yy * (weight_shape[0] + margin) + weight_shape[0]) 
                target_slice_x = slice(margin + xx * (weight_shape[1] + margin), margin + xx * (weight_shape[1] + margin) + weight_shape[1]) 
                
                weight_c = weights_select[:,:,:,idx] #-- Selected kernel
                
                #-- Normalization
                #weight_c = weight_c - np.amin(weight_c, axis=(0,1))
                #weight_c = weight_c / np.amax(weight_c, axis=(0,1))
                weight_c = weight_c / np.amax(np.abs(weight_c), axis=(0,1))
                weight_c = 255.0 * (weight_c + 1.0) / 2.0
                tracking_canvas[target_slice_y, target_slice_x, :] = weight_c
            
            self.tracking_canvas = tracking_canvas 
                
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
        
        """ Clean tracking canvas
        """
        plt.sca(self._trackingPlot)
        plt.cla()
        
        """ Draw Weight tracking
        """
        if np.any(self.tracking_canvas):
            self.tracking_canvas = self.tracking_canvas.astype(np.uint8)
            plt.imshow(self.tracking_canvas)
        self._trackingPlot.axis("off")
        self._trackingPlot.set_title("Weight Tracking", fontsize="small")

        
        # Update figure rendering   
        plt.gcf().canvas.draw()
        
        display.clear_output(wait=True)        
        display.display(plt.gcf())
        
        time.sleep(1e-3)  

    