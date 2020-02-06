# -*- coding: utf-8 -*-
"""
Created in January 2020
@author: mlafarge

[Utilitaty functions for exercice 2]

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
    
    _figure = plt.figure(figsize=(12,3), facecolor=(1,1,1)) #-- w,h
    splt = lambda y,x,h=1,w=1,**kwargs: plt.subplot2grid(spShape, (y,x), colspan=w, rowspan=h, **kwargs)
    
    
    """ TRAINING DATA
    """
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
        self._figure = plt.figure(figsize=(10,5)) #-- w,h
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
            
            weights_select = layerWeights[0][:,:9] #-- # [input, out]
            weights_select = np.transpose(weights_select, axes=[1,0])
            weights_select = np.reshape(weights_select, [-1,64,64,3]) #-- 2d formatting
            
            weight_shape = weights_select.shape[1:] #-- h,w,c
            
            margin = 10
            tracking_canvas = 255.0 * np.ones([margin + 3 * (weight_shape[0] + margin), margin + 3 * (weight_shape[1] + margin), 3])
            for idx in range(9):
                yy = idx // 3
                xx = idx - 3 * yy
                
                target_slice_y = slice(margin + yy * (weight_shape[0] + margin), margin + yy * (weight_shape[0] + margin) + weight_shape[0]) 
                target_slice_x = slice(margin + xx * (weight_shape[1] + margin), margin + xx * (weight_shape[1] + margin) + weight_shape[1]) 
                
                weight_c = weights_select[idx,]
                #-- Normalization
                weight_c = weight_c / np.amax(np.abs(weight_c), axis=(0,1))
                weight_c = 255.0 * (weight_c + 1.0) / 2.0
                
                tracking_canvas[target_slice_y, target_slice_x, :] = weight_c
            
            self.tracking_canvas = tracking_canvas 
                
        if self._batch_count % self._refresh_steps == 0:
            self.updateFigure()
        
        
    def updateFigure(self):
        #plt.clf() # Clear figure
        
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

    