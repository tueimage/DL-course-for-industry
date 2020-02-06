# -*- coding: utf-8 -*-
"""
Created in January 2020
@author: M Lafarge (m.w.lafarge@tue.nl)

[DL-course: Utilitary functions for exercise 1]

"""

import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Colors for plots
colors = ["b", "r", "g", "o", "p"]

def plot_2d_dataset(
        points_training,
        labels_training,

        points_validation = None,
        labels_validation = None,

        backgroundImage   = None):
    """ Generate a visualization of the dataset
    """
    spShape = (1, 2) # h,w
    _figure = plt.figure(figsize=(6,3), facecolor=(1,1,1)) # w,h
    splt = lambda y,x,h=1,w=1,**kwargs: plt.subplot2grid(spShape, (y,x), colspan=w, rowspan=h, **kwargs)

    # Other parameters
    scatter_size = 15

    
    ax = splt(0,0) #-- y,x
    plt.title("Training Set")

    if np.any(backgroundImage):
        ax.imshow(
            backgroundImage[::-1,], #-- y-axis flip
            extent=[-2,2,-2,2])

    labels_dict = {}
    for label in labels_training:
        if label not in labels_dict:
            labels_dict[label] = len(labels_dict)

    for label_c in labels_dict:
        x_values = [pt[1] for pt, className in zip(points_training, labels_training) if className == label_c]
        y_values = [pt[0] for pt, className in zip(points_training, labels_training) if className == label_c]

        plt.scatter(
            x_values,
            y_values,

            marker     = "o",
            edgecolors = "w",

            s      = scatter_size,
            color  = colors[labels_dict[label_c]],

            label  = "Class {}".format(label_c))

    # Legend
    plt.legend()

    # Axes
    plt.xlabel("$x_1$", fontsize=10)
    plt.ylabel("$x_2$", fontsize=10)

    ax.tick_params(
        top    = False,
        bottom = True,
        left   = True,
        right  = False,
        labelleft   = True,
        labelbottom = True)

    if np.any(points_validation) and np.any(labels_validation):
        ax = splt(0,1) #-- y,x
        plt.title("Test Set")

        if np.any(backgroundImage):
            ax.imshow(
                backgroundImage[::-1,], #-- y-axis flip
                extent=[-2,2, -2,2]) #-- left, right, bottom, top

        labels_dict = {}
        for label in labels_training:
            if label not in labels_dict:
                labels_dict[label] = len(labels_dict)

        for label_c in labels_dict:
            x_values = [pt[1] for pt, className in zip(points_validation, labels_validation) if className == label_c]
            y_values = [pt[0] for pt, className in zip(points_validation, labels_validation) if className == label_c]

            plt.scatter(
                x_values,
                y_values,

                marker     = "o",
                edgecolors = "w",

                s      = scatter_size,
                color  = colors[labels_dict[label_c]],

                label  = "Class {}".format(label_c))

        # Legend
        plt.legend()

        # Axes
        plt.xlabel("$x_1$", fontsize=10)
        plt.ylabel("$x_2$", fontsize=10)

        ax.tick_params(
            top    = False,
            bottom = True,
            left   = True,
            right  = False,
            labelleft   = True,
            labelbottom = True)

    """ FINAL RENDERING
    """
    #plt.gcf().show()
    plt.gcf().canvas.draw()
    time.sleep(1e-3)


def plot_learned_landscape(
        model,
        points_training   = None,
        labels_training   = None,
        points_validation = None,
        labels_validation = None):
    """ Generate a grid of points and feed it to the model
        Visualize data points on the learned prediction landscape
    """

    """ FIRST GENERATE A POINT GRID
    """
    bg_size = 100

    bg_points = np.mgrid[0:bg_size, 0:bg_size] #-- [(yx), y, x]
    bg_points = np.transpose(bg_points, [1,2,0]) #-- [y, x, (yx)]

    # Grid flattening
    bg_points_lin = np.reshape(bg_points, [bg_size*bg_size, -1])
    bg_points_lin = 2.0 * (2*bg_points_lin/(bg_size-1) - 1.0)

    # Application on bg grid
    bg_pred_flat = model.predict(
        bg_points_lin,
        batch_size = 32)

    bg_pred = np.reshape(bg_pred_flat[:, 0], [bg_size, bg_size])

    color_start = 255.0 * np.array([0.5, 0.5, 1.0])
    color_end   = 255.0 * np.array([1.0, 0.5, 0.5])

    bg_pred = np.stack([bg_pred]*3, axis=2)
    bg_pred = (1.0 - bg_pred) * color_start + bg_pred * color_end
    bg_pred = bg_pred.astype(np.uint8)

    """ VISUALIZATION
    """
    plot_2d_dataset(
        points_training,
        labels_training,

        points_validation,
        labels_validation,

        bg_pred)



from IPython import display
class Monitoring(tf.keras.callbacks.Callback):
    """ Monitoring class herited from keras Callback structure
        Generates a dynamic plotting matplotlib figure to track the training losses.
    """
    def __init__(self,
                 y_range=[-0.1, 1.0],
                 x_range=[0, 10000],
                refresh_steps = 100):
        super(tf.keras.callbacks.Callback, self).__init__()

        self._y_range = y_range
        self._x_range = x_range
        self._refresh_steps = refresh_steps

    def on_train_begin(self, logs=None):
        self._batch_count = 0
        self._train_loss_tracking = []
        self._val_loss_tracking   = []

        _figure = plt.figure(figsize=(12,3)) # w,h
        splt = lambda y,x,h=1,w=1,**kwargs: plt.subplot2grid((1,1), (y,x), colspan=w, rowspan=h, **kwargs)
        self._lossPlot = splt(0,0)


    def on_train_batch_begin(self, batch, logs=None):
        self._batch_count += 1


    def on_train_batch_end(self, batch, logs=None):
        train_loss_c = logs["loss"]
        self._train_loss_tracking.append([self._batch_count, train_loss_c])
        #print("Batch [{}] : loss = {}".format(self._batch_count, train_loss_c))


    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        test_loss_c = logs["loss"]
        self._val_loss_tracking.append([self._batch_count, test_loss_c])
        if self._batch_count % self._refresh_steps == 0:
            self.updateFigure()


    def updateFigure(self):
        plt.clf() # Clear figure

        # Training loss
        plot_pts = self._train_loss_tracking
        pt_freq = 1
        plot_x = [pt[0] for pt in plot_pts[::pt_freq]]
        plot_y = [pt[1] for pt in plot_pts[::pt_freq]]

        plt.plot(plot_x, plot_y, label="Training Loss")

        # Test loss
        plot_pts = self._val_loss_tracking
        pt_freq = 1
        plot_x = [pt[0] for pt in plot_pts[::pt_freq]]
        plot_y = [pt[1] for pt in plot_pts[::pt_freq]]

        plt.plot(plot_x, plot_y, label="Test Loss")


        plt.xlim(self._x_range)
        plt.ylim(self._y_range)
        plt.xlabel("iteration batches", fontsize=10)
        plt.ylabel("loss values", fontsize=10)

        # Update figure rendering
        plt.legend()
        plt.gcf().canvas.draw()

        display.clear_output(wait=True)
        display.display(plt.gcf())

        time.sleep(1e-3)
