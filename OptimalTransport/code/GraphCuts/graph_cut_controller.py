import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from tkinter import *
from PIL import Image

from graph_cut import GraphCut
from graph_cut_gui import GraphCutGui
import os


class GraphCutController:

    def __init__(self, args):
        self.__init_view(args.autoload)

    def __init_view(self, autoload=None):
        root = Tk()
        root.geometry("700x500")
        self._view = GraphCutGui(self, root)

        if autoload is not None:
            self._view.autoload(autoload)

        root.mainloop()

    # TODO: TASK 2.1
    def __get_color_histogram(self, image, seed, hist_res):
        """
        Compute a color histograms based on selected points from an image
        
        :param image: color image
        :param seed: Nx2 matrix containing the the position of pixels which will be
                    used to compute the color histogram
        :param histRes: resolution of the histogram
        :return hist: color histogram
        """

        sigma = 0.1
        img_segment = image[seed[:, 1], seed[:, 0]]
        hist, _ = np.histogramdd(img_segment, bins=hist_res, range=[(0, 255), (0, 255), (0, 255)])
        hist = ndimage.gaussian_filter(hist, sigma=sigma)
        return hist / np.sum(hist)



    # TODO: TASK 2.2
    # Hint: Set K very high using numpy's inf parameter
    def __get_unaries(self, image, lambda_param, hist_fg, hist_bg, seed_fg, seed_bg):
        """

        :param image: color image as a numpy array
        :param lambda_param: lamdba as set by the user
        :param hist_fg: foreground color histogram
        :param hist_bg: background color histogram
        :param seed_fg: pixels marked as foreground by the user
        :param seed_bg: pixels marked as background by the user
        :return: unaries : Nx2 numpy array containing the unary cost for every pixels in I (N = number of pixels in I)
        """
        eps = 1e-10
        h,w,_ = image.shape
        hist_step = 256 // hist_fg.shape[0]
        image = (image // hist_step).reshape(-1,3).T
        unaries = np.zeros((image.shape[1], 2))
        fg_indices = [ph*w + pw for pw, ph in seed_fg]
        bg_indices = [ph*w + pw for pw, ph in seed_bg]

        pr_fg = -lambda_param * np.log(hist_fg[image[0], image[1], image[2]] + eps)
        pr_bg = -lambda_param * np.log(hist_bg[image[0], image[1], image[2]] + eps)
        unaries = np.stack((pr_bg, pr_fg), axis=-1)
        unaries[fg_indices] = [np.inf, 0]
        unaries[bg_indices] = [0, np.inf]
        return unaries
        

    # TODO: TASK 2.3
    # Hint: Use coo_matrix from the scipy.sparse library to initialize large matrices
    # The coo_matrix has the following syntax for initialization: coo_matrix((data, (row, col)), shape=(width, height))
    def __get_pairwise(self, image):
        """
        Get pairwise terms for each pairs of pixels on image
        :param image: color image as a numpy array
        :return: pairwise : sparse square matrix containing the pairwise costs for image
        """
        sigma = 5
        h,w,_ = image.shape
        image = image.astype(np.float32).reshape(-1,3)
        num_pixels = image.shape[0]

        num_neigbors = (h-2)*(w-2)*8 + (h-2)*2*5 + (w-2)*2*5 + 12
        data, rows, cols = np.zeros(num_neigbors), np.zeros(num_neigbors), np.zeros(num_neigbors)

        n = 0
        shift_all = [-w-1, -w, -w+1, -1, 1, w-1, w, w+1]
        for idx, p in enumerate(image):
            row_img, col_img = idx // w, idx % w
            shift_arr = [-w, -w+1, 1, w, w+1] if col_img == 0 else [-w-1, -w, -1, w-1, w] if col_img == w-1 else shift_all
            q_idx = idx + np.array(shift_arr)
            q_idx = q_idx[(q_idx >= 0) & (q_idx < num_pixels)]

            value_diff = np.linalg.norm((p - image[q_idx]), axis=1)
            dist = np.sqrt((row_img - q_idx // w) ** 2 + (col_img - q_idx % w) ** 2)

            rows[n:n + len(q_idx)] = idx
            cols[n:n + len(q_idx)] = q_idx
            data[n:n + len(q_idx)] = np.exp(-value_diff**2/(2*sigma**2)) / dist
            n += len(q_idx)

        return coo_matrix((data, (rows, cols)), shape=(num_pixels, num_pixels))


    # TODO TASK 2.4 get segmented image to the view
    def __get_segmented_image(self, image, labels, background=None):
        """
        Return a segmented image, as well as an image with new background 
        :param image: color image as a numpy array
        :param label: labels a numpy array
        :param background: color image as a numpy array
        :return image_segmented: image as a numpy array with red foreground, blue background
        :return image_with_background: image as a numpy array with changed background if any (None if not)
        """
        h, w, chnls = image.shape
        image = image.reshape(-1,3)
        image_segmented = image.copy()
        image_segmented[labels == 0] = image[labels == 0] * [1, 0, 0]
        image_segmented[labels == 1] = image[labels == 1] * [0, 0, 1]

        image_with_background = None
        if background is not None:
            image_with_background = background[:h, :w].reshape(-1,3).copy()
            image_with_background[labels == 0] = image[labels == 0] 
            image_with_background = image_with_background.reshape(h, w, chnls)

        return image_segmented.reshape(h, w, chnls), image_with_background


    def segment_image(self, image, seed_fg, seed_bg, lambda_value, background=None):
        image_array = np.asarray(image)
        background_array = None
        if background:
            background_array = np.asarray(background)
        seed_fg = np.array(seed_fg)
        seed_bg = np.array(seed_bg)
        height, width = np.shape(image_array)[0:2]
        num_pixels = height * width

        # TODO: TASK 2.1 - get the color histogram for the unaries
        hist_res = 32
        cost_fg = self.__get_color_histogram(image_array, seed_fg, hist_res)
        cost_bg = self.__get_color_histogram(image_array, seed_bg, hist_res)

        # TODO: TASK 2.2-2.3 - set the unaries and the pairwise terms
        unaries = self.__get_unaries(image_array, lambda_value, cost_fg, cost_bg, seed_fg, seed_bg)
        pairwise = self.__get_pairwise(image_array)

        # TODO: TASK 2.4 - perform graph cut
        # Your code here
        num_neigbors = (height-2)*(width-2)*8 + (height-2)*2*5 + (width-2)*2*5 + 12
        graph_cut = GraphCut(num_pixels, num_neigbors)
        graph_cut.set_unary(unaries)
        graph_cut.set_neighbors(pairwise)

        graph_cut.minimize()
        labels = graph_cut.get_labeling()

        # TODO TASK 2.4 get segmented image to the view
        segmented_image, segmented_image_with_background = self.__get_segmented_image(image_array, labels,
                                                                                      background_array)
        # transform image array to an rgb image
        segmented_image = Image.fromarray(segmented_image, 'RGB')
        self._view.set_canvas_image(segmented_image)
        if segmented_image_with_background is not None:
            segmented_image_with_background = Image.fromarray(segmented_image_with_background, 'RGB')
            plt.imshow(segmented_image_with_background)
            plt.show()
