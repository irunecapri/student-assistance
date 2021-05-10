#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# *udacity-dog-breed-project/intropyproject-classify-pet-images/classify_images.py
#
# PROGRAMMER: Luke Wilson
# DATE CREATED: 2021-05-03
# REVISED DATE: 2021-05-04
# TODO 3: Define classify_images functions

from classifier import classifier #import classifier function

def classify_images(images_dir, results_dic, model):
    """
    Creates classifier labels with classifier function, compares pet labels to
    the classifier labels, and adds the classifier label and the comparison of
    the labels to the results dictionary using the extend function. This function
    uses the classifier() function defined in classifier.py within this function.
     Parameters:
      images_dir - The (full) path to the folder of images that are to be
                   classified by the classifier function (string)
      results_dic - Results Dictionary with 'key' as image filename and 'value'
                    as a List. Where the list will contain the following items:
                  index 0 = pet image label (string)
                --- where index 1 & index 2 are added by this function ---
                  NEW - index 1 = classifier label (string)
                  NEW - index 2 = 1/0 (int)  where 1 = match between pet image
                    and classifer labels and 0 = no match between labels
      model - Indicates which CNN model architecture will be used by the
              classifier function to classify the pet images,
              values must be either: resnet, alexnet, or vgg (string)
     Returns:
           None - results_dic is mutable data type so no return needed.
    """

    for key in results_dic: #iterate through each key (filename) in the pets dictionary from get_pets_labels
        model_label = classifier(images_dir + key, model).lower().strip() #return classified label(s) for each filename and format them
       # DATA TYPE THAT WILL BE RETURNED BY CLASSIFIER IS STRING AND MUST STAY STRING UNTIL BEEING ADDED TO LIST OF VALUES FOR EACH IMAGE/KEY
        truth = results_dic[key][0]
       

        if truth in model_label:
            results_dic[key].extend((model_label, 1)) # METHODS TO ADD ITEMS TO LIST ARE MAINLY .APPEND() AND .EXTEND(), += IS IN NUMERICAL OPERATIONS AND OTHERS
                   
        else:
            results_dic[key].extend((model_label, 0))