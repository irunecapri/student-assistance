#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/check_images_solution.py
#                                                                             
# PROGRAMMER: Adeline
# DATE CREATED: April 28, 2021                                     
# REVISED DATE: 


##

# Imports python modules
from time import time, sleep

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Imports functions created for this program
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results

# Main program function defined below
def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # Function that retrieves 3 Command Line Arugments from user's input
    in_arg = get_input_args()

    # Function that checks command line arguments using in_arg - 
    # Remove the # from in front of the function call after you have 
    # coded get_input_args to check your code
    check_command_line_arguments(in_arg)

    
    # Creates a dictionary that contains the results - called results
    results = get_pet_labels(in_arg.dir)

    # Function that checks Pet Images in the results Dictionary using results    
    # Remove the # from in front of the function call after you have 
    # coded get_pet_labels to check your code
    check_creating_pet_image_labels(results)

    
    # Creates Classifier Labels with classifier function, Compares Labels, 
    # and adds these results to the results dictionary - results
    classify_images(in_arg.dir, results, in_arg.arch)

    # Function that checks Results Dictionary - results    
    # Remove the # from in front of the function call after you have 
    # coded classify_images to check your code
    check_classifying_images(results)    

    
    # Adjusts the results dictionary to determine if classifier correctly 
    # classified images as 'a dog' or 'not a dog'. This demonstrates if 
    # model can correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(results, in_arg.dogfile)

    # Function that checks Results Dictionary for is-a-dog adjustment using results
    # Remove the # from in front of the function call after you have 
    # coded adjust_results4_isadog to check your code
    check_classifying_labels_as_dogs(results)

    
    # Calculates results of run and puts statistics in the Results Statistics
    # Dictionary - called results_stats
    results_stats = calculates_results_stats(results)

    # Function that checks Results Statistics Dictionary - results_stats
    # Remove the # from in front of the function call after you have 
    # coded calculates_results_stats to check your code
    check_calculating_results(results, results_stats)


    # Prints summary results, incorrect classifications of dogs
    # and breeds if requested
    print_results(results, results_stats, in_arg.arch, True, True)
    
    # Measure total program runtime by collecting end time
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

# Call to main function to run the program
if __name__ == "__main__":
    main()
