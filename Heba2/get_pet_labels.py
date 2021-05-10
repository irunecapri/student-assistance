 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Heba
# DATE CREATED:                                  
# REVISED DATE: 
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_ th dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
def get_pet_labels(image_dir): # THIS IS THE FUNCTION HEADER, EVERYTHING MUST BE PLACED INSIDE AND CORRECTLY INDENTED THE HEADER IS THE OUTER INDENT

    filename_list = listdir(image_dir) # MUST BE PLACEHOLDER ARGUMENT THAT WILL BE SET IN MAIN AND UTILIZED INSIDE THIS FUNCTION
    # THIS IS COMING FROM FUNCTION HEADER
    # print(filename_list)

    # Print 10 of the filenames from folder pet_images/
    #print("\nPrints 10 filenames from folder pet_images/") # LET ME GIVE YOU AN ADVICE ON THE WAY IN REGARDS TO NUMEROUS PRINT_STATEMENTS IN THE PROJECT: IT CAN BE A GOOD CONTROL MEASURE TO 'PRINT()' WHAT YOU HAVE CODED, BUT IF YOU DO NOT DELETE THOSE PRINTS AND ADD NEW ONES ON TOP OF THAT, THE MEASURE IS COUNTERPRODUCTIVE AS IT WILL BECOME HARD TO FOLLOW OUTPUT BECAUSE THERE ARE TOO MANY PRINTS!
                                                           # PLEASE HAVE THIS IN MIND WHEN DOING THIS, OR AT BEST DELETE THEM AFTER YOU HAVE CONTROLLED YOUR FUNCTIONS!
   #for idx in range(0, 40, 1):
    #print("{:2d} file: {:>25}".format(idx + 1, filename_list[idx]) )

    #create pet image labels
    #def pet_label(file_name): # DONT NEED AN INNER FUNCTION AND IT IS ALSO NOT THE RIGHT CONCEPT HERE

    results_dic = dict() # MUST CREATE MAIN DICTIONARY HERE, PLEASE SEE MINDMAP

    # Processes through each file in the directory, extracting only the words
    # of the file that contain the pet image label
    for idx in range(0, len(filename_list), 1): # TAKE THIS FROM THE HINTS FILE WHICH IS MEANT TO BE THE TEMPLATE
                                                # HERE YOU ITERATE THROUGH THE DATASET THAT CONTAINS THE IMAGES
        if filename_list[idx][0] != ".":        # EXCLUDING SYSTEMIC FILES FROM CONSIDERATION
            
            
            word_list = filename_list[idx].split("_") # LIST CANT BE LOWERED MUST BE DONE INSIDE THE FOR LOOP BELOW EHRE EACH WORD IS PROCESSED
            pet_name = ""

            for word in word_list:
                if word.isalpha():
                    pet_name += word.lower() + " " # .lower() MUST BE HERE
            pet_name = pet_name.strip() # NO RETURN HERE, THERE IS ONLY 1 RETURN AT THE END OF THE FUNCTION
                         # RETURNING RESULTS_DIC
                         # ALSO HERE YOU CANNOT MODIFIY "IN_PLACE" BUT MUST 

        ### MAJOR DELETIONS MUST BE DONE HERE, SECTION BELOW IS PART OF THE TEMPLATE AND ADDS
        ### ITEMS TO DICTIONARY, PLEASE SEE MINDMAP
    
            if filename_list[idx] not in results_dic:
                results_dic[filename_list[idx]] = [pet_name]
         
            else:
                print("** Warning: \"already exists in results_dic with value =", 
                   results_dic[filename_list[idx]])
    return results_dic