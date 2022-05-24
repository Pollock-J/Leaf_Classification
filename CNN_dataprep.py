#
# There are 3 datasets I am using for classifying leaf images
# Swedish - https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/
# Leafsnap - http://leafsnap.com/dataset/
# Kaggle - https://www.kaggle.com/competitions/leaf-classification/data
# In order to keep them the same, I will be segmenting one dataset
# I will be organizing a dataset to be able to better iterate through
# To prep the data to be fed into a CNN I will be squaring the images, then scaling them down
# I will also be augmenting 2 of the dataset that have between 1000-2000 images
#
import shutil
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.transform import rotate
import cv2
from PIL import Image
import numpy as np
import pandas.io.formats.excel


def swedish_leaves_segmentation():
    """
    This function segments the pictures from the swedish leaf dataset, this was to keep the data
    in the same format
    :return: path of file
    """
    # Setting a save location for the altered photos
    dest = '/home/li/PycharmProjects/Leaf Classification/swedish leaves/leaves segmented/'
    # Grabbing folder of images and looping through them
    leaves = glob.glob('/home/li/PycharmProjects/Leaf Classification/swedish leaves/leaves/*')
    for pic in leaves:
        img_data = plt.imread(pic)
        # Removing the file path
        picture = pic[67:-3]
        fn = str(dest + 's' + picture + 'png')
        # Other datasets have the leaves as white and rest black so we need to invert
        inverted_image = cv2.bitwise_not(img_data)
        # Otsu’s method of segmentation is being used because it is automatic
        # threshold_otsu needs images to already be in grayscale
        grayscale = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
        plt.imshow(grayscale)
        thresh = threshold_otsu(grayscale)
        binary = grayscale > thresh
        # This is the black and white image we need
        plt.imsave(fn, binary, cmap=plt.cm.gray)
    return dest


def reorganize():
    """
    Making it so that all segmented files are in one folder, and each file has their species name
    Also checking to see that the images are not blank, a few seem in leafsnap seem to be
    :return: Tally for total pictures, and the number of removed pictures
    """
    # Tally for pictures not copied because they were blank
    empty = 0  # leaf-snap 4352
    # Tally for total numer of pictures
    image = 0  # leaf-snap 30866
    # File location for dataset folders
    field = glob.glob('/home/li/PycharmProjects/Leaf Classification/leafsnap-dataset/dataset/segmented/field/*')
    lab = glob.glob('/home/li/PycharmProjects/Leaf Classification/leafsnap-dataset/dataset/segmented/lab/*')
    leaves100 = glob.glob('/home/li/PycharmProjects/Leaf Classification/100 leaves/data/*')
    ls_dest = '/home/li/PycharmProjects/Leaf Classification/leafsnap-dataset/dataset/segmented/all/'
    l100_dest = '/home/li/PycharmProjects/Leaf Classification/100 leaves/leaves/'
    folders = (leaves100, field, lab)
    for folder in folders:
        # Goes through all folders
        for species in folder:
            # Each folder labeled with the species name
            # This grabs the folder name, which was the leaf's name
            family = os.path.basename(os.path.realpath(species))
            family_path = os.path.realpath(species)
            pictures = glob.glob(family_path + '/*')
            for picture in pictures:
                image += 1
                # This will return none if there are no non-black pixels
                im = Image.open(picture)
                if not im.getbbox():
                    if folder != leaves100:
                        blank = '/home/li/PycharmProjects/Leaf Classification/leafsnap-dataset/dataset/segmented/blank/'
                        shutil.copy(picture, blank)
                        empty += 1
                else:
                    if folder == leaves100:
                        shutil.copy(picture, l100_dest)
                    else:
                        # This grabs the picture name, which is used in renaming
                        specimen = os.path.basename(os.path.realpath(picture))
                        shutil.copy(picture, ls_dest)
                        # Path of the picture that was just copied over
                        cfd = os.path.join(ls_dest, specimen)
                        # New name indicates the species, if it was from the lab it had ny in the name
                        pn = str(ls_dest + family + '-' + specimen)
                        os.rename(cfd, pn)
    return image, empty


def resize():
    """
    This will add black space to make the pictures squares, then it sizes them to 256x256
    :return: Returns True of false based on how many images are in the start and end folder
    """
    swedish_segmented = '/home/li/PycharmProjects/Leaf Classification/swedish leaves/leaves segmented/*'
    leaves100_segmented = '/home/li/PycharmProjects/Leaf Classification/100 leaves/leaves/*'
    # For the leafsnap-dataset using both segmented field and lab pictures
    leafsnap_segmented = '/home/li/PycharmProjects/Leaf Classification/leafsnap-dataset/dataset/segmented/all/*'
    segmented_datasets = (leaves100_segmented, swedish_segmented, leafsnap_segmented)
    for dataset in segmented_datasets:
        # Select the folder for resized images based on which dataset
        if dataset == swedish_segmented:
            dest = '/home/li/PycharmProjects/Leaf Classification/swedish leaves/resized_images/'
        elif dataset == leaves100_segmented:
            dest = '/home/li/PycharmProjects/Leaf Classification/100 leaves/resized_images/'
        elif dataset == leafsnap_segmented:
            dest = '/home/li/PycharmProjects/Leaf Classification/leafsnap-dataset/dataset/segmented/resized_images/'
        else:
            print('cannot set destination folder')
            break
        for leaf in glob.glob(dataset):
            # squaring each image
            image = Image.open(leaf)
            width, height = image.size
            square = width - height
            right = 0
            left = 0
            top = 0
            bottom = 0
            if square < 0:
                split = int(abs(square) / 2)
                right = split
                left = split
            else:
                split = int(abs(square) / 2)
                top = split
                bottom = split
            new_width = width + right + left
            new_height = height + top + bottom
            squaring = Image.new(image.mode, (new_width, new_height), 0)
            squaring.paste(image, (left, top))
            # Taking the squared image and scaling it
            scaled = squaring.resize((256, 256))
            # This is name for each picture
            fn = os.path.basename(os.path.realpath(leaf))
            # Creating the filename
            pic = (dest + fn)
            scaled.save(pic)
        if len(dataset) == len(dest):
            return True
        else:
            return False


def agument():
    """
    Using sklearn to rotate and flip the images, creating 5 new iteration for each image
    in the two data sets that had around 1,000 images
    :return:
    """
    swedish_resized = glob.glob('/home/li/PycharmProjects/Leaf Classification/swedish leaves/resized_images/*')
    leaves100_resized = glob.glob('/home/li/PycharmProjects/Leaf Classification/100 leaves/resized_images/*')
    datasets = (leaves100_resized, swedish_resized)
    for folder in datasets:
        for image in folder:
            # Load image into mplot
            img_data = plt.imread(image)
            # image rotation using skimage.transformation.rotate
            rotate30 = (rotate(img_data, angle=30))
            rotate45 = (rotate(img_data, angle=45))
            flip = np.flipud(img_data)
            flip_turn30 = np.flipud(rotate30)
            flip_turn45 = np.flipud(rotate45)
            trans = (rotate30, rotate45, flip, flip_turn30, flip_turn45)
            trans_names = ['rotate30-', 'rotate45-', 'flip-', 'flip_turn30-', 'flip_turn45-']
            count = -1
            for f in trans:
                count += 1
                # For the save file n and location
                # Transformation name
                tname = trans_names[count]
                # Folder name
                fpath = os.path.dirname(image)
                # Image name
                iname = os.path.basename(os.path.realpath(image))
                # Create new image name
                fn = str(fpath + '/' + tname + iname)
                # Save image
                plt.imsave(fn, f, cmap=plt.cm.gray)


def create_data_file():
    """
    Takes user input on whether the file will be Excel or Csv
    Makes a text file for each dataset so that the data can be fed into a cnn easily,
    with the features of plant type for all. Leafsnap will have if it was a field or lab pic,
    and the others will have augmentation if applicable.
    :return: the file location and name
    """
    leaves100 = '/home/li/PycharmProjects/Leaf Classification/100 leaves/resized_images/*'
    swedish = '/home/li/PycharmProjects/Leaf Classification/swedish leaves/resized_images/*'
    leafsnap = '/home/li/PycharmProjects/Leaf Classification/leafsnap-dataset/dataset/segmented/resized_images/*'
    pic_folders = leaves100, swedish, leafsnap
    ft = input("Type '1' for excel or '2' for csv:\n")
    if ft not in ("1", "2"):
        print("Invalid answer"), exit()
    complete = pd.DataFrame()
    for folder in pic_folders:
        # Sheet for data to be written to on file
        if folder == leaves100:
            sheet = '100 leaves'
            fn = str(folder[:-16] + "pic_data.csv")
        elif folder == swedish:
            sheet = 'Swedish leaves'
            fn = str(folder[:-16] + "pic_data.csv")
        else:
            sheet = 'Leafsnap'
            fn = str(folder[:-34] + "pic_data.csv")
        for pic in glob.glob(folder):
            pic_name = os.path.basename(os.path.realpath(pic))
            # makes a tuple with 3 parts, second being the separator '-'
            name_split = pic_name.partition('-')
            # leafsnap
            if folder == leafsnap:
                plant_type = name_split[0]
                if name_split[2][0].isdigit():
                    f_or_l = 'field'
                else:
                    f_or_l = 'lab'
            # augmented folders
            else:
                # augmented
                if '-' in pic_name:
                    augmentation = name_split[0]
                    # grabs after the '-' then removes the file type
                    if folder == leaves100:
                        pass
                        plant_type = name_split[2][:-10]
                    elif folder == swedish:
                        if name_split[2][3:4].isdigit():
                            plant_type = name_split[2][1:4]
                        else:
                            plant_type = name_split[2][1:3]
                # original pictures
                else:
                    augmentation = 'none'
                    if folder == leaves100:
                        plant_type = pic_name[:-10]
                    elif folder == swedish:
                        plant_type = pic_name[1:3]
            if folder == leafsnap:
                add = pd.DataFrame(np.array([[pic, pic_name, plant_type, f_or_l]]), columns=['Filepath', 'Picture Name', 'Plant Type', 'Lab or Field'])
            else:
                add = pd.DataFrame(np.array([[pic, pic_name, plant_type, augmentation]]), columns=['Filepath', 'Picture Name', 'Plant Type', 'Augmentation'])
            complete = complete.append(add, ignore_index=True)
        if ft == "1":
            writer = pd.ExcelWriter("pic_data.ods")
            complete.to_excel(writer, sheet_name=sheet, index=False)
            writer.save()
        if ft == "2":
            complete.to_csv(fn, sep='\t', index=False)
    return fn, "pic_data.ods"


if __name__ == '__main__':
    # swedish_leaves_segmentation() #finished
    # reorganize() #finished
    # resize() #finished
    # agument() #finished
    create_data_file()  # finished
    pass
