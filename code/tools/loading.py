import os
import csv
from config import DATA

# def load_nsd_images():
#     """
#     Loads the file paths of natural scene images from the NSD_IMAGES directory.

#     Returns:
#         list: A sorted list of full paths to the natural scene images.
#     """
#     NSD_IMAGES = os.path.join(DATA,'naturalscenes','images')
#     return sorted([os.path.join(NSD_IMAGES,image) for image in os.listdir(NSD_IMAGES)])

def load_nsd_images():
    """
    Loads the file paths of natural scene images from the NSD_IMAGES directory.

    Returns:
        list: A sorted list of full paths to the natural scene images.
    """
    NSD_IMAGES = os.path.join(DATA,'naturalscenes','images')
    return sorted([os.path.join(NSD_IMAGES,image) for image in os.listdir(NSD_IMAGES)])[::10]

def load_majaj_images(demo=False):
    """
    Loads the file paths of images from the MAJAJ_IMAGES directory.

    Returns:
        list: A sorted list of full paths to the images in the MAJAJ_IMAGES directory.
    """
    MAJAJ_IMAGES = os.path.join(DATA,'majajhong','image_dicarlo_hvm-public')
    if demo:
        return sorted([f'{MAJAJ_IMAGES}/{image}' for image in os.listdir(MAJAJ_IMAGES)])[:50]
    
    return sorted([f'{MAJAJ_IMAGES}/{image}' for image in os.listdir(MAJAJ_IMAGES)])


def load_places_val_images(demo=False):
    """
    Loads the file paths of validation images from the PLACES_IMAGES directory.

    Returns:
        list: A sorted list of full paths to the validation images.
    """
    PLACES_IMAGES = os.path.join(DATA,'places')
    images = os.listdir(os.path.join(PLACES_IMAGES,'val_images/val_256'))
    images_paths = [f'{PLACES_IMAGES}/val_images/val_256/{i}' for i in images]

    if demo:
        return sorted(images_paths)[:500]
    
    return sorted(images_paths)
      

def load_places_train_images(demo=False):
    PLACES_IMAGES = os.path.join(DATA,'places')
    images_paths = []
    base_dir = os.path.join(PLACES_IMAGES,'train_images_subset')
    
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    print(len(subdirs))
    for subdir in subdirs:
        # List all files in the subdirectory, including their full paths
        images_paths.extend([os.path.join(subdir, f) for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))])

    if demo:
        return sorted(images_paths)[:500]
    
    return sorted(images_paths)


def load_image_paths(dataset_name): 
    
    """
    Load image file paths based on a specified dataset name.

    Args:
        name (str): Name of the dataset ('naturalscenes', 'majajhong', or 'places').

    Returns:
        list: A sorted list of full paths to the images for the specified dataset.
    """
    
    match dataset_name:
        
        case 'naturalscenes' | 'naturalscenes_shuffled':
            return load_nsd_images()
            
        case 'majajhong' | 'majajhong_shuffled':
            return load_majaj_images()

        case 'majajhong_demo' | 'majajhong_demo_shuffled':
            return load_majaj_images(demo=True)
        
        case 'places_val':
            return load_places_val_images()
    
        case 'places_train':
            return load_places_train_images()

        case 'places_val_demo':
            return load_places_val_images(demo=True)
    
        case 'places_train_demo':
            return load_places_train_images(demo=True)
        

def get_image_labels(dataset_name, image_paths):
    
    """
    Get image labels based on a specified dataset.

    Args:
        dataset (str): Name of the dataset ('naturalscenes', 'majajhong', or 'places').
        images (list): List of image file paths for which to obtain labels.

    Returns:
        list: List of labels corresponding to the provided images.
    """
    
    match dataset_name:
        
        case 'naturalscenes' | 'naturalscenes_shuffled':
            return [os.path.basename(i).strip('.png') for i in image_paths]
        
        case 'majajhong' | 'majajhong_shuffled' | 'majajhong_demo' | 'majajhong_demo_shuffled':
            MAJAJ_NAME_DICT = os.path.join(DATA,'majajhong','image_dicarlo_hvm-public.csv')
            # name_dict = pd.read_csv(MAJAJ_NAME_DICT).set_index('image_file_name')['image_id'].to_dict()\
            name_dict = {}
            with open(MAJAJ_NAME_DICT, mode='r') as infile:
                reader = csv.DictReader(infile)
                name_dict = {rows['image_file_name']: rows['image_id'] for rows in reader}
            return [name_dict[os.path.basename(i)] for i in image_paths]
        
        case 'places_train' | 'places_train_demo' :
            return [multi_level_basename(i) for i in image_paths]
                                                  
        case 'places_val' | 'places_val_demo':
            return [os.path.basename(i).strip('.jpg') for i in image_paths]


def load_places_cat_labels():
    """
    Load category labels for placees dataset.

    Returns:
        dict: Dictionary where keys are image filenames and values are category labels.
    """    
    PLACES_IMAGES = os.path.join(DATA,'places')
    with open(os.path.join(PLACES_IMAGES,'places365_val.txt'), "r") as file:
        content = file.read()
    annotations = content.split('\n')
    cat_dict = {}
    for annotation in annotations:
        image = annotation.split(' ')[0]
        cat = annotation.split(' ')[1]
        cat_dict[image] = int(cat)
    return cat_dict   

            

def multi_level_basename(full_path, levels=2):
    full_path = os.path.normpath(full_path)
    path_parts = full_path.split(os.sep)
    if len(path_parts) >= levels:
        result = os.path.join(*path_parts[-levels:])
    else:
        result = os.path.join(*path_parts)
    return result


# def load_places_cat_ids(demo=False):
#     """
#     Load category names for the places dataset.

#     Returns:
#         list: List of category names for the validation images in the PLACES_IMAGES dataset.
#     """    
#     if demo:
#         val_image_paths = load_places_val_images(demp=True)
#     else:
#         val_image_paths = load_places_val_images()
#     val_image_names = [os.path.basename(i) for i in val_image_paths]
#     cat_dict = load_places_cat_labels()

#     return [cat_dict[i] for i in val_image_names]              


