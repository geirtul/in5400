import os
import json
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import pickle

class DataLoaderWrapper():
    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.data_dir_train = self.data_dir + '/train2017/'
        self.data_dir_val   = self.data_dir + '/val2017/'

        self.batch_size_train = 1
        self.batch_size_val   = 1
        self.imgSize          = [224, 224]

        self.records_list_train = self.load_records(is_train=True)
        self.records_list_val   = self.load_records(is_train=False)

        myDatasetTrain = CocoDataset(self.records_list_train, self.imgSize, self.data_dir_train)
        myDatasetVal   = CocoDataset(self.records_list_val, self.imgSize, self.data_dir_val)

        self.myDataLoaderTrain = DataLoader(myDatasetTrain, batch_size=self.batch_size_train, shuffle=False, num_workers=0)
        self.myDataLoaderVal   = DataLoader(myDatasetVal, batch_size=self.batch_size_val, shuffle=False, num_workers=0)



        return

    ########################################################################
    def load_records(self, is_train=True):
        if is_train:
            # Training-set.
            filename = "captions_train2017.json"
        else:
            # Validation-set.
            filename = "captions_val2017.json"

        # Full path for the data-file.
        path = os.path.join(self.data_dir, "annotations", filename)

        # Load the file.
        # path ='../data/coco/annotations/captions_train2017.json'
        with open(path, "r", encoding="utf-8") as file:
            data_raw = json.load(file)

        # Convenience variables.
        images      = data_raw['images']
        annotations = data_raw['annotations']

        # Initialize the dict for holding our data.
        # The lookup-key is the image-id.
        records = dict()

        # Collect all the filenames for the images.
        for image in images:
            # Get the id and filename for this image.
            image_id = image['id']
            filename = image['file_name']

            # Initialize a new data-record.
            record = dict()

            # Set the image-filename in the data-record.
            record['filename'] = filename

            # Initialize an empty list of image-captions
            # which will be filled further below.
            record['captions'] = list()

            # Save the record using the the image-id as the lookup-key.
            records[image_id] = record

        # Collect all the captions for the images.
        for ann in annotations:
            # Get the id and caption for an image.
            image_id = ann['image_id']
            caption  = ann['caption']

            # Lookup the data-record for this image-id.
            # This data-record should already exist from the loop above.
            record = records[image_id]

            # Append the current caption to the list of captions in the
            # data-record that was initialized in the loop above.
            record['captions'].append(caption)

        # Convert the records-dict to a list of tuples.
        coco_records_list = [(key, record['filename'], record['captions'])
                        for key, record in sorted(records.items())]

        # Convert the list of tuples to separate tuples with the data.
        #ids, filenames, captions = zip(*records_list)
        return coco_records_list

    ########################################################################
    def generate_vocabulary(self):
        from utils_data_preparation.generateVocabulary import generateVocabulary
        filename = self.data_dir + 'vocabulary/vocabulary.pickle'
        if not os.path.isfile(filename):
            generateVocabulary(self.data_dir, self.records_list_train, self.records_list_val)
        else:
            print('The file "vocabulary.pickle" has already been produced.')
        return

    ########################################################################
    def loadVocabulary(self):
        filename = self.data_dir + 'vocabulary/vocabulary.pickle'
        with open(filename, "rb") as input_file:
            vocabularyDict = pickle.load(input_file)
        return vocabularyDict



#######################################################################################################################
#######################################################################################################################
class CocoDataset():
    def __init__(self, records_list, imgSize, dir):

        self.records_list = records_list
        self.imgSize      = imgSize
        self.dir          = dir
        return

    def __len__(self):
        return len(self.records_list)

    def __getitem__(self, item):
        path     =  self.dir + self.records_list[item][1]
        fileName = self.records_list[item][1]
        captions = self.records_list[item][2]

        # Load the image using PIL.
        img  = Image.open(path)

        # Resize image if desired.
        img = img.resize(size=self.imgSize, resample=Image.LANCZOS)

        # Convert image to numpy array.
        img = np.array(img)

        # Convert 2-dim gray-scale array to 3-dim RGB array.
        if (len(img.shape) == 2):
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        #preprocesing
        img = img.astype(np.float32)
        img = img / 255
        mean_vec = np.array([[[0.485, 0.456, 0.406]]])
        std_vec  = np.array([[[0.229, 0.224, 0.225]]])
        img = (img - mean_vec) / std_vec

        outDict = {'img': img,
                   'img_path': self.records_list[item][1],
                   'fileName': fileName,
                   'captions': captions, #captions[:ind]}
                   'numbCaps': len(captions)}
        return outDict


#######################################################################################################################
def maybe_download_and_extract_coco(data_dir):
    """
    Download and extract the COCO data-set if the data-files don't
    already exist in data_dir.
    """
    from utils_data_preparation import downloadCoco

    # Base-URL for the data-sets on the internet.
    data_url = "http://images.cocodataset.org/"

    # Filenames to download from the internet.
    filenames = ["zips/train2017.zip", "zips/val2017.zip",
                 "annotations/annotations_trainval2017.zip"]

    # Download these files.
    for filename in filenames:
        # Create the full URL for the given file.
        url = data_url + filename
        print("Downloading " + url)
        downloadCoco.maybe_download_and_extract(url=url, download_dir=data_dir)

#######################################################################################################################
if __name__ == "__main__":


    data_dir = "../data/coco/"
    myData  =  DataLoaderWrapper(data_dir)

    minnumbCaps = 99

    from tqdm import tqdm
    for dataDict in tqdm(myData.myDataLoaderTrain, desc='', leave=True, mininterval=0.01):
        images    = dataDict['img']
        img_paths = dataDict['img_path']
        captions  = dataDict['captions']
        numbCaps = dataDict['numbCaps'].item()
        if minnumbCaps > numbCaps:
            minnumbCaps = numbCaps

    print(minnumbCaps)