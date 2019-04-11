import os
from utils_data_preparation.cocoDataset import maybe_download_and_extract_coco, DataLoaderWrapper
from utils_data_preparation.produceVGG16_fc7_features import produceVGG16_fc7_features

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda"    #"cuda" or "cpu"
data_dir = "data/coco/"

# Download coco dataset
maybe_download_and_extract_coco(data_dir)

# Generate dataloaders (train / val)
myDataLoader = DataLoaderWrapper(data_dir)

# Generate vocabulary
myDataLoader.generate_vocabulary()

# produce pickle files with fc7 features and captions (words and tokens)
produceVGG16_fc7_features(myDataLoader, device)
