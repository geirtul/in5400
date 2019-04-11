import pickle
import torchvision.models as models
import torch
from torch import nn
import os
from tqdm import tqdm

#######################################################################################################################
def produceVGG16_fc7_features(myDataLoader, device):

    # set path to pre trained models weights
    os.environ['TORCH_MODEL_ZOO'] = myDataLoader.data_dir+'\\models\\VGG16'

    #restore pretrained VGG16 model
    model = models.vgg16(pretrained=True)

    #Remove last layer
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])

    #set model in evaluation mode
    model.eval()
    model.to(device)

    # restore vocabulary dict
    vocabularyDict = myDataLoader.loadVocabulary()

    saveDirTrain = myDataLoader.data_dir + '/Train2017_vgg16_fc7/'
    saveDirVal   = myDataLoader.data_dir + '/Val2017_vgg16_fc7/'

    if not os.path.exists(saveDirTrain):
        os.makedirs(saveDirTrain)
    if not os.path.exists(saveDirVal):
        os.makedirs(saveDirVal)

    if (os.listdir(saveDirTrain) == []) or (os.listdir(saveDirVal) == []):
        # Store all fc7 layers as pickel objects.
        for dataDict in tqdm(myDataLoader.myDataLoaderVal, desc='', leave=True, mininterval=0.01):
            images    = dataDict['img']
            img_paths = dataDict['img_path']
            fileName  = dataDict['fileName']
            captions  = dataDict['captions']
            images = images.permute(0,3,1,2).float().to(device)
            with torch.no_grad():
                fc7_features = model(images)
            saveFc7AsPickle(fc7_features.detach().cpu().numpy(), img_paths, fileName, captions, vocabularyDict, saveDirVal)

        for dataDict in tqdm(myDataLoader.myDataLoaderTrain, desc='', leave=True, mininterval=0.01):
            images    = dataDict['img']
            img_paths = dataDict['img_path']
            fileName  = dataDict['fileName']
            captions  = dataDict['captions']
            images = images.permute(0,3,1,2).float().to(device)
            with torch.no_grad():
                fc7_features = model(images)
            saveFc7AsPickle(fc7_features.detach().cpu().numpy(), img_paths, fileName, captions, vocabularyDict, saveDirTrain)

    else:
        print(f"Pickle files have already been produced. If the pickle files miss elements, please delete the files within '{saveDirTrain}' and '{saveDirVal}' and try again")
    return




###########################################################################################################
def saveFc7AsPickle(fc7_val, path_list, fileName, original_captions_list, vocabularyDict, saveDir):

    wordToToken = vocabularyDict['wordToToken']

    #change the order in "original_captions_list"
    batch_size     = fc7_val.shape[0]
    numbOfCaptions = len(original_captions_list)
    if numbOfCaptions<5:
        raise ValueError('An image has less than 5 caption')
    tmpList = []
    for ind in range(batch_size):
        tmp = []
        for c in range(numbOfCaptions):
            cap = original_captions_list[c][ind]
            if cap == []:
                a = 1
            if cap == '':
                a = 1
            tmp.append(cap)
        tmpList.append(tmp)
    original_captions_list = tmpList

    #Iterate through images
    for ii in range(len(path_list)):
        original_captions = original_captions_list[ii]
        captionsAsTokens = []
        captions         = []

        # Iterate through each caption
        for jj in range(len(original_captions)):
            original_caption = original_captions[jj]
            tokenList = []
            captionList = []
            tokenList.append(wordToToken['ssss'])
            captionList.append('ssss')

            # Convert to lowercase and spilt based on spaces
            original_caption = original_caption.lower()
            original_caption = original_caption.split(' ')
            for kk in range(len(original_caption)):
                word = original_caption[kk]

                # Remove all special characters and add to vocabulary
                word = ''.join(e for e in word if e.isalnum())
                if word != '':
                    tokenList.append(wordToToken[word])
                    captionList.append(word)

                    #Add "end token" at the end of the caption
                    if kk==len(original_caption)-1:
                        tokenList.append(wordToToken['eeee'])
                        captionList.append('eeee')

            captionsAsTokens.append(tokenList)
            captions.append(captionList)

        dataDict = {}
        dataDict['vgg16_fc7'] = fc7_val[ii, :]
        dataDict['imgPath']   = path_list[ii]
        dataDict['original_captions'] = original_captions_list[ii]
        dataDict['captionsAsTokens']  = captionsAsTokens
        dataDict['captions']          = captions

        str = saveDir+fileName[ii][:-4] + '.pickle'
        with open(str, 'wb') as handle:
            pickle.dump(dataDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return
#######################################################################################################################






