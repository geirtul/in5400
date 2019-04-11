from utils.dataLoader import DataLoaderWrapper
from utils.saverRestorer import SaverRestorer
from utils.model import Model
from utils.trainer import Trainer
from utils.validate import plotImagesAndCaptions

def main(config, modelParam):
    # create an instance of the model you want
    model = Model(config, modelParam)

    # create an instacne of the saver and resoterer class
    saveRestorer = SaverRestorer(config, modelParam)
    model        = saveRestorer.restore(model)

    # create your data generator
    dataLoader = DataLoaderWrapper(config, modelParam)

    # here you train your model
    if modelParam['inference'] == False:
        # create trainer and pass all the previous components to it
        trainer = Trainer(model, modelParam, config, dataLoader, saveRestorer)
        trainer.train()

    #plotImagesAndCaptions
    if modelParam['inference'] == True:
        plotImagesAndCaptions(model, modelParam, config, dataLoader)

    return


########################################################################################################################
if __name__ == '__main__':
    data_dir = 'data/coco/'

    #train
    modelParam = {
        'batch_size': 32,  # Training batch size
        'cuda': {'use_cuda': False,  # Use_cuda=True: use GPU
                 'device_idx': 0},  # Select gpu index: 0,1,2,3
        'numbOfCPUThreadsUsed': 10,  # Number of cpu threads use in the dataloader
        'numbOfEpochs': 20,  # Number of epochs
        'data_dir': data_dir,  # data directory
        'img_dir': 'loss_images/',
        'modelsDir': 'storedModels/',
        'modelName': 'model_0/',  # name of your trained model
        'restoreModelLast': 0,
        'restoreModelBest': 0,
        'modeSetups': [['train', True], ['val', True]],
        'inNotebook': False,  # If running script in jupyter notebook
        'inference': False
    }

    config = {
        'optimizer': 'adam',  # 'SGD' | 'adam' | 'RMSprop'
        'learningRate': {'lr': 0.0005},  # learning rate to the optimizer
        'weight_decay': 0,  # weight_decay value
        'VggFc7Size': 4096,  # Fixed, do not change
        'embedding_size': 128,  # word embedding size
        'vocabulary_size': 100,  # number of different words
        'truncated_backprop_length': 7,
        'hidden_state_sizes': 64,  #
        'num_rnn_layers': 1,  # number of stacked rnn's
        'cellType': 'RNN'  # RNN or GRU
    }

    if modelParam['inference'] == True:
        modelParam['batch_size'] = 1
        modelParam['modeSetups'] = [['val', False]]
        modelParam['restoreModelBest'] = 1

    main(config, modelParam)

    aa = 1