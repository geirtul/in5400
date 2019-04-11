import torch
from tqdm import tqdm
from tqdm import trange, tqdm_notebook
from utils.plotter import Plotter
import sys


#######################################################################################################################
class Trainer():
    def __init__(self, model, modelParam, config, dataLoader, saveRestorer):
        self.model        = model
        self.modelParam   = modelParam
        self.config       = config
        self.dataLoader   = dataLoader
        self.saveRestorer = saveRestorer
        self.plotter      = Plotter(self.modelParam, self.config)
        return

    def train(self):
        for cur_epoch in range(self.model.start_epoch, self.modelParam['numbOfEpochs']):
            for modeSetup in self.modelParam['modeSetups']:
                mode     = modeSetup[0]
                is_train = modeSetup[1]
                if mode == 'train':
                    self.model.net.train()
                    loss = self.run_epoch(mode, self.model, is_train, cur_epoch)
                else:
                    with torch.no_grad():
                        self.model.net.eval()
                        loss = self.run_epoch(mode, self.model, is_train, cur_epoch)
                self.plotter.update(cur_epoch, loss, mode)
            self.saveRestorer.save(cur_epoch, loss, self.model)
        return


    def run_epoch(self, mode, model, is_train, cur_epoch):
        cur_it = -1
        epochTotalLoss = 0
        numbOfWordsInEpoch = 0

        if self.modelParam['inNotebook']:
            tt = tqdm_notebook(self.dataLoader.myDataDicts[mode], desc='', leave=True, mininterval=0.01, file=sys.stdout)
        else:
            # tt = tqdm_notebook(self.dataLoader.myDataDicts[mode], desc='', leave=True, mininterval=0.01,file=sys.stdout)
            tt = tqdm(self.dataLoader.myDataDicts[mode], desc='', leave=True, mininterval=0.01, file=sys.stdout)
        for dataDict in tt:
            for key in ['xTokens', 'yTokens', 'yWeights', 'vgg_fc7_features']:
                dataDict[key] = dataDict[key].to(model.device)
            cur_it += 1
            batchTotalLoss = 0
            numbOfWordsInBatch = 0
            for iter in range(dataDict['numbOfTruncatedSequences']):
                xTokens  = dataDict['xTokens'][:, :, iter]
                yTokens  = dataDict['yTokens'][:, :, iter]
                yWeights = dataDict['yWeights'][:, :, iter]
                vgg_fc7_features = dataDict['vgg_fc7_features']
                if iter==0:
                    logits, current_hidden_state = model.net(vgg_fc7_features, xTokens,  is_train)
                else:
                    logits, current_hidden_state_Ref = model.net(vgg_fc7_features, xTokens,  is_train, current_hidden_state)
                sumLoss, meanLoss = model.loss_fn(logits, yTokens, yWeights)

                if mode == 'train':
                    model.optimizer.zero_grad()
                    meanLoss.backward(retain_graph=True)
                    model.optimizer.step()

                batchTotalLoss += sumLoss.item()
                numbOfWordsInBatch += yWeights.sum().item()

            epochTotalLoss += batchTotalLoss
            numbOfWordsInEpoch +=numbOfWordsInBatch

            desc = f'{mode} | Epcohs={cur_epoch} | loss={batchTotalLoss/numbOfWordsInBatch:.4f}'
            tt.set_description(desc)
            tt.update()

            epochLoss = epochTotalLoss/numbOfWordsInEpoch
        return epochLoss
