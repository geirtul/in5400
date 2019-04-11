import torch
from sourceFiles.cocoSource import RNNCell
from sourceFiles.cocoSource import GRUCell
from sourceFiles.cocoSource import loss_fn
from sourceFiles.cocoSource import RNN
from sourceFiles.cocoSource import imageCaptionModel
from torch import nn
import numpy as np


def checkMatrix(mat, name, dims):
    flag = 0
    if torch.tensor(mat.shape).shape[0]==2:
        if mat.shape[0]==dims[0] and mat.shape[1] == dims[1]:
            flag = 1
    if flag == 1:
        print(f'The "{name}" matrix has correct shape')
    else:
        print(f'The "{name}" matrix has NOT the correct shape')

    return

def checkInitValues(mat, type, name, limit=5e-4):
    if type=="variance_scaling":
        diff = np.abs(np.sqrt(1/mat.shape[0]) - mat.std().item())
        # print(diff)
        if diff<limit:
            print(f'The "{name}" matrix has correct values')
        else:
            print(f'The "{name}" matrix has NOT the correct values')
    elif type=="bias":
        diff = mat.detach().numpy().sum()
        if diff==0:
            print(f'The "{name}" matrix has correct values')
        else:
            print(f'The "{name}" matrix has NOT the correct values')

    return

#######################################################################################################################
def RNNcell_test():

    hidden_size = 2048
    embed_size = 1024
    # hidden_size = 2
    # embed_size = 2
    batch_size = 2
    x = torch.randn(batch_size, embed_size)
    h0 = torch.randn(batch_size, hidden_size)
    myRnnCell      = RNNCell(hidden_size, embed_size)
    pytorchRnnCell = nn.RNNCell(embed_size, hidden_size)

    #check dimentions
    checkMatrix(myRnnCell.weight, "weight", dims=(hidden_size+embed_size, hidden_size))
    checkMatrix(myRnnCell.bias, "bias", dims=(1, hidden_size))

    #check vales
    checkInitValues(myRnnCell.weight, type="variance_scaling", name="weight")
    checkInitValues(myRnnCell.bias, type="bias", name="bias")

    #check output
    hx = pytorchRnnCell.weight_ih.t()
    hh = pytorchRnnCell.weight_hh.t()
    myRnnCell.weight = nn.Parameter(torch.cat((hx, hh), dim=0))
    pytorchRnnCell.bias_ih = pytorchRnnCell.bias_hh
    myRnnCell.bias         = nn.Parameter(2*pytorchRnnCell.bias_hh)

    myOut    = (myRnnCell.forward(x=x, state_old=h0)).detach()
    torchOut = (pytorchRnnCell.forward(input=x, hx=h0)).detach()

    flag = torch.all(torch.lt(torch.abs(torch.add(myOut, - torchOut)), 1e-6))
    if flag:
        print('The "forward" function is implemented correctly')
    else:
        print('The "forward" function is implemented incorrectly')

    return

#######################################################################################################################
def GRUcell_test():
    hidden_size = 2048
    embed_size = 1024
    myGRUCell      = GRUCell(hidden_size, embed_size)

    #check dimentions
    checkMatrix(myGRUCell.weight_r, "weight_r", dims=(hidden_size+embed_size, hidden_size))
    checkMatrix(myGRUCell.weight_r, "weight_u", dims=(hidden_size + embed_size, hidden_size))
    checkMatrix(myGRUCell.weight_r, "weight", dims=(hidden_size + embed_size, hidden_size))
    checkMatrix(myGRUCell.bias_r, "bias_r", dims=(1, hidden_size))
    checkMatrix(myGRUCell.bias_r, "bias_u", dims=(1, hidden_size))
    checkMatrix(myGRUCell.bias_r, "bias", dims=(1, hidden_size))

    #check vales
    checkInitValues(myGRUCell.weight_r, type="variance_scaling", name="weight_r")
    checkInitValues(myGRUCell.weight_u, type="variance_scaling", name="weight_u")
    checkInitValues(myGRUCell.weight, type="variance_scaling", name="weight")
    checkInitValues(myGRUCell.bias_r, type="bias", name="bias_r")
    checkInitValues(myGRUCell.bias_u, type="bias", name="bias_u")
    checkInitValues(myGRUCell.bias, type="bias", name="bias")

    #check output
    hidden_size = 24
    embed_size = 12
    batch_size = 3
    myGRUCell      = GRUCell(hidden_size, embed_size)

    weight_dir = 'unit_tests/GRU_weights.pt'
    # weight_dir = 'GRU_weights.pt'
    checkpoint = torch.load(weight_dir)
    myGRUCell.load_state_dict(checkpoint['model_state_dict'])
    referenceOutput = checkpoint['referenceOutput']
    x               = checkpoint['x']
    h0              = checkpoint['h0']
    myOutput = myGRUCell.forward(x, h0)

    flag = torch.all(torch.lt(torch.abs(torch.add(myOutput, - referenceOutput)), 1e-6))
    if flag:
        print('The "forward" function is implemented correctly')
    else:
        print('The "forward" function is implemented incorrectly')

    # store weights and the correct ouput
    # torch.save({'model_state_dict': myGRUCell.state_dict(), 'referenceOutput': myOutput, 'x': x, 'h0': h0}, weight_dir)

    return

#######################################################################################################################
def loss_fn_test():

    weight_dir = 'unit_tests/loss_fn_tensors.pt'
    # weight_dir = 'loss_fn_tensors.pt'
    checkpoint = torch.load(weight_dir)
        
    logits   = checkpoint['logits']
    yTokens  = checkpoint['yTokens']
    yWeights = checkpoint['yWeights']
    sumLoss  = checkpoint['sumLoss']
    meanLoss = checkpoint['meanLoss']

    sumLossCalc, meanLossCalc = loss_fn(logits, yTokens, yWeights)
    diffLimit=1e-8

    diff = torch.abs(sumLossCalc-sumLoss)/sumLoss
    if diff <diffLimit:
        print('sumLoss has correct value')
    else:
        print(f'sumLoss has NOT correct value. Diff ={diff.item():.7f}')

    diff = torch.abs(meanLossCalc-meanLoss)/meanLoss
    if diff <diffLimit:
        print('meanLoss has correct value')
    else:
        print(f'meanLoss has NOT correct value. Diff ={diff.item():.7f}')

    # store weights and the correct ouput
    # weight_dir = 'unit_tests/loss_fn_tensors.pt'
    # lossDict = {'logits': logits,
    #             'yTokens': yTokens,
    #             'yWeights': yWeights,
    #             'sumLoss': sumLoss,
    #             'meanLoss': meanLoss}
    # torch.save(lossDict, weight_dir)
    return

#######################################################################################################################
def RNN_test(is_train):

    print(f'-------- For is_train = {is_train} --------')
    my_dir = f'unit_tests/RNN_tensors_is_train_{is_train}.pt'
    # my_dir = f'RNN_tensors_is_train_{is_train}.pt'
    checkpoint = torch.load(my_dir)

    outputLayer = checkpoint['outputLayer']
    Embedding   = checkpoint['Embedding']
    xTokens     = checkpoint['xTokens']
    initial_hidden_state = checkpoint['initial_hidden_state']
    input_size           = checkpoint['input_size']
    hidden_state_size    = checkpoint['hidden_state_size']
    num_rnn_layers = checkpoint['num_rnn_layers']
    cell_type     = checkpoint['cell_type']
    is_train      = checkpoint['is_train']
    logitsRef        = checkpoint['logits']
    current_stateRef = checkpoint['current_state']
    myRNNref           = checkpoint['myRNN']

    # You should implement this function
    myRNN = RNN(input_size, hidden_state_size, num_rnn_layers, cell_type)
    myRNN.load_state_dict(myRNNref.state_dict())
    logits, current_state = myRNN(xTokens, initial_hidden_state, outputLayer, Embedding, is_train)

    batch_size                = xTokens.shape[0]
    truncated_backprop_length = xTokens.shape[1]
    vocabulary_size           = logitsRef.shape[2]

    flag = 1
    if torch.tensor(logits.shape).shape[0]==3:
        if logits.shape[0] != logitsRef.shape[0]:
            flag = 0
        if logits.shape[1] != logitsRef.shape[1]:
            flag = 0
        if logits.shape[2] != logitsRef.shape[2]:
            flag = 0
    else:
        flag = 0
    if flag == 1:
        print('"logits" has correct shape')
    else:
        print('"logits" has NOT correct shape')

    if flag==1:
        flag = torch.all(torch.lt(torch.abs(torch.add(logitsRef, - logits)), 1e-6))
        if flag:
            print('"logits" has correct values')
        else:
            print('"logits" has incorrect values')

    flag = 1
    if torch.tensor(current_state.shape).shape[0]==3:
        if current_state.shape[0] != current_stateRef.shape[0]:
            flag = 0
        if current_state.shape[1] != current_stateRef.shape[1]:
            flag = 0
        if current_state.shape[2] != current_stateRef.shape[2]:
            flag = 0
    else:
        flag = 0
    if flag == 1:
        print('"current_hidden_state" has correct shape')
    else:
        print('"current_hidden_state" has NOT correct shape')

    if flag==1:
        flag = torch.all(torch.lt(torch.abs(torch.add(current_stateRef, - current_state)), 1e-6))
        if flag:
            print('"current_hidden_state" has correct values')
        else:
            print('"current_hidden_state" has incorrect values')


    # store weights and the correct ouput
    # my_dir = f'unit_tests/RNN_tensors_is_train_{is_train}.pt'
    # myDict = {'outputLayer': outputLayer,
    #             'Embedding': Embedding,
    #             'xTokens': xTokens,
    #             'initial_hidden_state': initial_hidden_state,
    #           'input_size': self.input_size,
    #           'hidden_state_size': self.hidden_state_size,
    #           'num_rnn_layers': self.num_rnn_layers,
    #           'cell_type': self.cell_type,
    #           'is_train': is_train,
    #           'logits': logits,
    #           'current_state': current_state}
    # torch.save(myDict, my_dir)

    # my_dir = f'RNN_tensors_is_train_{is_train}.pt'
    # myDict = {'outputLayer': outputLayer,
    #             'Embedding': Embedding,
    #             'xTokens': xTokens,
    #             'initial_hidden_state': initial_hidden_state,
    #           'input_size': input_size,
    #           'hidden_state_size': hidden_state_size,
    #           'num_rnn_layers': num_rnn_layers,
    #           'cell_type': cell_type,
    #           'is_train': is_train,
    #           'logits': logits,
    #           'current_state': current_state,
    #           'myRNN': myRNN}
    # torch.save(myDict, my_dir)

    return

#######################################################################################################################
def imageCaptionModel_test():
    my_dir = 'unit_tests/imageCaptionModel_tensors.pt'
    # my_dir = f'imageCaptionModel_tensors.pt'
    checkpoint = torch.load(my_dir)
    config = checkpoint['config']

    vgg_fc7_features                  = checkpoint['vgg_fc7_features']
    xTokens                           = checkpoint['xTokens']
    is_train                          = checkpoint['is_train']
    myImageCaptionModelRef_state_dict = checkpoint['myImageCaptionModelRef_state_dict']
    logitsRef                         = checkpoint['logitsRef']
    current_hidden_state_Ref          = checkpoint['current_hidden_state_Ref']

    myImageCaptionModel = imageCaptionModel(config)
    myImageCaptionModel.load_state_dict(myImageCaptionModelRef_state_dict)
    logits, current_hidden_state = myImageCaptionModel(vgg_fc7_features, xTokens,  is_train)

    print(' ---- With "current_hidden_state"==None -----')

    flag = 1
    if torch.tensor(logits.shape).shape[0]==3:
        if logits.shape[0] != logitsRef.shape[0]:
            flag = 0
        if logits.shape[1] != logitsRef.shape[1]:
            flag = 0
        if logits.shape[2] != logitsRef.shape[2]:
            flag = 0
    else:
        flag = 0
    if flag == 1:
        print('"logits" has correct shape')
    else:
        print('"logits" has NOT correct shape')

    if flag==1:
        flag = torch.all(torch.lt(torch.abs(torch.add(logitsRef, - logits)), 1e-6))
        if flag:
            print('"logits" has correct values')
        else:
            print('"logits" has incorrect values')

    flag = 1
    if torch.tensor(current_hidden_state.shape).shape[0]==3:
        if current_hidden_state.shape[0] != current_hidden_state_Ref.shape[0]:
            flag = 0
        if current_hidden_state.shape[1] != current_hidden_state_Ref.shape[1]:
            flag = 0
        if current_hidden_state.shape[2] != current_hidden_state_Ref.shape[2]:
            flag = 0
    else:
        flag = 0
    if flag == 1:
        print('"current_hidden_state" has correct shape')
    else:
        print('"current_hidden_state" has NOT correct shape')

    if flag==1:
        flag = torch.all(torch.lt(torch.abs(torch.add(current_hidden_state_Ref, - current_hidden_state)), 1e-6))
        if flag:
            print('"current_hidden_state" has correct values')
        else:
            print('"current_hidden_state" has incorrect values')

    ###################################################################################################################
    print(' \n---- With "current_hidden_state"!=None -----')
    my_dir = 'unit_tests/imageCaptionModel_tensors_with_current_hidden_state.pt'
    # my_dir = f'imageCaptionModel_tensors_with_current_hidden_state.pt'
    checkpoint = torch.load(my_dir)
    config = checkpoint['config']

    vgg_fc7_features                  = checkpoint['vgg_fc7_features']
    xTokens                           = checkpoint['xTokens']
    is_train                          = checkpoint['is_train']
    myImageCaptionModelRef_state_dict = checkpoint['myImageCaptionModelRef_state_dict']
    logitsRef                         = checkpoint['logitsRef']
    current_hidden_state_Ref          = checkpoint['current_hidden_state_Ref']
    current_hidden_state              = checkpoint['current_hidden_state']

    myImageCaptionModel = imageCaptionModel(config)
    myImageCaptionModel.load_state_dict(myImageCaptionModelRef_state_dict)
    logits, current_hidden_state = myImageCaptionModel(vgg_fc7_features, xTokens,  is_train, current_hidden_state)

    flag = 1
    if torch.tensor(logits.shape).shape[0]==3:
        if logits.shape[0] != logitsRef.shape[0]:
            flag = 0
        if logits.shape[1] != logitsRef.shape[1]:
            flag = 0
        if logits.shape[2] != logitsRef.shape[2]:
            flag = 0
    else:
        flag = 0
    if flag == 1:
        print('"logits" has correct shape')
    else:
        print('"logits" has NOT correct shape')

    if flag==1:
        flag = torch.all(torch.lt(torch.abs(torch.add(logitsRef, - logits)), 1e-6))
        if flag:
            print('"logits" has correct values')
        else:
            print('"logits" has incorrect values')

    flag = 1
    if torch.tensor(current_hidden_state.shape).shape[0]==3:
        if current_hidden_state.shape[0] != current_hidden_state_Ref.shape[0]:
            flag = 0
        if current_hidden_state.shape[1] != current_hidden_state_Ref.shape[1]:
            flag = 0
        if current_hidden_state.shape[2] != current_hidden_state_Ref.shape[2]:
            flag = 0
    else:
        flag = 0
    if flag == 1:
        print('"current_hidden_state" has correct shape')
    else:
        print('"current_hidden_state" has NOT correct shape')

    if flag==1:
        flag = torch.all(torch.lt(torch.abs(torch.add(current_hidden_state_Ref, - current_hidden_state)), 1e-6))
        if flag:
            print('"current_hidden_state" has correct values')
        else:
            print('"current_hidden_state" has incorrect values')



    # store weights and the correct ouput
    # my_dir = f'unit_tests/imageCaptionModel_tensors_with_current_hidden_state.pt'
    # myDict = {'config' : self.config,
    #           'vgg_fc7_features': vgg_fc7_features,
    #           'xTokens': xTokens,
    #           'initial_hidden_state': current_hidden_state,
    #           'is_train': is_train,
    #           'myImageCaptionModelRef_state_dict': model.net.state_dict(),
    #           'logitsRef': logits,
    #           'current_hidden_state': current_hidden_state,
    #           'current_hidden_state_Ref': current_hidden_state_Ref}
    # torch.save(myDict, my_dir)

    return

#######################################################################################################################
if __name__ == '__main__':
    GRUcell_test()