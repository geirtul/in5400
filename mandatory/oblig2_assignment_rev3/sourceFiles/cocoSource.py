from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


######################################################################################################################
class imageCaptionModel(nn.Module):
    def __init__(self, config):
        super(imageCaptionModel, self).__init__()
        """
        "imageCaptionModel" is the main module class for the image captioning network
        
        Args:
            config: Dictionary holding neural network configuration

        Returns:
            self.Embedding  : An instance of nn.Embedding, shape[vocabulary_size, embedding_size]
            self.inputLayer : An instance of nn.Linear, shape[VggFc7Size, hidden_state_sizes]
            self.rnn        : An instance of RNN
            self.outputLayer: An instance of nn.Linear, shape[hidden_state_sizes, vocabulary_size]
        """
        self.config = config
        self.vocabulary_size    = config['vocabulary_size']
        self.embedding_size     = config['embedding_size']
        self.VggFc7Size         = config['VggFc7Size']
        self.hidden_state_size = config['hidden_state_sizes']
        self.num_rnn_layers     = config['num_rnn_layers']
        self.cell_type          = config['cellType']

        # ToDo
        self.Embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)

        self.inputLayer = nn.Linear(self.VggFc7Size, self.hidden_state_size)

        self.rnn = RNN(self.embedding_size, self.hidden_state_size, self.num_rnn_layers, self.cell_type)

        self.outputLayer = nn.Linear(self.hidden_state_size, self.vocabulary_size)
        return

    def forward(self, vgg_fc7_features, xTokens, is_train, current_hidden_state=None):
        """
        Args:
            vgg_fc7_features    : Features from the VGG16 network, shape[batch_size, VggFc7Size]
            xTokens             : Shape[batch_size, truncated_backprop_length]
            is_train            : "is_train" is a flag used to select whether or not to use estimated token as input
            current_hidden_state: If not None, "current_hidden_state" should be passed into the rnn module
                                  shape[num_rnn_layers, batch_size, hidden_state_sizes]

        Returns:
            logits              : Shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_hidden_state: shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        # ToDO
        # Get "initial_hidden_state" shape[num_rnn_layers, batch_size, hidden_state_sizes].
        # Remember that each rnn cell needs its own initial state.
        # use self.rnn to calculate "logits" and "current_hidden_state"
        
        # Get batch size from vgg_fc7_features
        batch_size = vgg_fc7_features.shape[0]
        
        # Feed vgg_fc7_features through inputLayer, to be passed to all rnn cells.
        input_prepped = self.inputLayer(vgg_fc7_features)
        
        # Define initial_hidden_state of current is None
        if current_hidden_state is None:
            initial_hidden_state = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_state_size)
            # Prepare initial state for each RNN cell
            for i in range(self.num_rnn_layers):
                initial_hidden_state[i] = input_prepped.clone().detach()
        else:
            initial_hidden_state = current_hidden_state
        
        logits, current_hidden_state_out = self.rnn.forward(xTokens, 
                                                            initial_hidden_state,
                                                            self.outputLayer,
                                                            self.Embedding,
                                                            is_train)

        return logits, current_hidden_state_out

######################################################################################################################
class RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, cell_type='RNN'):
        super(RNN, self).__init__()
        """
        Args:
            input_size (Int)        : embedding_size
            hidden_state_size (Int) : Number of features in the rnn cells (will be equal for all rnn layers) 
            num_rnn_layers (Int)    : Number of stacked rnns
            cell_type               : Whether to use vanilla or GRU cells
            
        Returns:
            self.cells              : A nn.ModuleList with entities of "RNNCell" or "GRUCell"
        """
        self.input_size        = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers    = num_rnn_layers
        self.cell_type         = cell_type

        # ToDo
        # Your task is to create a list (self.cells) of type "nn.ModuleList" and populate 
        # it with cells of type "self.cell_type".
        
        cells = []
        if self.cell_type == 'GRU':
            for i in range(num_rnn_layers):
                if i == 0:
                    cells.append(GRUCell(self.hidden_state_size, self.input_size))
                else:
                    cells.append(GRUCell(self.hidden_state_size, self.hidden_state_size))
        else:
            for i in range(num_rnn_layers):
                if i == 0:
                    cells.append(RNNCell(self.hidden_state_size, self.input_size))
                else:
                    cells.append(RNNCell(self.hidden_state_size, self.hidden_state_size))
            
        self.cells = nn.ModuleList(cells)


    def forward(self, xTokens, initial_hidden_state, outputLayer, Embedding, is_train=True):
        """
        Args:
            xTokens:        shape [batch_size, truncated_backprop_length]
            initial_hidden_state:  shape [num_rnn_layers, batch_size, hidden_state_size]
            outputLayer:    handle to the last fully connected layer (an instance of nn.Linear)
            Embedding:      An instance of nn.Embedding. This is the embedding matrix.
            is_train:       flag: whether or not to feed in the predicated token vector as input for next step

        Returns:
            logits        : The predicted logits. shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_state : The hidden state from the last iteration (in time/words).
                            Shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        if is_train==True:
            seqLen = xTokens.shape[1] #truncated_backprop_length
        else:
            seqLen = 40 #Max sequence length to be generated

        # ToDo
        # While iterate through the (stacked) rnn, it may be easier to use lists instead of indexing the tensors.
        # You can use "list(torch.unbind())" and "torch.stack()" to convert from pytorch tensor to lists and back again.
        # get input embedding vectors
        # Use for loops to run over "seqLen" and "self.num_rnn_layers" to calculate logits
        # Produce outputs
        
        batch_size = initial_hidden_state.shape[1]

        input_tokens = Embedding(xTokens)
        
        logits = torch.zeros(batch_size, seqLen, outputLayer.out_features)
                
        # keep track of all states because Autograd doesn't play nice with in-place modification
        all_states = torch.zeros(seqLen+1, self.num_rnn_layers, batch_size, self.hidden_state_size)
        all_states[0, :, :, :] = initial_hidden_state
        
        if is_train:
            for i in range(seqLen):
                cell_input = input_tokens[:, i, :]
                for j in range(self.num_rnn_layers):
                    current_cell = self.cells[j]
                    current_state = all_states[i, j, :, :].clone().detach()
                    all_states[i+1, j, :, :] = current_cell.forward(cell_input, current_state)

                    # Update cell input with the new state
                    cell_input = all_states[i+1, j, :, :].clone().detach()

                # Calculate logit with final state for the current word
                # Force keeping gradient, otherwise backprop produces bad results.
                logits[:, i, :] = outputLayer(all_states[i+1, -1, :, :]).clone().detach().requires_grad_(True)
        else:
            cell_input = input_tokens[:, 0, :]
            for i in range(seqLen):
                for j in range(self.num_rnn_layers):
                    current_cell = self.cells[j]
                    current_state = all_states[i, j, :, :].clone().detach()
                    all_states[i+1, j, :, :] = current_cell.forward(cell_input, current_state)

                    # Update cell input so that the next cell receives the new state.
                    cell_input = all_states[i+1, j, :, :].clone().detach()

                # Calculate logit with final state for the current word
                logits[:, i, :] = outputLayer(all_states[i+1, -1, :, :])
                
                # set cell_input to embedded version of the best candidate word
                current_output = F.softmax(logits[:, i, :], dim=1)
                best_word = torch.argmax(current_output, dim=1)
                cell_input = Embedding(best_word)


        return logits, all_states[-1, :, :, :]

########################################################################################################################
class GRUCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(GRUCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight_u: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean. 

            self.weight_r: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean. 

            self.weight: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean. 

            self.bias_u: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

            self.bias_r: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        self.hidden_state_size = hidden_state_size

        # TODO:
        w_u = torch.empty(self.hidden_state_size+input_size, self.hidden_state_size)
        self.weight_u = nn.Parameter(nn.init.normal_(w_u, 0.0, 1/np.sqrt(input_size+self.hidden_state_size)))
        b_u = torch.empty(1, self.hidden_state_size)
        self.bias_u = nn.Parameter(nn.init.constant_(b_u, 0.0))

        w_r = torch.empty(self.hidden_state_size+input_size, self.hidden_state_size)
        self.weight_r = nn.Parameter(nn.init.normal_(w_r, 0.0, 1/np.sqrt(input_size+self.hidden_state_size)))
        b_r = torch.empty(1, self.hidden_state_size)
        self.bias_r = nn.Parameter(nn.init.constant_(b_u, 0.0))

        w = torch.empty(self.hidden_state_size+input_size, self.hidden_state_size)
        self.weight = nn.Parameter(nn.init.normal_(w, 0.0, 1/np.sqrt(input_size+self.hidden_state_size)))
        b = torch.empty(1, self.hidden_state_size)
        self.bias = nn.Parameter(nn.init.constant_(b, 0.0))
        
        return

    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        # TODO:
        tmp_conc = torch.cat((x, state_old), 1)
        
        update_gate = torch.sigmoid(torch.matmul(tmp_conc, self.weight_u) + self.bias_u)
        update_reset = torch.sigmoid(torch.matmul(tmp_conc, self.weight_r) + self.bias_r)
        
        tmp_conc_candidate = torch.cat((x, update_reset*state_old), 1)
        candidate_cell = torch.tanh(torch.matmul(tmp_conc_candidate, self.weight) + self.bias)
        
        state_new = update_gate*state_old + (1 - update_gate)*candidate_cell
        return state_new

######################################################################################################################
class RNNCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(RNNCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight: A nn.Parameter with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean.

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        self.hidden_state_size = hidden_state_size

        # TODO:
        w = torch.empty(self.hidden_state_size+input_size, self.hidden_state_size)
        self.weight = nn.Parameter(nn.init.normal_(w, 0.0, 1/np.sqrt(input_size+self.hidden_state_size)))
        
        b = torch.empty(1, self.hidden_state_size)
        self.bias = nn.Parameter(nn.init.constant_(b, 0.0))


    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        # TODO:
        tmp = torch.cat((x, state_old), 1)
        state_new = torch.tanh(torch.matmul(tmp, self.weight) + self.bias)
        return state_new

######################################################################################################################
def loss_fn(logits, yTokens, yWeights):
    """
    Weighted softmax cross entropy loss.

    Args:
        logits          : shape[batch_size, truncated_backprop_length, vocabulary_size]
        yTokens (labels): Shape[batch_size, truncated_backprop_length]
        yWeights        : Shape[batch_size, truncated_backprop_length]. Add contribution to the total loss only from words existing 
                          (the sequence lengths may not add up to #*truncated_backprop_length)

    Returns:
        sumLoss: The total cross entropy loss for all words
        meanLoss: The averaged cross entropy loss for all words

    Tips:
        F.cross_entropy
    """
    eps = 0.0000000001 #used to not divide by zero
    
    # TODO:
    softmax = F.log_softmax(logits, 2)
    loss = nn.NLLLoss(reduction='none')
    output = loss(torch.transpose(softmax, 1, 2), yTokens)
    output_weighted = output*yWeights
    
    sumLoss  = torch.sum(output_weighted)
    meanLoss = sumLoss/torch.sum(yWeights)

    return sumLoss, meanLoss


