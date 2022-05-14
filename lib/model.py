from torch import nn
import torch

class GRU4REC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, final_act='tanh',
                 dropout_hidden=.5, dropout_input=0, batch_size=50, embedding_dim=-1, use_cuda=False):
        super(GRU4REC, self).__init__()
        self.input_size = input_size #len(train_data.items)
        self.hidden_size = hidden_size#100 (no.of nodes in hidden layer)
        self.output_size = output_size #=input size 
        self.num_layers = num_layers #3
        self.dropout_hidden = dropout_hidden #0.5
        self.dropout_input = dropout_input #0
        self.embedding_dim = embedding_dim #-1 ==> one hot encodding 
        self.batch_size = batch_size #50 
        self.use_cuda = use_cuda 
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.onehot_buffer = self.init_emb()   #2d array of size (50 , len(train_data.item))
        self.h2o = nn.Linear(hidden_size, output_size) #hidden to output 
        self.create_final_activation(final_act) #final acivation function 
        if self.embedding_dim != -1:
            self.look_up = nn.Embedding(input_size, self.embedding_dim)
            self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        else:
            #that is our goal 
            self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        self = self.to(self.device)

    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))

    def forward(self, input, hidden):
        '''
        # input is a vector of size (50) that contain user current item 
        # output is the vec of items that contain target item 
        # return scores for the next item ==> now we have scores and target so compute loss then if needed update do it   
        Args:
        
            input (B,): a batch of item indices from a session-parallel mini-batch.
            target (B,): torch.LongTensor of next item indices from a session-parallel mini-batch.

        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            
            hidden: GRU hidden state
        '''

        if self.embedding_dim == -1:
            embedded = self.onehot_encode(input)
            ## not our case 
            if self.training and self.dropout_input > 0: embedded = self.embedding_dropout(embedded)
            embedded = embedded.unsqueeze(0)
        else:
            #NOT OUR CASE 
            #torch.unsqueeze(input, dim) → Tensor
            embedded = input.unsqueeze(0)
            embedded = self.look_up(embedded)

        #gru takes last hidden state(ht-1) + new input (x) ==> and return new hidden 
        # output that is the gru out and input for feed forward layers
        output, hidden = self.gru(embedded, hidden) #(num_layer, B, H)
        output = output.view(-1, output.size(-1))#==>flatten  #(B,H) 
        logit = self.final_activation(self.h2o(output)) #scores 

        return logit, hidden

    def init_emb(self):
        '''
        Initialize the one_hot embedding buffer, which will be used for producing the one-hot embeddings efficiently
        '''
        onehot_buffer = torch.FloatTensor(self.batch_size, self.output_size)#2d array of size (50 , len(train_data.item))
        onehot_buffer = onehot_buffer.to(self.device)
        return onehot_buffer

    def onehot_encode(self, input):
        """
        Returns a one-hot vector corresponding to the input
        Args:
            >>input is a vec of len 50(batch size) contain item indices 
            input (B,): torch.LongTensor of item indices
            
            buffer (B,output_size): buffer that stores the one-hot vector
        Returns:
            one_hot (B,C): torch.FloatTensor of one-hot vectors
        """
        self.onehot_buffer.zero_()
        index = input.view(-1, 1)#column instead of row here the dim will 
        one_hot = self.onehot_buffer.scatter_(1, index, 1) #it will add 1 on location of index 
        return one_hot #it will return matrix of dim (b(batch) , len(rain_data.item)) there is 1 on index of item 
        #i have row vec(input) that contain item index for each item in batch here we make row to be a column 
        # and then we have bufffer matrix of zeros 
        # all we will do is to add ones on idex in onput on buffer matrix 
         
    # i have not use it but it make drop out on embedding layer 
    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), 1).fill_(1 - self.dropout_input)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)
        mask = mask.to(self.device)
        input = input * mask
        return input

    #first state on gru will be zeros
    def init_hidden(self):
        '''
        Initialize the hidden state of the GRU
        '''
        try:
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        except:
            self.device = 'cpu'
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return h0