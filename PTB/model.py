import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import weight_norm


class RNN(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size,
                 dropout_rate, bos_idx, eos_idx, pad_idx):
        super(RNN, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        # RNN
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, length):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        input_seq = input_seq[sorted_idx]
        embedded_input = self.embedding(input_seq)
        drop_input = F.dropout(embedded_input, p=self.dropout_rate,
                               training=self.training)

        # RNN forward
        pack_input = pack_padded_sequence(drop_input, sorted_len + 1,
                                          batch_first=True)
        pack_output, _ = self.rnn(pack_input)
        output, _ = pad_packed_sequence(pack_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        # project output
        drop_output = F.dropout(output, p=self.dropout_rate,
                                training=self.training)
        batch_size, seq_len, hidden_size = drop_output.size()
        logit = self.output(drop_output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp
    
class two_layer_RNN(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size,
                 dropout_rate, bos_idx, eos_idx, pad_idx):
        super(two_layer_RNN, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        # RNN1
        self.rnn1 = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # RNN2
        self.rnn2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, length):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        input_seq = input_seq[sorted_idx]
        embedded_input = self.embedding(input_seq)
        drop_input = F.dropout(embedded_input, p=self.dropout_rate,
                               training=self.training)

        # RNN forward
        pack_input1 = pack_padded_sequence(drop_input, sorted_len + 1,batch_first=True)
        pack_output1, _ = self.rnn1(pack_input1)
        pack_output2, _ = self.rnn2(pack_output1)
        output, _ = pad_packed_sequence(pack_output2, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]
        
        # project output
        drop_output = F.dropout(output, p=self.dropout_rate, training=self.training)
        batch_size, seq_len, hidden_size = drop_output.size()
        logit = self.output(drop_output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp




criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=0)
# negative log likelihood
def NLL(logp, target, length):
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp[:, :torch.max(length).item(),:].contiguous().view(-1, logp.size(-1)) # logp = logp.view(-1, logp.size(-1))
    return criterion(logp, target)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.num_levels = len(num_channels)
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        count = 0
        final_d = x
        for layer in self.network:
            final_d = layer(final_d)
            if count is 0:
                d = torch.unsqueeze(final_d,0) #size = (1,batch_size,hidden_states,length)
            elif count is (self.num_levels-1):
                break
                #last_d = torch.unsqueeze(temp_d,0) #size = (1,batch_size,embedding_size,length)
            else:
                d = torch.cat((d,torch.unsqueeze(final_d,0)), 0) #size = (count+1,batch_size,hidden_states,length)
            count = count+1
        return final_d, d
        #for layer in self.network:
        #    x=layer(x)
        #return x
        #return self.network(x)
    
    

class TCN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_channels, bos_idx, eos_idx, pad_idx, kernel_size=2, dropout=0.2, emb_dropout=0.2):
        super(TCN, self).__init__()
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.decoder = nn.Linear(embed_size, vocab_size)
        self.decoder.weight = self.encoder.weight
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()
        

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, length, target):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        emb = self.drop(self.encoder(x))
        y, __ = self.tcn(emb.transpose(1, 2))
        o = self.decoder(y.transpose(1, 2))
        o = self.drop(o)
        
        logp = o.contiguous()
        NLL_loss = NLL(logp, target, length + 1)
        return logp, NLL_loss
    
class CRN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_channels , time_step, hidden_size,
                 dropout_rate, bos_idx, eos_idx, pad_idx, kernel_size=2, emb_dropout=0.2):
        super(CRN, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.emb_dropout = emb_dropout
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,padding_idx=pad_idx)
        
        # TCN
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size=kernel_size, dropout=dropout_rate)
        
        # RNN
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, length, target):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        input_seq = input_seq[sorted_idx]
        embedded_input = self.embedding(input_seq)
        drop_input = F.dropout(embedded_input, p=self.emb_dropout, training=self.training)
        # TCN forward
        z,__ = self.tcn(drop_input.transpose(1, 2))
        z = F.dropout(z, p=self.emb_dropout, training=self.training)
        # RNN forward
        pack_input = pack_padded_sequence(z.transpose(1, 2), sorted_len + 1,batch_first=True)
        pack_output, _ = self.rnn(pack_input)
        output, _ = pad_packed_sequence(pack_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        # project output
        drop_output = F.dropout(output, p=self.emb_dropout,
                                training=self.training)
        batch_size, seq_len, hidden_size = drop_output.size()
        logit = self.output(drop_output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)
        
        NLL_loss = NLL(logp, target, length + 1)

        return logp, NLL_loss
    
class RCN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_channels , time_step, hidden_size,
                 dropout_rate, bos_idx, eos_idx, pad_idx, kernel_size=2, emb_dropout=0.2):
        super(RCN, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.emb_dropout = emb_dropout
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,padding_idx=pad_idx)
        
        # TCN
        self.tcn = TemporalConvNet(hidden_size, num_channels, kernel_size=kernel_size, dropout=dropout_rate)
        
        # RNN
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # output
        self.output = nn.Linear(embed_size, vocab_size)
        
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, input_seq, length, target):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        input_seq = input_seq[sorted_idx]
        embedded_input = self.embedding(input_seq)
        drop_input = F.dropout(embedded_input, p=self.emb_dropout,training=self.training)

        # RNN forward
        pack_input = pack_padded_sequence(drop_input, sorted_len + 1,
                                          batch_first=True)
        pack_output, _ = self.rnn(pack_input)
        z, _ = pad_packed_sequence(pack_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        z = z[reversed_idx]
        
        # project output
        z = F.dropout(z, p=self.emb_dropout, training=self.training)
        
        
        # TCN forward
        y, __ = self.tcn(z.transpose(1, 2))
        o = self.output(y.transpose(1, 2))
        o = self.drop(o)
        
        logp = o.contiguous()
        NLL_loss = NLL(logp, target, length + 1)

        return logp, NLL_loss


    
class multi_resolution_CRN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_channels , time_step, hidden_size,
                 dropout_rate, bos_idx, eos_idx, pad_idx, kernel_size=2, emb_dropout=0.2):
        super(multi_resolution_CRN, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.emb_dropout = emb_dropout
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.num_levels = len(num_channels)

        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,padding_idx=pad_idx)
        
        # TCN
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size=kernel_size, dropout=dropout_rate)
        
        # RNN
        RNNs = []
        for i in range(self.num_levels):
            RNNs += [nn.LSTM(num_channels[i], hidden_size, batch_first=True)]
            
        self.network = nn.Sequential(*RNNs)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, length, target):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        input_seq = input_seq[sorted_idx]
        embedded_input = self.embedding(input_seq)
        drop_input = F.dropout(embedded_input, p=self.emb_dropout, training=self.training)
        # TCN forward
        final_z,z = self.tcn(drop_input.transpose(1, 2))
        z = F.dropout(z, p=self.emb_dropout, training=self.training)
        final_z = F.dropout(final_z, p=self.emb_dropout, training=self.training)
        
        
        # RNN forward
        for i,layer in enumerate(self.network):
            if i==(self.num_levels-1):
                pack_input = pack_padded_sequence(final_z.transpose(1, 2), sorted_len + 1,batch_first=True)
                pack_output, _ = layer(pack_input)
                output, _ = pad_packed_sequence(pack_output, batch_first=True)
                _, reversed_idx = torch.sort(sorted_idx)
                output = output[reversed_idx]
                all_out=all_out.add(output)
            elif i==0:
                pack_input = pack_padded_sequence(z[i,:,:,:].transpose(1, 2), sorted_len + 1,batch_first=True)
                pack_output, _ = layer(pack_input)
                output, _ = pad_packed_sequence(pack_output, batch_first=True)
                _, reversed_idx = torch.sort(sorted_idx)
                output = output[reversed_idx]
                all_out=output
            else:
                pack_input = pack_padded_sequence(z[i,:,:,:].transpose(1, 2), sorted_len + 1,batch_first=True)
                pack_output, _ = layer(pack_input)
                output, _ = pad_packed_sequence(pack_output, batch_first=True)
                _, reversed_idx = torch.sort(sorted_idx)
                output = output[reversed_idx]
                all_out=all_out.add(output)
                

        # project output
        drop_output = F.dropout(all_out, p=self.emb_dropout,
                                training=self.training)
        batch_size, seq_len, hidden_size = drop_output.size()
        logit = self.output(drop_output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)
        
        NLL_loss = NLL(logp, target, length + 1)

        return logp, NLL_loss