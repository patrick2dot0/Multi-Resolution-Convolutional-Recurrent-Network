import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils import weight_norm
from tqdm import tqdm

data_path = "./jpegs_256/"


## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


## ---------------------- Dataloaders ---------------------- ##
# for 3DCNN
class Dataset_3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, folders, labels, frames, transform=None):
        "Initialization"
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, data_path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(data_path, selected_folder, 'frame{:06d}.jpg'.format(i))).convert('L')

            if use_transform is not None:
                image = use_transform(image)

            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(data_path, folder, self.transform).unsqueeze_(0)                      # (input) spatial images
        y = torch.from_numpy(np.array(self.labels[index])).type(torch.LongTensor)   # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y


# for CRNN
class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, folders, labels, frames, transform=None):
        "Initialization"
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, data_path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(data_path, selected_folder, 'frame{:06d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(data_path, folder, self.transform)                  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])   # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y


# for CRNN- varying lengths (frames)
class Dataset_CRNN_varlen(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, lists, labels, frames, transform=None):
        "Initialization"
        self.labels = labels
        self.folders, self.frame_num = list(zip(*lists))
        self.frames = frames
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, data_path, selected_folder, selected_frame_num, use_transform):
        # load selected frames
        selected_frames = np.arange(self.frames['begin'],
                                    min(selected_frame_num, self.frames['max']) + 1, self.frames['skip'])
        # max frame number
        max_frame_num = len(np.arange(self.frames['begin'], self.frames['max'] + 1, self.frames['skip']))

        X = []
        for i in selected_frames:
            image = Image.open(os.path.join(data_path, selected_folder, 'frame{:06d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        # pad to max frame
        if max_frame_num - X.size(0) > 0:
            # padding if frames are shorter than max
            X_padded = torch.cat((X, torch.zeros(max_frame_num - X.size(0), X.size(1), X.size(2), X.size(3))), 0)
        else:
            X_padded = X

        del X

        return X_padded

    def __getitem__(self, index):
        "Generates one sample of data"
        # select sample
        selected_folder = self.folders[index]
        frame_length = self.frame_num[index]

        # Load data
        X_padded = self.read_images(data_path, selected_folder, frame_length, self.transform)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print('selected folder:{}'.format(selected_folder))
        # print('X_padded size:', X_padded.size(), 'frame_length: ', frame_length)
        frame_length = torch.LongTensor([frame_length])

        return X_padded, frame_length, y

## ---------------------- end of Dataloaders ---------------------- ##



## -------------------- (reload) model prediction ---------------------- ##
def Conv3d_final_prediction(model, device, loader):
    model.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred


def CRNN_final_prediction(model, device, loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = rnn_decoder(cnn_encoder(X))
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred

## -------------------- end of model prediction ---------------------- ##



## ------------------------ 3D CNN module ---------------------- ##
def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

class CNN3D(nn.Module):
    def __init__(self, t_dim=120, img_x=90, img_y=120, drop_p=0.2, fc_hidden1=256, fc_hidden2=128, num_classes=50):
        super(CNN3D, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2],
                             self.fc_hidden1)  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = multi-classes

    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x

## --------------------- end of 3D CNN module ---------------- ##



## ------------------------ CRNN module ---------------------- ##

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape


# 2D CNN encoder train from scratch (no transfer learning)
class EncoderCNN(nn.Module):
    def __init__(self, img_x=90, img_y=120, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.pd1, self.k1, self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),                      
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.drop = nn.Dropout2d(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1], self.fc_hidden1)   # fully connected layer, output k classes
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)   # output = CNN embedding latent variables

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # CNNs
            x = self.conv1(x_3d[:, t, :, :, :])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)           # flatten the output of conv

            # FC layers
            x = F.relu(self.fc1(x))
            # x = F.dropout(x, p=self.drop_p, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


# 2D CNN encoder using ResNet-152 pretrained
class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)             # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN_varlen(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN_varlen, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN, x_lengths):
        
        # Sort the input and lengths as the descending order
        # lengths, perm_index = lengths.sort(0, descending=True)
        # input = input[perm_index]
        #
        # packed_input = pack(input, list(lengths.data), batch_first=True)
        # output, hidden = self.rnn(packed_input, hidden)
        # output = unpack(output, batch_first=True)[0]

        # Sort the input and lengths as the descending order
        # a = torch.arange(0, x_RNN.size(0)).to(x_RNN.device)
        # length_order = torch.cat((x_lengths.view(-1, 1), a.view(-1, 1)), 1)
        # _, perm_index = length_order.sort(0, descending=True)
        # decrease_length, order, inverse_order = _[:, 0], perm_index[:, 0], perm_index[:, 1]
        
        N, T, n = x_RNN.size()
        # print('x_RNN.size:', x_RNN.size(), 'x_lengths:', x_lengths)

        for i in range(N):
            if x_lengths[i] < T:
                x_RNN[i, x_lengths[i]:, :] = torch.zeros(T - x_lengths[i], n, dtype=torch.float, device=x_RNN.device)

        x_lengths[x_lengths > T] = T
        lengths_ordered, perm_idx = x_lengths.sort(0, descending=True)

        # use input of descending length
        packed_x_RNN = torch.nn.utils.rnn.pack_padded_sequence(x_RNN[perm_idx], lengths_ordered, batch_first=True)
        # self.LSTM.flatten_parameters()
        packed_RNN_out, (h_n_sorted, h_c_sorted) = self.LSTM(packed_x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        RNN_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out, batch_first=True)
        RNN_out = RNN_out.contiguous()
        # RNN_out = RNN_out.view(-1, RNN_out.size(2))
        
        # reverse back to original sequence order
        _, unperm_idx = perm_idx.sort(0)
        RNN_out = RNN_out[unperm_idx]

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

## ---------------------- end of CRNN module ---------------------- ##

## ----------------------     TCN module     ---------------------- ##

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

class DecoderTCN(nn.Module):
    def __init__(self, CNN_embed_dim=300, num_channels=[256]*3, layers=3, h_FC_dim=128, kernel_size=2, drop_p=0.1, num_classes=50):
        super(DecoderTCN, self).__init__()

        self.TCN_input_size = CNN_embed_dim
        self.num_channels = num_channels
        self.layers = layers
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.tcn = TemporalConvNet(CNN_embed_dim, num_channels, kernel_size=kernel_size, dropout=drop_p)

        self.fc1 = nn.Linear(self.num_channels[layers-1], self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_TCN, x_lengths):
        
        N, T, n = x_TCN.size()
        # print('x_TCN.size:', x_TCN.size(), 'x_lengths:', x_lengths)

        TCN_out, __ = self.tcn(x_TCN.transpose(1, 2))
        TCN_out = TCN_out.transpose(1, 2)

        # FC layers
        x = self.fc1(TCN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

## ---------------------- end of TCN module  ---------------------- ##

## ----------------------     CRN module     ---------------------- ##

class DecoderCRN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=1, h_RNN=256, num_channels=[256]*3, layers=3, h_FC_dim=128, kernel_size=2, drop_p=0.1, num_classes=50):
        super(DecoderCRN, self).__init__()

        self.TCN_input_size = CNN_embed_dim
        self.num_channels = num_channels
        self.layers = layers
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.tcn = TemporalConvNet(CNN_embed_dim, num_channels, kernel_size=kernel_size, dropout=drop_p)

        self.LSTM = nn.LSTM(
            input_size=self.num_channels[layers-1],
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_TCN, x_lengths):

        N, T, n = x_TCN.size()
        # print('x_RNN.size:', x_RNN.size(), 'x_lengths:', x_lengths)

        

        TCN_out, __ = self.tcn(x_TCN.transpose(1, 2))
        TCN_out = TCN_out.transpose(1, 2)

        for i in range(N):
            if x_lengths[i] < T:
                TCN_out[i, x_lengths[i]:, :] = torch.zeros(T - x_lengths[i], n, dtype=torch.float, device=x_TCN.device)


        x_lengths[x_lengths > T] = T
        lengths_ordered, perm_idx = x_lengths.sort(0, descending=True)

        # use input of descending length
        packed_x_RNN = torch.nn.utils.rnn.pack_padded_sequence(TCN_out[perm_idx], lengths_ordered, batch_first=True)
        # self.LSTM.flatten_parameters()
        packed_RNN_out, (h_n_sorted, h_c_sorted) = self.LSTM(packed_x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        RNN_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out, batch_first=True)
        RNN_out = RNN_out.contiguous()
        # RNN_out = RNN_out.view(-1, RNN_out.size(2))
        
        # reverse back to original sequence order
        _, unperm_idx = perm_idx.sort(0)
        RNN_out = RNN_out[unperm_idx]
        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

## ---------------------- end of MRCRN module  ---------------------- ##

class DecoderMRCRN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=1, h_RNN=256, num_channels=[256]*3, layers=3, h_FC_dim=128, kernel_size=2, drop_p=0.1, num_classes=50):
        super(DecoderMRCRN, self).__init__()

        self.TCN_input_size = CNN_embed_dim
        self.num_channels = num_channels
        self.layers = layers
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.tcn = TemporalConvNet(CNN_embed_dim, num_channels, kernel_size=kernel_size, dropout=drop_p)

        RNNs = []
        for i in range(self.layers):
            RNNs += [nn.LSTM(
                input_size=self.num_channels[layers-1],
                hidden_size=self.h_RNN,        
                num_layers=h_RNN_layers,       
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )]
        self.network = nn.Sequential(*RNNs)

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_TCN, x_lengths):

        N, T, n = x_TCN.size()
        # print('x_RNN.size:', x_RNN.size(), 'x_lengths:', x_lengths)

        

        TCN_out, z = self.tcn(x_TCN.transpose(1, 2))
        TCN_out = TCN_out.transpose(1, 2)
        z = z.transpose(2,3)
        
        for i in range(N):
            if x_lengths[i] < T:
                TCN_out[i, x_lengths[i]:, :] = torch.zeros(T - x_lengths[i], n, dtype=torch.float, device=x_TCN.device)
                for j in range(self.layers-1):
                    z[j,i, x_lengths[i]:, :] = torch.zeros(T - x_lengths[i], n, dtype=torch.float, device=x_TCN.device)
                
        x_lengths[x_lengths > T] = T
        lengths_ordered, perm_idx = x_lengths.sort(0, descending=True)

        for j,layer in enumerate(self.network):
            if j==0:

                # use input of descending length
                packed_x_RNN = torch.nn.utils.rnn.pack_padded_sequence(z[j,perm_idx], lengths_ordered, batch_first=True)
                # self.LSTM.flatten_parameters()
                packed_RNN_out, (h_n_sorted, h_c_sorted) = layer(packed_x_RNN, None)
                """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
                """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

                RNN_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out, batch_first=True)
                RNN_out = RNN_out.contiguous()
                # RNN_out = RNN_out.view(-1, RNN_out.size(2))
                    
                # reverse back to original sequence order
                _, unperm_idx = perm_idx.sort(0)
                RNN_out = RNN_out[unperm_idx]
                all_out = RNN_out
            elif j==(self.layers - 1):

                # use input of descending length
                packed_x_RNN = torch.nn.utils.rnn.pack_padded_sequence(TCN_out[perm_idx], lengths_ordered, batch_first=True)
                # self.LSTM.flatten_parameters()
                packed_RNN_out, (h_n_sorted, h_c_sorted) = layer(packed_x_RNN, None)
                """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
                """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

                RNN_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out, batch_first=True)
                RNN_out = RNN_out.contiguous()
                # RNN_out = RNN_out.view(-1, RNN_out.size(2))
                    
                # reverse back to original sequence order
                _, unperm_idx = perm_idx.sort(0)
                RNN_out = RNN_out[unperm_idx]
                all_out.add(RNN_out)
            else:

                # use input of descending length
                packed_x_RNN = torch.nn.utils.rnn.pack_padded_sequence(z[j,perm_idx], lengths_ordered, batch_first=True)
                # self.LSTM.flatten_parameters()
                packed_RNN_out, (h_n_sorted, h_c_sorted) = layer(packed_x_RNN, None)
                """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
                """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

                RNN_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out, batch_first=True)
                RNN_out = RNN_out.contiguous()
                # RNN_out = RNN_out.view(-1, RNN_out.size(2))
                    
                # reverse back to original sequence order
                _, unperm_idx = perm_idx.sort(0)
                RNN_out = RNN_out[unperm_idx]
                all_out.add(RNN_out)


        # FC layers
        x = self.fc1(all_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

## ---------------------- end of MRCRN module  ---------------------- ##