import os
import re
import time
import numpy as np
import pickle as pk 

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn.utils as utils
from Levenshtein import distance 
from torch.autograd import Variable
from torch.distributions.gumbel import Gumbel
from torch.utils.data import DataLoader, Dataset

if torch.cuda.is_available():
    USE_CUDA = True
    print("Cuda available!")
else:
    USE_CUDA = False 

class BackHook(torch.nn.Module):
    def __init__(self, hook):
        super(BackHook, self).__init__()
        self._hook = hook
        self.register_backward_hook(self._backward)

    def forward(self, *inp):
        return inp

    @staticmethod
    def _backward(self, grad_in, grad_out):
        self._hook()
        return None

class WeightDrop(torch.nn.Module):
    """
    Implements drop-connect, as per Merity, https://arxiv.org/abs/1708.02182
    """
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()
        self.hooker = BackHook(lambda: self._backward())

    def _setup(self):
        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(
                self.dropout, name_w))
            w = getattr(self.module, name_w)
            self.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            if self.training:
                mask = raw_w.new_ones((raw_w.size(0), 1))
                mask = torch.nn.functional.dropout(mask,
                                                   p=self.dropout,
                                                   training=True)
                w = mask.expand_as(raw_w) * raw_w
                setattr(self, name_w + "_mask", mask)
            else:
                w = raw_w
            rnn_w = getattr(self.module, name_w)
            rnn_w.data.copy_(w)

    def _backward(self):
        # transfer gradients from embeddedRNN to raw params
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            rnn_w = getattr(self.module, name_w)
            raw_w.grad = rnn_w.grad * getattr(self, name_w + "_mask")

    def forward(self, *args):
        self._setweights()
        return self.module(*self.hooker(*args))

class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                # Code borrowed from Pytorch forums: https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791
                for name, param in self.blstm.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_normal_(param)

    def forward(self, x):
        return self.blstm(x)

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, dropout=0.1):
        if not self.training or not dropout:
            return x
        m = x.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask*x

class Listener(nn.Module):
    def __init__(self, input_dim, hidden_dim, value_size=128, key_size=128):
        super(Listener, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)

        # Define blocks of pyramidal pBLSTMs
        self.prnn = pBLSTM(input_dim=hidden_dim*2, hidden_dim=hidden_dim)
        self.dropout = LockedDropout()

        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                # Code borrowed from Pytorch forums: https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791
                for name, param in self.lstm.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_normal_(param)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def reshape_and_pool(self, inputs):
        # Check if the input length is odd and cut it
        if inputs.shape[0] % 2 != 0:
            inputs = inputs[:-1,:,:]

        outputs = torch.transpose(inputs,0,1)
        outputs = outputs.view(outputs.shape[0],outputs.shape[1]//2,2,outputs.shape[2])
        outputs = torch.mean(outputs, 2)
        outputs = torch.transpose(outputs,0,1)

        return outputs

    def layers_lstm(self, linear_input, lens):
        outputs = self.reshape_and_pool(linear_input)
        lens = lens//2
        rnn_inp = utils.rnn.pack_padded_sequence(outputs, lengths=lens, enforce_sorted=False)
        outputs, _ = self.prnn(rnn_inp)

        return WeightDrop(outputs, ['weight_hh_l0', 'weight_hh_l0_reverse'], dropout=0.5), lens

    def forward(self, x, lens):
        rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, enforce_sorted=False)
        outputs, _ = self.lstm(rnn_inp)
        linear_input, _ = utils.rnn.pad_packed_sequence(outputs)

        # Pass outputs through pBLSTM blocks
        # Layer 1
        outputs, lens = self.layers_lstm(linear_input=linear_input, lens=lens)
        op = self.dropout(outputs, dropout=0.2)
        linear_input, _ = utils.rnn.pad_packed_sequence(op)

        # Layer 2
        outputs, lens = self.layers_lstm(linear_input=linear_input, lens=lens)
        op = self.dropout(outputs, dropout=0.2)
        linear_input, _ = utils.rnn.pad_packed_sequence(op)

        # Layer 3
        outputs, lens = self.layers_lstm(linear_input=linear_input, lens=lens)
        op = self.dropout(outputs, dropout=0.2)
        linear_input, _ = utils.rnn.pad_packed_sequence(op)

        # Generate key and value pairs
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value, lens

class Attention(nn.Module):
  def __init__(self):
        super(Attention, self).__init__()
        self.sm = nn.Softmax(dim=1)
  def forward(self, query, key, value, lens):
        '''
        :param query :(N,context_size) Query is the output of LSTMCell from Decoder
        :param key: (N,key_size) Key Projection from Encoder per time step
        :param value: (N,value_size) Value Projection from Encoder per time step
        :return output: Attended Context
        :return attention_mask: Attention mask that can be plotted  
        '''
        # Convert the shape of Key from (T,N,H) -> (N,T,H)
        key = torch.transpose(key, 0, 1)                    # Key is a padded sequence
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        mask = torch.arange(energy.size(1)).unsqueeze(0) >= lens.unsqueeze(1)
        mask = mask.to(device)
        energy.masked_fill_(mask, -1e9)
        attention = self.sm(energy)
        
        value = torch.transpose(value,0,1)
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)

        return context

class Speller(nn.Module):
    def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
        super(Speller, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim+value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)
        self.isAttended = isAttended
        if(isAttended):
            self.attention = Attention()
        self.character_prob = nn.Linear(key_size+value_size,vocab_size)

    def forward(self, key, values, lens, text=None, train=True):
        '''
        :param key :(T,N,key_size) Output of the Encoder Key projection layer
        :param values: (T,N,value_size) Output of the Encoder Value projection layer
        :param text: (N,text_len) Batch input of text with text_length
        :param lens: (N,) Lengths of sequence inputs
        :param train: Train or eval mode
        :return predictions: Returns the character perdiction probability 
        '''
        batch_size = key.shape[1]
        if(train):
            # Max_len here defines the batch
            text = torch.transpose(text,0,1)
            max_len =  text.shape[1]
            embeddings = self.embedding(text)
        else:
            max_len = 250
        
        predictions = []
        hidden_states = [None, None]
        prediction = torch.zeros(batch_size,1).to(device)
        context = values[0,:,:]

        for i in range(max_len):
            if(train):
                # Teacher Forcing and Gumbel Noise
                rand = np.random.random_sample()
                if rand > TEACHER_FORCING_PARAM:
                    prediction = Gumbel(prediction.to('cpu'), torch.tensor([0.25])).sample().to(device)
                    char_embed = self.embedding(prediction.argmax(dim=-1))
                else:
                    char_embed = embeddings[:,i,:]
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1))
            
            # Compute outputs from two LSTM cells
            inp = torch.cat([char_embed, context], dim=1)
            hidden_states[0] = self.lstm1(inp,hidden_states[0])
            
            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2,hidden_states[1])

            # Compute attention from the output of the second LSTM Cell
            output = hidden_states[1][0]
            context = self.attention(output, key, values, lens)
            
            # Pass the output through a linear layer
            prediction = self.character_prob(torch.cat([output, context], dim=1))
            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)

class Speech2Text_Dataset(Dataset):
    def __init__(self, speech, text=None, train=True):
        self.speech = speech
        self.train = train
        if (text is not None):
            self.text = text
    def __len__(self):
        return self.speech.shape[0]
    def __getitem__(self, index):
        if(self.train):
            text = self.text[index]
            decoder_labels = text[:-1]
            true_labels = text[1:]
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(decoder_labels), torch.tensor(true_labels)
        else:
            return torch.tensor(self.speech[index].astype(np.float32))

class Seq2Seq(nn.Module):
    def __init__(self,input_dim,vocab_size,hidden_dim,value_size=128, key_size=128,isAttended=True):
        super(Seq2Seq,self).__init__()

        # Define the encoder and decoder 
        self.encoder = Listener(input_dim, hidden_dim)
        self.decoder = Speller(vocab_size, hidden_dim, isAttended=True)

    def forward(self,speech_input, speech_len, text_input=None,train=True):
        key, value, lens = self.encoder(speech_input, speech_len)
        if(train):
            predictions = self.decoder(key, value, lens, text_input)
        else:
            predictions = self.decoder(key, value, lens, text=None, train=False)
        return predictions

def transform_letter_to_index(transcript, letter_list):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    index_list = []
    # Loops over each utterance
    for i in range(transcript.shape[0]):
        word_list = []
        word_list.append(letter_list.index('<sos>'))
        # Loops over each word in an utterance
        for j in range(transcript[i].shape[0]):
            # Loops over each letter in a word
            for elem in list(transcript[i][j].decode("utf-8")):
                word_list.append(letter_list.index(elem))
            if j != transcript[i].shape[0]-1:
                word_list.append(letter_list.index(' '))
        word_list.append(letter_list.index('<eos>'))
        index_list.append(word_list)

    return index_list

def transform_index_to_letter(transcript, letter_list):
    '''
    :param transcript : 
    :param letter_list: Letter list defined above
    :return index_to_letter: Returns a list for all the transcript sentence to index
    '''
    op_list = ""
    final_str = transcript
    for i in range(final_str.shape[0]):
        op_list += letter_list[final_str[i]]
    
    final_str = re.sub(r'<eos>([\w+]*)<sos>','<eos> <sos>', op_list)
    #print(final_str)
    return op_list

def load_data():
    speech_train = np.load('train_new.npy', allow_pickle=True, encoding='bytes')
    speech_dev = np.load('dev_new.npy', allow_pickle=True, encoding='bytes')
    speech_test = np.load('test_new.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load('./train_transcripts.npy', allow_pickle=True, encoding='bytes')
    transcript_valid = np.load('./dev_transcripts.npy', allow_pickle=True, encoding='bytes')

    print("Data loaded successfully")

    return speech_train, speech_dev, speech_test, transcript_train, transcript_valid

# Collate function for test
def test_collate(batch):
    inputs = batch
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    input_len = torch.LongTensor([len(inp) for inp in inputs])
    inputs = utils.rnn.pad_sequence(inputs)

    return inputs, input_len

# Collate function for train
def speech_collate(batch):
    inputs, decode_labels, true_labels = zip(*batch)

    lens = [len(seq) for seq in inputs]
    dec_lens = [len(seq) for seq in decode_labels]
    true_lens = [len(seq) for seq in true_labels]
    
    inputs = [inputs[i] for i in range(len(lens))]
    decode_labels = [decode_labels[i] for i in range(len(dec_lens))]
    true_labels = [true_labels[i] for i in range(len(true_lens))]

    input_len = torch.LongTensor([len(inp) for inp in inputs])
    decode_labels_len = torch.LongTensor([len(inp) for inp in decode_labels])
    true_labels_len = torch.LongTensor([len(inp) for inp in true_labels])

    inputs = utils.rnn.pad_sequence(inputs)
    decode_labels = utils.rnn.pad_sequence(decode_labels)
    true_labels = utils.rnn.pad_sequence(true_labels)
    
    return inputs, decode_labels, true_labels, input_len, decode_labels_len, true_labels_len

def Main():

    global device 
    global TEACHER_FORCING_PARAM

    NUM_EPOCHS = 50
    TEACHER_FORCING_PARAM = 0.8     # % of how much true labels you want to send into network
    BATCH_SIZE = 256
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    kwargs = {'num_workers':8, 'pin_memory':True} if USE_CUDA else {}

    # List containing all Letters 
    letter_list = ['@','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',\
             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

    # Load data
    speech_train, speech_dev, speech_test, transcript_train, transcript_valid = load_data()

    # Convert the transcript to a list of indexes
    character_text_train = transform_letter_to_index(transcript_train, letter_list)
    character_text_valid = transform_letter_to_index(transcript_valid, letter_list)

    print("List of indexes generated succesfully")

    model = Seq2Seq(input_dim=40,vocab_size=len(letter_list),hidden_dim=512, value_size=128, key_size=128)
    if USE_CUDA:
        model.to(device)

    # Initialize the data
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduce=False).to(device)

    train_dataset = Speech2Text_Dataset(speech_train, character_text_train)
    val_dataset = Speech2Text_Dataset(speech_dev, character_text_valid)
    test_dataset = Speech2Text_Dataset(speech_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=speech_collate, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=speech_collate, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_collate, **kwargs)
    
    for epoch in range(NUM_EPOCHS):
        # Train the model
        model.train()
        tot_loss = 0.0

        for batch_num,(train, decode_labels, true_labels, train_len, decode_labels_len, true_labels_len) in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                if USE_CUDA:
                    train, decode_labels, true_labels = train.to(device), decode_labels.to(device), true_labels.to(device)

                optimizer.zero_grad()

                pred = model(train, train_len, decode_labels)
                mask = torch.zeros(true_labels.size()).to(device)

                pred = pred.contiguous().view(-1, pred.size(-1))
                true_labels = torch.transpose(true_labels,0,1).contiguous().view(-1)

                for idx, length in enumerate(true_labels_len):
                    mask[:length,idx] = 1
                
                mask = mask.view(-1).to(device)

                loss = criterion(pred, true_labels)
                masked_loss = torch.sum(loss*mask)

                masked_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()

                current_loss = float(masked_loss.item())/int(torch.sum(mask).item())

                if batch_num % 25 == 1:
                    # Decode the output using Greedy Search
                    # TODO: Implement Beam Search here
                    logits = pred.data
                    pred_word_list = []
                    _, max_index = torch.max(logits, dim=1)

                    decoded_str = transform_index_to_letter(max_index, letter_list)
                    correct_labels = transform_index_to_letter(true_labels, letter_list)

                    computed_dist = distance(decoded_str, correct_labels)
                    print("Epoch ", epoch, "Training Loss ", current_loss, "Distance ", computed_dist)

                torch.cuda.empty_cache()
                del train 
                del decode_labels
                del true_labels
                del mask

        # Save the model
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    },os.path.join(os.getcwd(),"models","model_run2","model{}.pth".format(epoch+1)))

        model.eval()

        # Validate the model
        for batch_num,(dev, decode_labels, true_labels, dev_len, decode_labels_len, true_labels_len) in enumerate(val_loader):
            if USE_CUDA:
                dev = dev.to(device)

            pred = model(dev, dev_len, decode_labels, train=False) 
            pred = pred.contiguous().view(-1, pred.size(-1))

            if batch_num % 100 == 1:
                # Decode the output using Greedy Search
                # TODO: Implement Beam Search here
                logits = pred.data
                pred_word_list = []
                _, max_index = torch.max(logits, dim=1)

                decoded_str = transform_index_to_letter(max_index, letter_list)
                correct_labels = transform_index_to_letter(true_labels, letter_list)

                computed_dist = distance(decoded_str, correct_labels)
                print(decoded_str)
                print(computed_dist)

            torch.cuda.empty_cache()
            del dev 
            
if __name__=="__main__":
    Main()

    