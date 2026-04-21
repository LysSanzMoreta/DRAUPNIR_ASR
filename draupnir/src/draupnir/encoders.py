import torch.nn as nn
import torch
import  draupnir
from draupnir.models_utils import *

class RNNEncoder(nn.Module):
    def __init__(self, align_seq_len,
                 aa_prob,n_leaves,
                 gru_hidden_dim,
                 z_dim,
                 input_size,
                 kappa_addition,
                 num_layers,
                 pretrained_params):
        super(RNNEncoder, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.n_leaves = n_leaves
        self.input_size = input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.nndim = int(self.gru_hidden_dim/2)
        self.fc1 = nn.Linear(self.gru_hidden_dim, self.nndim)
        self.linear_means = nn.Linear(self.nndim, self.z_dim)
        self.linear_std = nn.Linear(self.nndim, self.z_dim)

        #self.layernorm = nn.LayerNorm(self.input_size)
        self.softplus = nn.Softplus()

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)

        #todo: necessary? maybe
        #self.init_gru_bias()
    def init_gru_bias(self): #some funky suggestion by llm
        for name, param in self.rnn.named_parameters(): #initialization if the gru bias to avoid dominance
            if "bias_ih_l0" in name:
                param.data[self.gru_hidden_dim:2 * self.gru_hidden_dim] = -1.0  # reset gate bias
    def forward(self, input, hidden):

        #input = self.layernorm(input)
        rnn_hidden_states, rnn_final_bidirectional = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim*2] | [num_layers*2,n_nodes,gru_dim]
        # print("Encoder rnn states")
        # print(rnn_final_bidirectional.var(dim=0).mean())  # should not be tiny
        # print(rnn_hidden_states.var(dim=0).mean())  # should not be tiny
        forward_out_r,backward_out_r = rnn_hidden_states[:,:,:self.gru_hidden_dim],rnn_hidden_states[:,:,self.gru_hidden_dim:]

        rnn_hidden_states = forward_out_r + backward_out_r #original
        rnn_final_forward_backward_sum = rnn_hidden_states[:,-1] #original takes the last state of the forward and the first state of the backward

        #rnn_final_forward_backward_sum = forward_out_r[:,-1] + backward_out_r[:,0] #pick the last state forward and the first state from backwards
        rnn_final_hidden_state = self.fc1(rnn_final_forward_backward_sum)
        z_loc = self.linear_means(rnn_final_hidden_state)
        z_scale = self.softplus(self.linear_std(rnn_final_hidden_state))

        return  {"z_loc":z_loc,
                 "z_scale":z_scale,
                 "rnn_final_bidirectional":rnn_final_bidirectional,
                 "rnn_final_forward_backward_sum":rnn_final_forward_backward_sum.unsqueeze(0), #not transformed
                 "rnn_hidden_states":rnn_hidden_states,
                 "rnn_final_hidden_state":rnn_final_hidden_state}

class RNNEncoder_1b(nn.Module):
    def __init__(self, align_seq_len,
                 aa_prob,n_leaves,
                 gru_hidden_dim,
                 z_dim,
                 input_size,
                 kappa_addition,
                 num_layers,
                 pretrained_params):
        super(RNNEncoder_1b, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.n_leaves = n_leaves
        self.input_size = input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.nndim = int(self.gru_hidden_dim/2)

        self.linear_hidden1 = nn.Linear(int(self.gru_hidden_dim*2),self.gru_hidden_dim)
        self.linear_hidden2 = nn.Linear(self.gru_hidden_dim,self.z_dim)
        self.fc1 = nn.Linear(self.z_dim, self.z_dim)
        self.linear_means = nn.Linear(self.z_dim, self.z_dim)
        self.linear_std = nn.Linear(self.z_dim, self.z_dim)

        self.layernorm0 = nn.LayerNorm(self.gru_hidden_dim)
        self.layernorm1 = nn.LayerNorm(int(self.gru_hidden_dim*2))
        self.layernorm2 = nn.LayerNorm(self.gru_hidden_dim)
        self.layernorm3 = nn.LayerNorm(self.z_dim)
        self.layernorm4 = nn.LayerNorm(self.z_dim)
        self.layernorm5 = nn.LayerNorm(self.z_dim)
        self.layernorm6 = nn.LayerNorm(self.z_dim)

        self.softplus = nn.Softplus()

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)

        #todo: necessary? maybe
        #self.init_gru_bias()
    def init_gru_bias(self): #some funky suggestion by llm
        for name, param in self.rnn.named_parameters(): #initialization if the gru bias to avoid dominance
            if "bias_ih_l0" in name:
                param.data[self.gru_hidden_dim:2 * self.gru_hidden_dim] = -1.0  # reset gate bias
    def forward(self, input, hidden):


        rnn_hidden_states, rnn_final_bidirectional = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim*2] | [num_layers*2,n_nodes,gru_dim]
        # print("Encoder rnn states")
        # print(rnn_final_bidirectional.var(dim=0).mean())  # should not be tiny
        # print(rnn_hidden_states.var(dim=0).mean())  # should not be tiny


        rnn_hidden_states, rnn_final_bidirectional = self.layernorm1(rnn_hidden_states), self.layernorm0(rnn_final_bidirectional)
        rnn_hidden_states = self.layernorm2(self.linear_hidden1(rnn_hidden_states))
        rnn_hidden_states = self.layernorm3(self.linear_hidden2(rnn_hidden_states))
        rnn_hidden_states = self.tanh(rnn_hidden_states)
        #forward_out_r,backward_out_r = rnn_hidden_states[:,:,:self.gru_hidden_dim],rnn_hidden_states[:,:,self.gru_hidden_dim:]

        #rnn_hidden_states = forward_out_r + backward_out_r #original
        rnn_final_forward_backward_sum = rnn_hidden_states[:,-1] #original takes the last state of the forward and the first state of the backward

        #rnn_final_forward_backward_sum = forward_out_r[:,-1] + backward_out_r[:,0] #pick the last state forward and the first state from backwards
        #todo:
        # the hidden states need to be transformed to the zdim state


        rnn_final_hidden_state = self.layernorm4(self.fc1(rnn_final_forward_backward_sum))
        z_loc = self.layernorm5(self.linear_means(rnn_final_hidden_state))
        z_scale = self.softplus(self.layernorm6(self.linear_std(rnn_final_hidden_state)))


        return  {"z_loc":z_loc,
                 "z_scale":z_scale,
                 "rnn_final_bidirectional":rnn_final_bidirectional,
                 "rnn_final_forward_backward_sum":rnn_final_forward_backward_sum.unsqueeze(0), #not transformed
                 "rnn_hidden_states":rnn_hidden_states,
                 "rnn_final_hidden_state":rnn_final_hidden_state}

class FCLEncoder(nn.Module):
    def __init__(self, align_seq_len,aa_prob,n_leaves,gru_hidden_dim, z_dim,input_size, num_layers):
        super(FCLEncoder, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.n_leaves = n_leaves
        self.input_size = input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(self.input_size,self.gru_hidden_dim) #todo: put in sequential module
        self.fc2 = nn.Linear(self.gru_hidden_dim,self.gru_hidden_dim)
        self.linear_means = nn.Linear(self.gru_hidden_dim, self.z_dim)
        self.linear_std = nn.Linear(self.gru_hidden_dim, self.z_dim)
        self.layernorm = nn.LayerNorm(self.input_size)
        self.softplus = nn.Softplus()


        #todo: necessary? maybe
        #self.init_gru_bias()
    def init_gru_bias(self):
        for name, param in self.rnn.named_parameters(): #initialization if the gru bias to avoid dominance
            if "bias_ih_l0" in name:
                param.data[self.gru_hidden_dim:2 * self.gru_hidden_dim] = -1.0  # reset gate bias


    def forward(self, input, hidden):

        #input = self.layernorm(input)


        output = self.fc2(self.fc1(input))
        z_loc = self.linear_means(output)
        z_scale = self.softplus(self.linear_std(output))

        return  {"z_loc":z_loc,
                 "z_scale":z_scale}

        #return output_means,output_std

class xLSTMEncoder(nn.Module):
    def __init__(self,max_len,input_size,z_dim):
        super(xLSTMEncoder, self).__init__()
        from xlstm import (
            xLSTMBlockStack,
            xLSTMBlockStackConfig,
            mLSTMBlockConfig,
            mLSTMLayerConfig,
            sLSTMBlockConfig,
            sLSTMLayerConfig,
            FeedForwardConfig,
        )

        self.input_size = input_size
        self.z_dim = z_dim
        self.num_heads = 3

        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,  # seems to be unaffected
                    qkv_proj_blocksize=self.num_heads,  # set to num_heads
                    num_heads=self.num_heads,
                    proj_factor=2.0,
                    embedding_dim=-1,
                    bias=False,
                    dropout=0.0,
                    context_length=-1,
                    round_proj_up_to_multiple_of=self.input_size*self.num_heads,  # inner_embedding_dim, must be divisible by num_heads
                    round_proj_up_dim_up=True
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    embedding_dim=-1,
                    num_heads=self.num_heads,
                    # this must divide the hidden size, is not yet supported by all versions in this directory
                    conv1d_kernel_size=4,
                    group_norm_weight=True,
                    dropout=0,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3,
                                              act_fn="gelu",
                                              embedding_dim=-1,
                                              dropout=0,
                                              bias=False,
                                              ff_type="ffn_gated"),
            ),
            context_length=max_len,
            num_blocks=4,  # does not seem to matter
            embedding_dim=self.input_size,
            slstm_at=[1],
            add_post_blocks_norm=True,
            bias=False,
            dropout=0.0,

        )

        self.xlstm_stack = xLSTMBlockStack(self.cfg)
        self.softplus = nn.Softplus(dim=None)
        # self.fc1 = nn.Linear(self.input_size, self.input_size)
        # self.linear_means = nn.Linear(self.input_size, self.z_dim)
        # self.linear_std = nn.Linear(self.input_size, self.z_dim)


    def forward(self,input):

        embedding = self.xlstm_stack(input)

        latent = embedding.mean(axis=1)
        # latent = self.fc1(latent)
        # z_loc = self.linear_means(latent)
        # z_scale = self.softplus(self.linear_std(latent))

        return {"embeddings": embedding,
                # "z_loc": z_loc,
                # "z_scale": z_scale
                }

class TransformerEncoder(nn.Module):
    def __init__(self,input_dim,align_seq_len,output_dim):
        super(TransformerEncoder, self).__init__()
        self.enable_flash = True
        self.dropout_rate = 0
        self.input_dim = input_dim
        self.output_dim = output_dim #it is not a bug, in this case we keep it the same at the
        self.align_seq_len = align_seq_len
        self.num_heads = 1
        self.use_bias = False
        self.tril_mask = True
        self.softplus = nn.Softplus()
        self.key_fc = nn.Linear(self.input_dim,self.align_seq_len * self.num_heads,
                                  bias=self.use_bias,
                                  )
        self.context_fc = nn.Linear(self.input_dim, self.output_dim * self.num_heads,
                                        bias=self.use_bias,
                                        )
        self.output_fc = nn.Linear(self.output_dim * self.num_heads, self.output_dim * self.num_heads,
                                        bias=self.use_bias,
                                        )
        self.output_fc_loc = nn.Linear(self.output_dim * self.num_heads,self.output_dim * self.num_heads,
                                  bias=self.use_bias,
                                  )
        self.output_fc_scale = nn.Linear(self.output_dim * self.num_heads, self.output_dim * self.num_heads,
                                        bias=self.use_bias,
                                        )
    def sequence_masking(self,x, mask=None, value=0, axis=None, bias=None, return_mask=False):
        """为序列条件mask的函数
        mask: 形如(batch_size, seq_len)的bool矩阵；
        value: mask部分要被替换成的值，可以是'-inf'或'inf'；
        axis: 序列所在轴，默认为1；
        bias: 额外的偏置项，或者附加的mask；
        return_mask: 是否同时返回对齐后的mask。
        """
        xndim = x.ndim
        if not (mask is None and bias is None):
            if mask is None:
                if bias.dtype == 'bool':
                    mask = bias
                    x = torch.where(mask, x, value)
                else:
                    x = x + bias
            else:
                if axis is None:
                    axes = [1]
                elif isinstance(axis, list):
                    axes = axis
                else:
                    axes = [axis]
                axes = [axis if axis >= 0 else xndim + axis for axis in axes]
                if mask.dtype != 'bool':
                    mask = mask.bool()

                full_mask = self.align(mask, [0, axes[0]], xndim)
                for axis in axes[1:]:
                    full_mask = full_mask & self.align(mask, [0, axis], xndim)

                mask = full_mask
                if bias is None:
                    x = torch.where(mask, x, value)
                elif bias.dtype == 'bool':
                    mask = mask & bias
                    x = torch.where(mask, x, value)
                else:
                    x = torch.where(mask, x + bias, value)

        if return_mask:
            return x, mask
        else:
            return x
    def attention_normalize(self, a:torch.Tensor, dim:float=-1, method:str='softmax'):
        """ Normalize or bound the attention values between 0 and 1
        methods:
            softmax；
            squared_relu: from the Gated Attention Unit with a GLU(Gated Linear Unit) https://arxiv.org/abs/2202.10447 ；
            softmax_plus: Uses entropy invariance scaling for datasets using longer lengths  https://kexue.fm/archives/8823 。
        """
        if method == 'softmax':
            return F.softmax(a, dim=dim)
        else:
            mask = (a > -1e11).float()
            l = torch.maximum(torch.sum(mask, dim=dim, keepdims=True), torch.tensor(1).to(mask))
            if method == 'squared_relu':
                return F.relu(a) ** 2 / l
            elif method == 'softmax_plus':
                return F.softmax(a * torch.log(l) / torch.log(torch.tensor(512.0)).to(mask), dim=dim)
        return a
    def attention(self,query,keys,values,mask):
        att_scores = torch.einsum('bmd,bnd->bmn', query, keys) / self.output_dim ** 0.5 #attention scores/logits(softmax)
        att_scores = self.attention_normalize(att_scores,-1,method="softmax") #[nseqs,L,L]


        #Original code is for seq2seq https://github.com/bojone/bert4keras/blob/2072f06dd410ea885a9c6850ba539effce50b22b/bert4keras/layers.py#L1485-L1487
        # bias = self.query_fc(input) / 2
        # att_scores = att_scores[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None] #in the original code the bias is divided in 2 parts because bias[:,::2,None] belongs to the query and bisas[:,1::2,:,None] belongs to the keys

        #Method2: https://github.com/Tongjilibo/bert4torch/blob/ce644b9cefa72801a4a23366cb2cd9b895511951/bert4torch/layers/global_point.py#L90-L93
        bias_input = self.key_fc(keys)  # [..., heads*feat_dim] #in the original the inputs are 1 big array?
        bias2 = torch.stack(torch.chunk(bias_input, self.num_heads, dim=-1), dim=-2).transpose(1,2) / 2  # [btz, heads, seq_len, seq_len*num_heads]

        logits = att_scores.unsqueeze(1) + bias2  # [batch_size, num_heads, seq_len, seq_len]

        if self.tril_mask:
            tril_mask = torch.triu(torch.ones_like(att_scores[0]), 0)
            tril_mask = tril_mask.bool()
        else:
            tril_mask = None

        att_scores = self.sequence_masking(att_scores, mask, -torch.inf, [2, 3], tril_mask) #TODO: understand and simplify?

        hidden_states = torch.einsum('bij, bjd -> bid', att_scores, values) #[N,L,feat_dim/hidden_dim?]


        hidden_states = self.context_fc(hidden_states) #https://github.com/Tongjilibo/bert4torch/blob/ce644b9cefa72801a4a23366cb2cd9b895511951/bert4torch/layers/attention.py#L725C9-L725C91
        #TODO: What to do with the output when num_heads > 2

        #TODO: Average? I need z_mean and z_scale for the encoder and logits for the decoder? wft
        context_vector = hidden_states.mean(1)
        context_vector = self.output_fc(context_vector)
        z_loc = self.output_fc_loc(context_vector)
        z_scale = self.softplus(self.output_fc_scale(context_vector))

        return {"attention_scores":att_scores,
                "hidden_states": hidden_states,
                "context_vector":context_vector, #TODO: Can be used as logits? [nseqs, alignlen, feat_dim]
                "attention_logits":logits,
                "z_loc":z_loc, #TODO: One loc and scale per position!!! requires changing the distribution
                "z_scale":z_scale}
    def forward(self,query,key,values,mask):
        return self.attention(query,key,values,mask)

class TransformerEncoder2(nn.Module):
    def __init__(self,input_dim,align_seq_len,output_dim):
        super(TransformerEncoder2, self).__init__()
        self.enable_flash = True
        self.dropout_rate = 0
        self.input_dim = input_dim
        self.output_dim = output_dim #it is not a bug, in this case we keep it the same at the
        self.align_seq_len = align_seq_len
        self.num_heads = 1
        self.use_bias = True
        self.softplus = nn.Softplus()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=2)
        self.norm_layer = nn.LayerNorm(self.input_dim)
        self.encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=1, norm=self.norm_layer)
        self.hidden_states_fc = nn.Linear(self.input_dim, self.output_dim,
                                        bias=self.use_bias,
                                        )

        self.output_fc = nn.Linear(self.output_dim, self.output_dim,
                                        bias=self.use_bias,
                                        )
        self.output_fc_loc = nn.Linear(self.output_dim ,self.output_dim ,
                                  bias=self.use_bias,
                                  )
        self.output_fc_scale = nn.Linear(self.output_dim, self.output_dim,
                                        bias=self.use_bias,
                                        )
    def forward(self,input,mask):

        hidden_states = self.encoder(input)
        hidden_states = self.hidden_states_fc(hidden_states)
        context_vector = hidden_states.mean(axis=1)

        context_vector = self.output_fc(context_vector)
        z_loc = self.output_fc_loc(context_vector)
        z_scale = self.softplus(self.output_fc_scale(context_vector))


        return {"context_vector":context_vector,
                "hidden_states":hidden_states, #TODO: Can be used as logits? [nseqs, alignlen, feat_dim]
                "z_loc":z_loc,
                "z_scale":z_scale}

class TransformerEncoder3(nn.Module):
    def __init__(self,input_dim,adapted_input_dim, align_seq_len,output_dim):
        """
        input_dim is the embedding size (i.e aa probs)
        adapted_input_dim is the corrected input size for when the number is odd

        """
        super(TransformerEncoder3, self).__init__()

        self.dropout_rate = 0
        self.input_dim = input_dim
        self.adapted_input_dim = adapted_input_dim
        self.output_dim = output_dim

        self.align_seq_len = align_seq_len
        self.num_heads = 1
        self.use_bias = True
        self.softplus = nn.Softplus()
        self.positional_embeddings = PositionalEncodings(self.align_seq_len,self.adapted_input_dim,base=1000, type="sinusoidal")
        self.layernorm = nn.LayerNorm(self.adapted_input_dim)  # todo: not sure yet
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.adapted_input_dim, nhead=2) #includes residual connections
        self.encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=1, norm=self.layernorm)
        self.hidden_states_fc = nn.Linear(self.adapted_input_dim, self.output_dim,
                                        bias=self.use_bias,
                                        )
        self.augmented_output_dim = int(self.output_dim*2)
        self.output_fc = nn.Linear(self.output_dim, self.augmented_output_dim,
                                        bias=self.use_bias,
                                        )
        self.output_fc_loc = nn.Linear(self.augmented_output_dim ,self.output_dim,
                                  bias=self.use_bias,
                                  )
        self.output_fc_scale = nn.Linear(self.augmented_output_dim, self.output_dim,
                                        bias=self.use_bias,
                                        )
    def forward(self,input,mask):

        #todo: masking & more heads?

        input = self.positional_embeddings(input) # simple summation, this returns 21 + 1 dimensions, otherwise the transformer does not work
        hidden_states = self.encoder(input)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.hidden_states_fc(hidden_states) # i have projected them to 30 perhaps project to 30)

        #context_vector = hidden_states.mean(axis=1) # mean pooling

        context_vector = hidden_states[:,0,:] # get the CLS token (the one with most information)

        context_vector = self.output_fc(context_vector)
        z_loc = self.output_fc_loc(context_vector)
        z_scale = self.softplus(self.output_fc_scale(context_vector))




        return {"context_vector":context_vector,
                "hidden_states":hidden_states,
                "z_loc":z_loc,
                "z_scale":z_scale}