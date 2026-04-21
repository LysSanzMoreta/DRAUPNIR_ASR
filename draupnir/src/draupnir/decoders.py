from draupnir.models_utils import *
from draupnir.likelihoods import LowRankPottsPseudoLikelihood

class SequenceDecoderBase(nn.Module, ABC):
    """Common interface for scaffold-compatible sequence decoders."""

    @abstractmethod
    def reconstruction_log_prob(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return one reconstruction score per sequence in the batch."""
        raise NotImplementedError

    @abstractmethod
    def decode_sequences(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode a sequence tensor from latent vectors."""
        raise NotImplementedError
class RNNDecoder_Tiling(nn.Module):
    def __init__(self,
                 align_seq_len,
                 aa_probs,
                 gru_hidden_dim,
                 z_dim,
                 input_size,
                 kappa_addition,
                 num_layers,
                 pretrained_params):
        super(RNNDecoder_Tiling, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.input_size = input_size
        self.align_seq_len = align_seq_len
        self.aa_probs = aa_probs
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        #self.layernorm = nn.LayerNorm(self.input_size)
        self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
        self.linear_probs = nn.Linear(self.gru_hidden_dim, self.aa_probs)
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)
        #todo: necessary? does not seem to hurt

        #self.init_gru_bias()

    def init_gru_bias(self):
        for name, param in self.rnn.named_parameters(): #initialization if the gru bias to avoid dominance
            if "bias_ih_l0" in name:
                param.data[self.gru_hidden_dim:2 * self.gru_hidden_dim] = -1.0  # reset gate bias


    # def forward(self, input, hidden): #old function for RNNDecoder_Tiling_new
    #     rnn_hidden_states, rnn_final_hidden = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim]
    #     forward_hidden_states = rnn_hidden_states[:, :, :self.gru_hidden_dim]
    #     backward_hidden_states = rnn_hidden_states[:, :, self.gru_hidden_dim:]
    #     rnn_hidden_states = forward_hidden_states + backward_hidden_states
    #     #rnn_final_hidden = rnn_hidden_states[:,-1]
    #
    #     rnn_final_hidden = self.fc1(rnn_hidden_states)
    #     output_logits = self.logsoftmax(self.linear_probs(rnn_final_hidden))  # [n_nodes,align_seq_len,aa_probs]
    #     #output_logits = self.logsoftmax(self.linear_probs(self.fc1(rnn_final_hidden)))  # [n_nodes,align_seq_len,aa_probs]
    #     return output_logits

    def forward(self, input, hidden):
        """One-shot, non-autoregressive sequence generation"""
        #input = self.layernorm(input) #added extra

        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim] #rnn_out is not expressive? whereas the hidden states are

        # print("Decoder rnn states")
        # print(rnn_output.var(dim=0).mean())  # should not be tiny
        # print(rnn_hidden.var(dim=0).mean())  # should not be tiny
        #forward_out = rnn_output[:, :, :self.gru_hidden_dim]
        #backward_out = rnn_output[:, :, self.gru_hidden_dim:]
        #rnn_output_out = torch.cat((forward_out, backward_out), dim=2)
        #rnn_output = self.layernorm(rnn_output) #worsens training, not necessary, since all the sequences an in the same "scale"
        output_logits = self.logsoftmax(self.linear_probs(self.fc1(rnn_output)))  # [n_nodes,align_seq_len,aa_probs]

        return output_logits

# ----------------------------
# Bahdanau Attention
# ----------------------------
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.W_enc = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.W_dec = nn.Linear(dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_outputs):
        """
        encoder_outputs: [B, src_len, enc_hid_dim]
        decoder_outputs: [B, tgt_len, dec_hid_dim]

        Returns:
            context: [B, tgt_len, enc_hid_dim]
            attention_weights: [B, tgt_len, src_len]
        """


        B, src_len, _ = encoder_outputs.size()
        tgt_len = decoder_outputs.size(1)


        # Expand for broadcasting
        enc_proj = self.W_enc(encoder_outputs)  # [B, src_len, dec_hid_dim]
        dec_proj = self.W_dec(decoder_outputs)  # [B, tgt_len, dec_hid_dim]

        enc_proj = enc_proj.unsqueeze(1)  # [B, 1, src_len, dec_hid_dim]
        dec_proj = dec_proj.unsqueeze(2)  # [B, tgt_len, 1, dec_hid_dim]

        energy = torch.tanh(enc_proj + dec_proj)  # [B, tgt_len, src_len, dec_hid_dim] #across the second dimension, each of the rows of the decoder broadcasts+sum those of the encoder

        scores = self.v(energy).squeeze(-1)  # [B, tgt_len, src_len]

        attn_weights = F.softmax(scores, dim=-1)  # [B, tgt_len, src_len]

        context = torch.bmm(attn_weights, encoder_outputs)# [B, tgt_len, enc_hid_dim]

        return context, attn_weights
class RNNDecoder_CrossAttention(nn.Module):
    def __init__(self,
                 align_seq_len,
                 aa_probs,
                 gru_hidden_dim,
                 z_dim,
                 input_size,
                 kappa_addition,
                 num_layers,
                 pretrained_params):
        super(RNNDecoder_CrossAttention, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.input_size = input_size
        self.align_seq_len = align_seq_len
        self.aa_probs = aa_probs
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        #self.layernorm = nn.LayerNorm(self.input_size)
        self.fc1 = nn.Linear(2*self.gru_hidden_dim, self.gru_hidden_dim)
        self.linear_probs = nn.Linear(self.gru_hidden_dim, self.aa_probs)
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)
        self.layernorm1 = nn.LayerNorm(2*self.gru_hidden_dim)
        self.layernorm2 = nn.LayerNorm(self.gru_hidden_dim)
        self.layernorm3 = nn.LayerNorm(self.aa_probs)
        self.attention = Attention(self.gru_hidden_dim,self.gru_hidden_dim)


    def init_gru_bias(self):
        for name, param in self.rnn.named_parameters(): #initialization if the gru bias to avoid dominance
            if "bias_ih_l0" in name:
                param.data[self.gru_hidden_dim:2 * self.gru_hidden_dim] = -1.0  # reset gate bias

    def forward(self, input, hidden,encoder_hidden_states= None):
        """Autoregressive sequence generation"""


        decoder_hidden_states, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim] #rnn_out is not expressive? whereas the hidden states are


        # print("Decoder rnn states")
        # print(rnn_output.var(dim=0).mean())  # should not be tiny
        # print(rnn_hidden.var(dim=0).mean())  # should not be tiny
        forward_out = decoder_hidden_states[:, :, :self.gru_hidden_dim]
        backward_out = decoder_hidden_states[:, :, self.gru_hidden_dim:]

        decoder_hidden_states = forward_out + backward_out
        #rnn_output_out = torch.cat((forward_out, backward_out), dim=2)
        context, attn = self.attention(encoder_hidden_states,decoder_hidden_states)

        rnn_output = torch.concat((decoder_hidden_states,context),dim=-1) #[n_nodes,L,feat_dim]

        rnn_output = self.layernorm1(rnn_output) #worsens training, not necessary, since all the sequences an in the same "scale"
        rnn_output = self.layernorm2(self.fc1(rnn_output))
        rnn_output = self.layernorm3(self.linear_probs(rnn_output))
        output_logits = self.logsoftmax(rnn_output)  # [n_nodes,align_seq_len,aa_probs]

        return output_logits
class RNNDecoder_Tiling_Angles(nn.Module):
    def __init__(self, align_seq_len,aa_prob,gru_hidden_dim, z_dim,input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNDecoder_Tiling_Angles, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.input_size = input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = 4
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
        self.fc2_probs = nn.Linear(self.gru_hidden_dim, self.aa_prob)
        self.fc2_means = nn.Linear(self.gru_hidden_dim, 2)
        self.fc2_kappas = nn.Linear(self.gru_hidden_dim, 2)
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)

    def forward(self, input, hidden):

        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        #forward_out = rnn_output[:, :, :self.gru_hidden_dim]
        #backward_out = rnn_output[:, :, self.gru_hidden_dim:]
        #rnn_output_out = torch.cat((forward_out, backward_out), dim=2)
        output = self.fc1(rnn_output)
        output_logits = self.logsoftmax(self.fc2_probs((output)))  # [n_nodes,align_seq_len,aa_probs]
        output_means = self.tanh(self.fc2_means(output))*math.pi
        output_kappas = self.kappa_addition + self.softplus(self.fc2_kappas(output))
        return output_logits,output_means,output_kappas
class RNNDecoder_Tiling_AnglesComplex(nn.Module):
    def __init__(self, align_seq_len,aa_prob,gru_hidden_dim, z_dim,input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNDecoder_Tiling_AnglesComplex, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.input_size = input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        self.fc1 = nn.Linear(2 * self.gru_hidden_dim, self.gru_hidden_dim)
        self.fc1_probs = nn.Linear(self.gru_hidden_dim,int((self.gru_hidden_dim)//2))
        self.fc1_means = nn.Linear(self.gru_hidden_dim, int((self.gru_hidden_dim) // 2))
        self.fc1_kappas = nn.Linear(self.gru_hidden_dim, int((self.gru_hidden_dim) // 2))
        self.fc2_probs = nn.Linear(int((self.gru_hidden_dim) // 2), self.aa_prob)
        self.fc2_means = nn.Linear(int((self.gru_hidden_dim) // 2), 2)
        self.fc2_kappas = nn.Linear(int((self.gru_hidden_dim) // 2), 2)
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=self.num_layers,
                          dropout=0.0)

    def forward(self, input, hidden):

        rnn_output, rnn_hidden = self.rnn(input, hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim]
        #forward_out = rnn_output[:, :, :self.gru_hidden_dim]
        #backward_out = rnn_output[:, :, self.gru_hidden_dim:]
        #rnn_output_out = torch.cat((forward_out, backward_out), dim=2)
        output = self.fc1(rnn_output)
        output_logits = self.logsoftmax(self.fc2_probs(self.fc1_probs(output)))  # [n_nodes,align_seq_len,aa_probs]
        output_means = self.tanh(self.fc2_means(self.fc1_means(output)))*math.pi
        output_kappas = self.kappa_addition + self.softplus(self.fc2_kappas(self.fc1_kappas(output)))
        return output_logits,output_means,output_kappas
class RNNDecoder_TeacherForcing(nn.Module): #Highlight: faster learning for some reason
    def __init__(self, align_seq_len,aa_prob,gru_hidden_dim, z_dim,input_size, kappa_addition,num_layers,pretrained_params):
        super(RNNDecoder_TeacherForcing, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.z_dim = z_dim
        self.input_size = input_size
        self.align_seq_len = align_seq_len
        self.aa_prob = aa_prob
        self.num_layers = num_layers
        self.kappa_addition = kappa_addition
        self.softmax = nn.Softmax()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.method = "normal"
        if self.method == "tf":
            self.layernorm = nn.LayerNorm(self.gru_hidden_dim)
            self.fc1 = nn.Linear(self.gru_hidden_dim, self.gru_hidden_dim)
            self.fc2 = nn.Linear(self.gru_hidden_dim,self.input_size)
            self.linear_probs = nn.Linear(self.input_size, self.aa_prob)
        else:
            #self.layernorm = nn.LayerNorm(self.gru_hidden_dim)

            self.fc1 = nn.Linear( self.gru_hidden_dim, self.input_size ) #+ 2*self.gru_hidden_dim
            self.fc2 = nn.Linear(self.input_size , self.gru_hidden_dim)
            self.linear_probs = nn.Linear(self.gru_hidden_dim, self.aa_prob)
        #self.fc_hidden = nn.Linear(2*self.input_size,2*self.input_size)


        self.rnn = nn.GRU(input_size=self.input_size, # + 2*self.gru_hidden_dim,
                          hidden_size=self.gru_hidden_dim,
                          batch_first=True,
                          bidirectional=False,
                          num_layers=self.num_layers)


    def forward_teacher_forcing(self, input, hidden): #todo: delete, apparently, teacher forcing is only used during sampling, not training

        """
        :param hidden: encoder hidden states (except in the first round)

        :return

        output_logits: [N,L,feat_dim], where feat_dim == aa_probs
        """
        #hidden = torch.tanh(self.fc_hidden(hidden)) # need to scale the encoder hidden state

        if hidden.ndim < 3:
            hidden = hidden.unsqueeze(0)
        token = input[:,0].unsqueeze(1)
        Bs, L = input.shape[0], input.shape[1]
        output_logits = torch.zeros((Bs,L,self.aa_prob))
        for t in range(L):
            rnn_out_t, rnn_hidden = self.rnn(token, hidden)  # [n_seqs,1, 2* gru_dim] | [n_directions, n_seqs, gru_dim]
            rnn_out_t = self.fc2(self.fc1(rnn_out_t))
            rnn_out_t = self.layernorm(rnn_out_t)
            output_logit_t = self.logsoftmax(self.linear_probs(rnn_out_t))
            output_logits[:,t] = output_logit_t.squeeze(1)
            if t+1 < L:
                token = input[:, t + 1].unsqueeze(1)  # todo: try teacher forcing
                #token = input[:,t].unsqueeze(1) if torch.rand(1) < self.teacher_forcing_ratio else rnn_out_t
        #output_logits = self.layernorm(output_logits) # [n_nodes,align_seq_len,aa_probs]
        return output_logits

    def forward_normal(self,input,hidden,mode="train"):
        """
        :input: embeddings, not direct tokens
        """

        if mode == "train":
            #hidden = torch.tanh(self.fc_hidden(hidden))

            encoder_hidden_states, decoder_hidden = hidden["encoder_hidden_states"], hidden["decoder"]
            #input = torch.cat([encoder_hidden_states,input],dim=-1)
            rnn_output, rnn_hidden = self.rnn(input, decoder_hidden)  # [n_nodes,align_seq_len,gru_dim] | [1,n_nodes,gru_dim]
            #rnn_output = self.layernorm(rnn_output) #worsens training, not necessary, since all the sequences an in the same "scale"
            rnn_output = self.fc1(rnn_output)
            output_logits = self.logsoftmax(self.linear_probs(self.fc2(rnn_output)))  # [n_nodes,align_seq_len,aa_probs]

        elif mode == "sample":
            encoder_hidden_states, decoder_hidden = hidden["encoder_hidden_states"], hidden["decoder"]
            input = torch.cat([encoder_hidden_states, input], dim=-1)
            Bs, L, feat_dim = input.shape

            #input_t = input[:, 0].unsqueeze(1) # "first token"
            input_t = torch.zeros([Bs,1, feat_dim]) # "start token"

            output_logits = torch.zeros((Bs, L, self.aa_prob))
            for t in range(L):
                rnn_output_t, rnn_hidden_t = self.rnn(input_t, decoder_hidden)
                # #rnn_output_t = self.layernorm(rnn_output_t)
                rnn_output_t = self.fc1(rnn_output_t)

                if 0 < t  < L :
                    input_t = rnn_output_t + input[:,t].unsqueeze(1)# just the rnn_output_t is not working

                else:
                    input_t = rnn_output_t
                output_logit_t = self.logsoftmax(self.linear_probs(self.fc2(rnn_output_t)))
                output_logits[:, t] = output_logit_t.squeeze(1)



            #output_logits = output_logits[:,1:]



        return output_logits

    def forward(self,input,hidden,mode="train"):
        if self.method == "tf":
            return self.forward_teacher_forcing(input,hidden)

        else:
            return self.forward_normal(input,hidden,mode)
class xLSTMDecoder(nn.Module):
    def __init__(self,max_len,input_size,z_dim,output_size):
        super(xLSTMDecoder, self).__init__()
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
        self.output_size = output_size
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
                    round_proj_up_to_multiple_of=90,  # inner_embedding_dim, must be divisible by num_heads
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
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.input_size, self.input_size)
        self.linear_probs = nn.Linear(self.input_size, self.output_size)

    def forward(self,input):

        embedding = self.xlstm_stack(input)
        #
        output_logits = self.logsoftmax(self.relu(self.linear_probs(self.fc1(embedding)))) #todo: layernorm

        return output_logits
class TransformerDecoder(nn.Module):
    def __init__(self,input_dim_l, input_dim_r,align_seq_len,hidden_dim,output_dim):
        super(TransformerDecoder, self).__init__()
        self.num_heads = 1
        self.num_decoder_layers = 1
        self.input_dim_l = input_dim_l
        self.input_dim_r = input_dim_r
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.align_seq_len = align_seq_len
        self.positional_embeddings = PositionalEncodings(self.align_seq_len, self.hidden_dim, base=1000,
                                                         type="sinusoidal")
        self.input_fc = nn.Linear(self.input_dim_l,self.hidden_dim)
        self.latent_fc = nn.Linear(self.input_dim_r,self.hidden_dim)
        self.layernorm = nn.LayerNorm(self.hidden_dim)  # todo: not sure yet
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=self.num_heads),
            num_layers= self.num_decoder_layers
        )

        self.output_fc = nn.Linear(self.hidden_dim,self.output_dim)
        self.logsoftmax = nn.LogSoftmax()
    def forward(self,input,latent):
        """
        :param torch.tensor tgt_seq : [nseq,L,featdim]
        :param torch.tensor latent: [nseq,L,zdim]
        """
        latent = self.latent_fc(latent)
        latent = self.positional_embeddings(latent)
        latent = self.layernorm(latent)
        input = self.input_fc(input)
        input = self.positional_embeddings(input)
        input = self.layernorm(input)



        decoded = self.decoder(tgt=input, memory=latent)
        logits = self.logsoftmax(self.output_fc(decoded))

        return logits
class GlobalLowRankPottsDecoder(SequenceDecoderBase):
    """Draupnir2-style Potts decoder with one fixed design.

    Design summary
    --------------
    - latent vector z_k controls the sitewise fields,
    - pairwise couplings come from a global learnable tensor U,
    - decoding uses coordinate ascent,
    - Gibbs sampling is available through the Potts distribution object.
    """

    def __init__(
        self,
        latent_dim: int,
        seq_len: int,
        alphabet_size: int,
        potts_rank: int = 8,
        hidden_dim: int = 64,
        site_embedding_dim: int = 16,
        interaction_rank: int = 8,
        factor_scale: float = 0.08,
        center_factors: bool = True,
    ) -> None:
        super().__init__()
        assert latent_dim >= 1, "latent_dim must be positive"
        assert seq_len >= 1, "seq_len must be positive"
        assert alphabet_size >= 2, "alphabet_size must be at least 2"
        assert potts_rank >= 1, "potts_rank must be positive"
        assert hidden_dim >= 1, "hidden_dim must be positive"
        assert site_embedding_dim >= 1, "site_embedding_dim must be positive"
        assert interaction_rank >= 1, "interaction_rank must be positive"
        assert factor_scale > 0.0, "factor_scale must be positive"

        self.latent_dim = int(latent_dim)
        self.seq_len = int(seq_len)
        self.alphabet_size = int(alphabet_size)
        self.potts_rank = int(potts_rank)
        self.hidden_dim = int(hidden_dim)
        self.factor_scale = float(factor_scale)
        self.center_factors = bool(center_factors)

        # Site embeddings provide position-specific information.
        self.site_embedding = nn.Embedding(self.seq_len, site_embedding_dim)

        # These layers build the hidden representation h_{k,i}.
        self.z_to_hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.site_to_hidden = nn.Linear(site_embedding_dim, self.hidden_dim)
        self.hidden_refine = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hidden_to_fields = nn.Linear(self.hidden_dim, self.alphabet_size)

        # Interaction between latent and site features.
        self.z_to_interaction = nn.Linear(self.latent_dim, interaction_rank)
        self.site_to_interaction = nn.Linear(site_embedding_dim, interaction_rank)
        self.interaction_to_hidden = nn.Linear(interaction_rank, self.hidden_dim)

        self.site_hidden_bias = nn.Parameter(torch.zeros(self.seq_len, self.hidden_dim))
        self.site_field_bias = nn.Parameter(torch.zeros(self.seq_len, self.alphabet_size))
        self.global_field_bias = nn.Parameter(torch.zeros(self.alphabet_size))

        # Global low-rank factors U[i, a, :] shared across all sequences.
        self.global_factors = nn.Parameter(torch.zeros(self.seq_len, self.alphabet_size, self.potts_rank))

    def hidden_state(self, z: torch.Tensor) -> torch.Tensor:
        """Construct the hidden site representation h_{k,i}."""
        assert isinstance(z, torch.Tensor), "z must be a tensor"
        assert z.ndim == 2, "z must have shape [batch, latent_dim]"
        assert z.shape[1] == self.latent_dim, "wrong latent dimension"

        batch_size = z.shape[0]

        site_ids = torch.arange(self.seq_len, device=z.device, dtype=torch.long)
        site_embeddings = self.site_embedding(site_ids)

        z_hidden = self.z_to_hidden(z).unsqueeze(1).expand(batch_size, self.seq_len, self.hidden_dim)
        site_hidden = self.site_to_hidden(site_embeddings).unsqueeze(0).expand(batch_size, self.seq_len, self.hidden_dim)

        hidden = z_hidden + site_hidden + self.site_hidden_bias.unsqueeze(0)

        z_interaction = self.z_to_interaction(z).unsqueeze(1).expand(batch_size, self.seq_len, self.z_to_interaction.out_features)
        site_interaction = self.site_to_interaction(site_embeddings).unsqueeze(0).expand(batch_size, self.seq_len, self.site_to_interaction.out_features)
        interaction = z_interaction * site_interaction

        hidden = hidden + self.interaction_to_hidden(interaction)
        hidden = torch.tanh(hidden)
        hidden = torch.tanh(self.hidden_refine(hidden))

        assert hidden.shape == (batch_size, self.seq_len, self.hidden_dim)
        return hidden

    def fields_and_factors(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Potts fields and global low-rank factors.

        Fields depend on z.
        Factors are global parameters repeated across the batch.
        """
        hidden = self.hidden_state(z)
        batch_size = z.shape[0]

        fields = self.hidden_to_fields(hidden)
        fields = fields + self.site_field_bias.unsqueeze(0)
        fields = fields + self.global_field_bias.view(1, 1, self.alphabet_size)

        factors = self.factor_scale * self.global_factors.unsqueeze(0).expand(batch_size, -1, -1, -1)

        assert fields.shape == (batch_size, self.seq_len, self.alphabet_size)
        assert factors.shape == (batch_size, self.seq_len, self.alphabet_size, self.potts_rank)
        return fields, factors

    def distribution(self, z: torch.Tensor) -> LowRankPottsPseudoLikelihood:
        """Wrap decoder parameters in a Potts pseudo-likelihood object."""
        fields, factors = self.fields_and_factors(z)
        return LowRankPottsPseudoLikelihood(
            fields=fields,
            factors=factors,
            center_factors=self.center_factors,
        )

    def reconstruction_log_prob(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return the Potts pseudo-log-likelihood."""
        assert isinstance(x, torch.Tensor), "x must be a tensor"
        assert x.ndim == 2, "x must have shape [batch, seq_len]"
        assert x.shape[1] == self.seq_len, "wrong sequence length"

        distribution = self.distribution(z)
        log_prob = distribution.log_prob(x.long())

        assert log_prob.shape == (x.shape[0],)
        return log_prob

    def decode_sequences(self, z: torch.Tensor, num_sweeps: int = 40, initial_x: torch.Tensor | None = None) -> torch.Tensor:
        """Decode a Potts sequence by deterministic coordinate ascent.

        The default initialization is the sitewise field argmax.
        """
        assert num_sweeps >= 1, "num_sweeps must be positive"

        distribution = self.distribution(z)
        fields, _ = self.fields_and_factors(z)

        if initial_x is None:
            x = torch.argmax(fields, dim=-1)
        else:
            assert isinstance(initial_x, torch.Tensor), "initial_x must be a tensor"
            assert initial_x.ndim == 2, "initial_x must have shape [batch, seq_len]"
            assert initial_x.shape == (z.shape[0], self.seq_len), "initial_x has wrong shape"
            x = initial_x.clone().long()

        for _ in range(num_sweeps):
            changed = 0
            for site in range(self.seq_len):
                logits = distribution.conditional_logits(x)
                new_state = torch.argmax(logits[:, site, :], dim=-1)
                changed = changed + int((new_state != x[:, site]).sum().item())
                x[:, site] = new_state

            if changed == 0:
                break

        assert x.shape == (z.shape[0], self.seq_len)
        return x


