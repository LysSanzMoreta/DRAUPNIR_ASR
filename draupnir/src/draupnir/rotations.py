import  torch
def batch_multiplication(a,b,n_data,L,feat_dim):
    """Inspired by https://github.com/pytorch/pytorch/issues/3172"""
    c = torch.bmm(a[:,:,:,None].view(n_data,-1,1),b[:,None,:,:].view(n_data,1,-1))
    c = c.view(n_data,L,feat_dim,L,feat_dim)
    c = c.permute(0,1,3,2,4)[:,torch.arange(L),torch.arange(L)]

    return c

def rotate_blosum_batch(data,data_mask,degree_rotation):
    """

    :return:

    Notes:
        -https://www.rollpie.com/post/311
        -https://math.stackexchange.com/questions/2144153/n-dimensional-rotation-matrix
        -https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm
        -https://math.stackexchange.com/questions/209768/transformation-matrix-to-go-from-one-vector-to-another
    """
    n_data,L,feat_dim = data.shape

    # input vectors
    v2 = torch.ones_like(data) #[N,L,feat_dim]

    # Gram-Schmidt orthogonalization

    n1 = data/torch.linalg.norm(data,dim=2)[:,:,None] #[N,L,feat_dim]
    v2 = v2 - torch.matmul(n1,v2[0,0])[:,:,None]*n1 #works [N,L,feat_dim]
    n2 = v2 / torch.linalg.norm(v2,dim=2)[:,:,None]

    # rotation by pi/2 (np.pi = 180)
    sign = torch.randn(1) > 0
    #sign = torch.Tensor([True])
    degree = torch.rand(1)  #A degree 0 will not rotate the vector

    sign_dict ={True:torch.tensor([-1]),False:torch.tensor([1])}
    a = sign_dict[sign.item()]*(torch.pi*degree)
    #a = torch.rand(-1,1,(1))*torch.pi #degrees
    I = torch.eye(feat_dim)


    one = batch_multiplication(n2,n1,n_data,L,feat_dim)
    two = batch_multiplication(n1,n2,n_data,L,feat_dim)
    three = batch_multiplication(n1,n1,n_data,L,feat_dim)
    four = batch_multiplication(n2,n2,n_data,L,feat_dim)

    R = I + (one - two) * torch.sin(a) + (three + four) * (torch.cos(a) - 1)

    # check result
    data_rotated = torch.matmul(R,n1[:,:,:,None]).squeeze(-1)
    data_rotated_unnormalized = data_rotated*torch.linalg.norm(data,dim=2)[:,:,None]
    data_mask = torch.tile(data_mask[:,:,None],(1,1,data.shape[-1]))

    data[~data_mask] = 0
    data_rotated_unnormalized[data_mask] = 0
    data_transformed = data + data_rotated_unnormalized

    return data_transformed