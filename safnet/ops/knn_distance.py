import torch
try:
	from . import knn_distance_cuda
except:
	import knn_distance_cuda
import pdb


class KNNDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_xyz, key_xyz, k):
        index, distance = knn_distance_cuda.knn_distance(query_xyz, key_xyz, k)
        return index, distance

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) * len(grad_outputs)


def knn_distance(query, key, k, transpose=True):
    """For each point in query set, find its distances to k nearest neighbors in key set.

    Args:
        query: (B, 3, N1), xyz of the query points.
        key: (B, 3, N2), xyz of the key points.
        k (int): K nearest neighbor
        transpose (bool): whether to transpose xyz

    Returns:
        index: (B, N1, K), indices of these neighbors in the key.
        distance: (B, N1, K), distance to the k nearest neighbors in the key.

    """
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    query = query.contiguous()
    key = key.contiguous()
    # pdb.set_trace()
    index, distance = KNNDistanceFunction.apply(query, key, k)
    return index, distance

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """

    b,c,n,k = xyz.shape
    xyz = xyz.permute(0,2,3,1)
    xyz = xyz.reshape(-1,k,c)
    new_xyz = new_xyz.permute(0,2,3,1)
    new_xyz = new_xyz.reshape(-1,k,c)

    sqrdists = square_distance(new_xyz, xyz)
    distance, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    distance = distance.reshape(b,n,k,nsample)
    return distance,group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm?
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

if __name__=="__main__":
    import torch
    a = torch.randn(1,3,100).cuda().float()
    b = torch.randn(1,3,100).cuda().float()
    _,distance = knn_distance(a,b,3)
    pdb.set_trace()
    print(distance.shape)

