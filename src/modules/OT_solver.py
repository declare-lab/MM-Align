import torch
import torch.nn as nn

class OTSolver(nn.Module):
    def __init__(self, lam=0.2, maxIter=10, eps=1e-7, win_size=2, method='cosine', cuda=True):
        super().__init__()
        self.lam = lam
        self.maxIter = maxIter
        self.epsilon = eps
        self.win_size = win_size
        self.method = method
        self.iscuda = cuda
    
    def _compute_optimal_transport_batch(self, m1, m2, length):
        """ 
        Compute the optimal transport for a batch
        Inputs:
            - m1 : modality 1 (L, B, 1)
            - m2 : modality 2 (L, B, 1)
        """
        res = []
        costs = []
        for i, l in enumerate(length):
            mm1, mm2 = m1[:l,i], m2[:l,i] # (L, D), (L, D)
            M = self.pairwise_distances(mm1, mm2)
            r, cost = self.compute_optimal_transport(M)
            res.append(r)
            costs.append(cost)
        return res, costs

    def forward(self, m1, m2, lengths):
        return self._compute_optimal_transport_batch(m1, m2, lengths)
        
    def compute_optimal_transport(self, M):
        """
        Computes the optimal transport matrix and Slinkhorn distance using the
        Sinkhorn-Knopp algorithm
        Inputs:
            - M : cost matrix (n x m)
            - r : vector of marginals (n, )
            - c : vector of marginals (m, )
            - lam : strength of the entropic regularization
            - epsilon : convergence parameter
        Outputs:
            - P : optimal transport matrix (n x m)
            - dist : Sinkhorn distance
        Reference:
            https://github.com/rythei/PyTorchOT/blob/master/ot_pytorch.py#L11-L59
        """
        lam = self.lam
        maxIter = self.maxIter
        epsilon = self.epsilon
        
        n, m = M.shape
        r = torch.ones((n,)) / n  # (n,)
        c = torch.ones((m,)) / m  # (m,)
        if self.iscuda:
            r, c = r.cuda(), c.cuda()

        P = torch.exp(- lam * M)    # (n, m)
        P /= P.sum()
        u = torch.zeros(n)
        if self.iscuda:
            u = u.cuda()
        
        num_iter = 0

        while torch.max(torch.abs(u - P.sum(1))) > epsilon and num_iter < maxIter:
            u = P.sum(1)
            P *= (r / u).reshape((-1, 1))
            P *= (c / P.sum(0)).reshape((-1, 1))

            num_iter += 1

        return P / P.sum(-1), torch.sum(P*M)

    def pairwise_distances(self, x, y):
        n = x.size()[0]
        m = y.size()[0]
        d = x.size()[1]

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        
        win_size = self.win_size
        method = self.method
        if method == 'l1':
            dist = torch.abs(x - y).sum(2)
        elif method == 'l2':
            dist = torch.pow(x - y, 2).sum(2)
        elif method == 'cosine':
            cos = nn.CosineSimilarity(dim=2, eps=1e-5)
            dist = 1 - cos(x, y)
        
        # add barrier function to the cost matrix
        if win_size > 0:
            c = torch.Tensor([100.0]).expand_as(dist)
            cm1 = torch.triu(c, diagonal=win_size+1)
            cm2 = torch.tril(c, diagonal=-(win_size+1))
            dist += (cm1 + cm2)

        return dist.float()

def main():
    x, y = torch.randn(5, 4), torch.randn(5, 4)
    solver = OTSolver(win_size=1)
    dist = solver.pairwise_distances(x.abs(), y.abs())
    trans, cost = solver.compute_optimal_transport(dist.cuda())
    print('distance matrx:', dist)
    print('transport plan:', trans)
    print('total cost for this plan:', cost)

if __name__ == '__main__':
    main()