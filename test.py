import unittest

import numpy as np
import torch
import torch.nn as nn

import marlin_reproduction


seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device('cuda:0')


def gen_quant4(m, n, groupsize=-1, use_identity=False):
    tile = 16
    maxq = 2 ** 4 - 1
    
    if use_identity:
        w = torch.eye(m, n, dtype=torch.half, device=DEV)
    else:
        w = torch.randn((m, n), dtype=torch.half, device=DEV)
        
    # if groupsize != -1:
    #     w = w.reshape((-1, groupsize, n))
    #     w = w.permute(1, 0, 2)
        # w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    # print("s", s)
    s *= 2 / maxq
    # print("s", s)   
    w = torch.round(w / s).int()
    # print("w", w)
    w += (maxq + 1) // 2
    # print("w", w)
    w = torch.clamp(w, 0, maxq)
    # print("w", w)
    ref = (w - (maxq + 1) // 2).half() * s
    # if groupsize != -1:
    #     def reshape(w):
    #         w = w.reshape((groupsize, -1, n))
    #         w = w.permute(1, 0, 2)
    #         w = w.reshape((m, n)).contiguous()
    #         return w
    #     ref = reshape(ref)
    #     w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    
    layer = marlin_reproduction.Layer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    q = layer.B
    s = layer.s
    return ref, q, s

class Test(unittest.TestCase):

    def run_problem(self, m, n, k, thread_k, thread_n, groupsize=-1):
        print('% 5d % 6d % 6d % 4d % 4d % 4d' % (m, n, k, thread_k, thread_n, groupsize))
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        B_ref, B, s = gen_quant4(k, n, groupsize=groupsize, use_identity=True)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        C_ref = torch.matmul(A, B_ref)
        workspace = torch.zeros(n // 128 * 16, device=DEV)
        marlin_reproduction.mul(A, B, C, s, workspace, thread_k, thread_n, -1)
        torch.cuda.synchronize()
        print('C')
        for row in C[:5]: 
            print(' '.join(f'{x:.2f}' for x in row / 7))
        print('C_ref')
        for row in C_ref[:5]: 
            print(row)
        # print('C', C)
        self.assertLess(torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)), 0.001)

    def run_problem_idx(self, m, n, k, thread_k, thread_n, groupsize=-1): 
        print('% 5d % 6d % 6d % 4d % 4d % 4d' % (m, n, k, thread_k, thread_n, groupsize))
        A = torch.arange(m*k, dtype=torch.int32, device=DEV).reshape(m,k).to(torch.float32) / 100
        A = A.half()
        # Check for inf values in A
        if torch.isinf(A).any():
            print("Warning: A contains inf values")
            print("Inf locations:", torch.nonzero(torch.isinf(A)))
        # for row in A[-5:]: 
        #     for elem in row:
        #         print(f"{elem:.2f}", end=" ")
        #     print()
        B_ref, B, s = gen_quant4(k, n, groupsize=groupsize, use_identity=True)
        if torch.isinf(B).any():
            print("Warning: B contains inf values")
            print("Inf locations:", torch.nonzero(torch.isinf(B)))
        print("s", s)
        # print('B', B)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        C_ref = torch.matmul(A, B_ref)
        workspace = torch.zeros(n // 128 * 16, device=DEV)
        marlin_reproduction.mul(A, B, C, s, workspace, thread_k, thread_n, -1)
        torch.cuda.synchronize()

        print('C')
        for row in C[:5]: 
            print(' '.join(f'{x:.2f}' for x in row / 7))
            print()
        print('C_ref')
        for row in C_ref[:5]: 
            print(row)
        self.assertLess(torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)), 0.001)
    

    def test_tiles(self):
        print()
        # for m in [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 118, 128, 152, 768, 1024]:
        for m in [128]:
            for thread_k, thread_n in [(64, 256)]:
                if m > 16 and thread_k == 128:
                    continue
                self.run_problem_idx(m, 512, 1024, thread_k, thread_n)

    def test_k_stages_divisibility(self):
        print()
        return
        for k in [3 * 64 + 64 * 4 * 2 + 64 * i for i in range(1, 4)]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_very_few_stages(self):
        print()
        return
        for k in [64, 128, 192]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_llama_shapes(self):
        print()
        return
        MODELS = {
            ' 7B': [
                (4096, 3 * 4096),
                (4096, 4096),
                (4096, 2 * 10752),
                (10752, 4096)
            ],
            '13B': [
                (5120, 3 * 5120),
                (5120, 5120),
                (5120, 2 * 13568),
                (13568, 5120)
            ],
            '33B': [
                (6656, 3 * 6656),
                (6656, 6656),
                (6656, 2 * 17664),
                (17664, 6656)
            ],
            '70B': [
                (8192, 3 * 8192),
                (8192, 8192),
                (8192, 2 * 21760),
                (21760, 8192)
            ]
        }
        for _, layers in MODELS.items():
            for layer in layers:
                for thread_k, thread_n in [(128, 128)]:
                    for batch in [1, 16]:
                        self.run_problem(batch, layer[1], layer[0], thread_k, thread_n)

    def test_errors(self):
        print()
        return
        m, n, k = 16, 256, 64
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        B_ref, B, s = gen_quant4(k, n)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        workspace = torch.zeros(n // 128, device=DEV)
        err = False
        try:
            marlin_reproduction.mul(A, B, C, s, workspace, 128, 128, -1)
        except:
            err = True 
        self.assertTrue(err)
        err = False
        try:
            marlin_reproduction.mul(A, B, C, s, workspace, 256, 256, -1)
        except:
            err = True 
        self.assertTrue(err)
        s = torch.zeros((2, n), dtype=torch.half, device=DEV)
        err = False
        try:
            marlin_reproduction.mul(A, B, C, s, workspace, 256, 256, -1)
        except:
            err = True 
        self.assertTrue(err)

    def test_groups(self):
        print()
        return
        for m in [16]:
            for groupsize in [128]:
                for n, k in [(256, 512), (256, 1024), (256 * 128, 1024)]:
                    for thread_shape in [(128, 128), (64, 256)]:
                        self.run_problem(m, n, k, *thread_shape, groupsize)


if __name__ == '__main__':
    unittest.main()
