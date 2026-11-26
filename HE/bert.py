"""Attention and FFN linear modules for the ciphertext encoder.

Softmax, GeLU and LayerNorm are attached by the chain. Score and context
use plaintext complement masks.
"""
import numpy as np
import torch
from liberate.fhe.data_struct import DataStruct

from linear import ThorLinearEvaluator
        
class ThorModule:
    def __init__(self) -> None:
        self.weights = {}
        self.devices = []
        
    def to(self, devices=[0]):
        print(f"Memory allocated to GPU: {torch.cuda.memory_allocated(devices[0]) /1024**3} before layer allocation")

        for key in self.keys:
            for index in np.ndindex(self.weights[key].shape):
                if isinstance(self.weights[key][index], torch.Tensor):
                    encoded = [self.weights[key][index].to(f"cuda:{devices[0]}")]
                    self.weights[key][index] = encoded
        self.devices = devices
                    
    def cpu(self):
        for key in self.keys:
            for index in np.ndindex(self.weights[key].shape):
                if isinstance(self.weights[key][index], list):
                    self.weights[key][index] = self.weights[key][index][0].cpu()          
        self.devices = []

class ThorBertAttention(ThorModule):
    def __init__(self, evaluator:ThorLinearEvaluator, weights: dict[np.ndarray[DataStruct]], layer_idx: int):
        self.evaluator = evaluator
        self.engine = self.evaluator.engine
        self.layer_idx = layer_idx
        self.weights = {}
        for qkv in ['query', 'key', 'value']:
            self.weights[f"{qkv}.weight"] = weights[f'bert.encoder.layer.{layer_idx}.attention.self.{qkv}.weight']
            self.weights[f"{qkv}.bias"] = weights[f'bert.encoder.layer.{layer_idx}.attention.self.{qkv}.bias']
        self.weights['dense.weight'] = weights[f'bert.encoder.layer.{layer_idx}.attention.output.dense.weight']
        self.weights['dense.bias'] = weights[f'bert.encoder.layer.{layer_idx}.attention.output.dense.bias']
        self.weights['LayerNorm.weight'] = weights[f'bert.encoder.layer.{layer_idx}.attention.output.LayerNorm.weight']
        self.weights['LayerNorm.bias'] = weights[f'bert.encoder.layer.{layer_idx}.attention.output.LayerNorm.bias']
        
        self.keys = ['query.weight', 'query.bias', 
                         'key.weight', 'key.bias', 
                         'value.weight', 'value.bias', 
                         'dense.weight', 'dense.bias',
                         'LayerNorm.weight', 'LayerNorm.bias']

        self.devices = []
        
    def query(self, x: np.ndarray[DataStruct]):
        """"
        Calculates the query layer of the self attention
        @param x: Complexified lower diagonals of the input, ciphertext array of shape (64,)
        @return: Lower diagonals of the query layer, ciphertext array of shape (4,)
        """
        if x.shape != (64,):
            raise ValueError("Input of self attention should be (64,)")
        wx = self.evaluator.pt_ct_matmul(self.weights['query.weight'], x)
        if wx.shape[0] != 4:
            raise ValueError("Shape of wx should be (4,)")
        
        q = np.full((4,), None, dtype=DataStruct)
        for i in range(4):
            q[i] = self.engine.pc_add(self.weights['query.bias'][i], wx[i])
            q[i] = self.engine.cc_add(q[i], self.engine.conjugate(q[i]))
        return q
        
    def key(self, x: np.ndarray[DataStruct]):
        """
        Calculates the key layer of the self attention
        @param x: Complexified lower diagonals of the input, ciphertext array of shape (64,)
        @return: Lower diagonals of the key layer, ciphertext array of shape (4,)
        """
        if x.shape != (64,):
            raise ValueError("Input of self attention should be (64,)")
        wx = self.evaluator.pt_ct_matmul(self.weights['key.weight'], x)
        if wx.shape[0] != 4:
            raise ValueError("Shape of wx should be (4,)")
        for i in range(4):
            wx[i] = self.engine.pc_add(self.weights['key.bias'][i], wx[i])
            wx[i] = self.engine.cc_add(wx[i], self.engine.conjugate(wx[i]))
        return wx
    
    def value(self, x: np.ndarray[DataStruct]):
        """
        Calculates the value layer of the self attention
        @param x: Complexified lower diagonals of the input, ciphertext array of shape (64,)
        @return: Lower diagonals of the value layer, ciphertext array of shape (4,)
        """
        if x.shape != (64,):
            raise ValueError("Input of self attention should be (64,)")
        wx = self.evaluator.pt_ct_matmul(self.weights['value.weight'], x)
        if wx.shape[0] != 4:
            raise ValueError("Shape of wx should be (4,)")
        for i in range(4):
            wx[i] = self.engine.pc_add(self.weights['value.bias'][i], wx[i])
            wx[i] = self.engine.cc_add(wx[i], self.engine.conjugate(wx[i]))
        return wx
    
    def calculate_attention_score(self, k_cplx: np.ndarray[DataStruct], q_copies: np.ndarray[DataStruct], scale=1, bootstrap=True, rescale=False, debug=False):
        """
        Calculates the attention score(qk^T)
        @param k_cplx: key, ciphertext array of shape (4,) [ [l0 +il64| l1 | ...l15+il79], ...[l48+il112| l49+il113|...l63+il127] ]
        @param q_copies: query, ciphertext array of shape (64,) [ [l0 | l0| ...], [l1...] ... [l64...] ]
        @return: output, ciphertext array of shape (8,)
        """
        if k_cplx.shape != (4,) or q_copies.shape != (64,):
            raise ValueError("Shape of k_cplx should be (4,) and q_copies should be (64,)")
        
        if k_cplx[0].level_calc != q_copies[0].level_calc:
            if k_cplx[0].level_calc > q_copies[0].level_calc:
                for i in range(len(q_copies)):
                    q_copies[i]=self.engine.level_up(q_copies[i],k_cplx[0].level_calc)
            else:
                for i in range(len(k_cplx)):
                    k_cplx[i]=self.engine.level_up(k_cplx[i],q_copies[0].level_calc)
            # raise ValueError("Levels of k_cplx and q_copies should be the same")
            
        n_in = 64
        n_out = 4
        lev = k_cplx[0].level_calc #Level: l
        
        ttemp = np.full((n_out,4), None, dtype=DataStruct)
        
        if bootstrap:
            scale = scale / 2
        scales = np.full((2**15,),scale, dtype=float)
        scale_pt = self.engine.encode(scales, level=lev)
        
        masks = self.evaluator.masks['ct_ct_matmul']
        
        #n = 0
        for out in range(n_out):
            temp_ct_ct = self.engine.ct_ct_mult(k_cplx[out], q_copies[0], rescale=rescale, relin=False)
            ttemp[out, 0] = self.engine.pt_ct_mult_extended(scale_pt, temp_ct_ct)                      
            
        for n in range(1, n_in):
            q = (n // 16)
            j = n % 16
            rot = n
            l_prime = q_copies[n]
            temp = np.full((n_out,4), None, dtype=DataStruct)
            rrot = (2**11)*j - 16 * rot
            rrot = rrot%(2**15)

            if j == 0:
                for out in range(n_out):
                    rot_temp = self.engine.rotate_left(k_cplx[out], -rrot, self.engine.rot_keys[(self.engine.num_slots-rrot)])
                    temp_ct_ct = self.engine.ct_ct_mult(rot_temp, l_prime, rescale=rescale, relin=False)

                    for i in range(1):
                        temp[out, i] = self.engine.pt_ct_mult_extended(masks[i][n], temp_ct_ct)
                    temp[out, 1] = self.engine.pt_ct_mult_extended(masks[1][n], temp_ct_ct)
                
                if q == 1:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[3,0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[3,1])
                    
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[0,0])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[0,1])
                    
                    ttemp[2,0] = self.engine.cc_add(ttemp[2,0], temp[1,0])
                    ttemp[2,1] = self.engine.cc_add(ttemp[2,1], temp[1,1])
                    
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[2,0])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[2,1])
                    
                elif q == 2:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[2,0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[2,1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[3,0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[3,1])
                    
                    ttemp[2,0] = self.engine.cc_add(ttemp[2,0], temp[0,0])
                    ttemp[2,1] = self.engine.cc_add(ttemp[2,1], temp[0,1])
                    
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[1,0])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[1,1])
                    
                elif q == 3:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[1,0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[1,1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[2,0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[2,1])
                    
                    ttemp[2,2] = self.engine.cc_add(ttemp[2,2], temp[3,0])
                    ttemp[2,3] = self.engine.cc_add(ttemp[2,3], temp[3,1])
                    
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[0,0])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[0,1])

            else:  
                for out in range(n_out):
                    rot_temp = self.engine.rotate_left(k_cplx[out], -rrot, self.engine.rot_keys[self.engine.num_slots-rrot])
                    temp_ct_ct = self.engine.ct_ct_mult(rot_temp, l_prime, rescale=rescale, relin=False)
                    for i in range(3):
                        temp[out, i] = self.engine.pt_ct_mult_extended(masks[i][n], temp_ct_ct)
                    temp[out, 3] = self.engine.pt_ct_mult_extended(masks[3][n], temp_ct_ct)
                
                if q == 0:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[3, 2])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[3, 3])
                    ttemp[0,0] = self.engine.cc_add(ttemp[0,0], temp[0, 0])
                    ttemp[0,1] = self.engine.cc_add(ttemp[0,1], temp[0, 1])
                    
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[0, 2])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[0, 3])
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[1, 0])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[1, 1])
                    
                    ttemp[2,0] = self.engine.cc_add(ttemp[2,0], temp[1, 2])
                    ttemp[2,1] = self.engine.cc_add(ttemp[2,1], temp[1, 3])
                    ttemp[2,0] = self.engine.cc_add(ttemp[2,0], temp[2, 0])
                    ttemp[2,1] = self.engine.cc_add(ttemp[2,1], temp[2, 1])
                    
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[2, 2])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[2, 3])
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[3, 0])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[3, 1])
                    
                
                elif q == 1:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[2, 2])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[2, 3])
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[3, 0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[3, 1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[3, 2])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[3, 3])
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[0, 0])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[0, 1])
                    
                    ttemp[2,0] = self.engine.cc_add(ttemp[2,0], temp[0, 2])
                    ttemp[2,1] = self.engine.cc_add(ttemp[2,1], temp[0, 3])
                    ttemp[2,0] = self.engine.cc_add(ttemp[2,0], temp[1, 0])
                    ttemp[2,1] = self.engine.cc_add(ttemp[2,1], temp[1, 1])
                    
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[1, 2])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[1, 3])
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[2, 0])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[2, 1])
                    
                elif q == 2:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[1, 2])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[1, 3])
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[2, 0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[2, 1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[2, 2])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[2, 3])
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[3, 0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[3, 1])
                    
                    ttemp[2,2] = self.engine.cc_add(ttemp[2,2], temp[3, 2])
                    ttemp[2,3] = self.engine.cc_add(ttemp[2,3], temp[3, 3])
                    ttemp[2,0] = self.engine.cc_add(ttemp[2,0], temp[0, 0])
                    ttemp[2,1] = self.engine.cc_add(ttemp[2,1], temp[0, 1])
                    
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[0, 2])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[0, 3])
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[1, 0])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[1, 1])
                    
                elif q == 3:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[0, 2])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[0, 3])
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[1, 0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[1, 1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[1, 2])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[1, 3])
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[2, 0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[2, 1])
                    
                    ttemp[2,2] = self.engine.cc_add(ttemp[2,2], temp[2, 2])
                    ttemp[2,3] = self.engine.cc_add(ttemp[2,3], temp[2, 3])
                    ttemp[2,2] = self.engine.cc_add(ttemp[2,2], temp[3, 0])
                    ttemp[2,3] = self.engine.cc_add(ttemp[2,3], temp[3, 1])
                    
                    ttemp[3,2] = self.engine.cc_add(ttemp[3,2], temp[3, 2])
                    ttemp[3,3] = self.engine.cc_add(ttemp[3,3], temp[3, 3])
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[0, 0])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[0, 1])
        
        output = np.full((8,), None, dtype=DataStruct)
        ttemp_relin = np.full((n_out,4), None, dtype=DataStruct)
        output_cplx = np.full((4,), None, dtype=DataStruct)
        output_cplx_scaled = np.full((4,), None, dtype=DataStruct)

        for out in range(n_out):
            for i in range(4):
                ttemp_relin[out, i] = self.engine.relinearize(ttemp[out, i])
            ttemp_relin[out, 1] = self.engine.rotate_left(ttemp_relin[out, 1], -2**11)
            ttemp_relin[out, 3] = self.engine.rotate_left(ttemp_relin[out, 3], -2**11)
            ttemp_relin[out, 2] = self.engine.imult(self.engine.conjugate(self.engine.cc_add(ttemp_relin[out, 2], ttemp_relin[out, 3])))
            output_cplx[out] = (self.engine.cc_add(self.engine.cc_add(ttemp_relin[out, 0], ttemp_relin[out, 1]), ttemp_relin[out, 2]))
            output_cplx_scaled[out] = self.engine.rescale(output_cplx[out]) #Scale: delta^2
            if bootstrap:
                output_cplx_scaled[out] = self.engine.bootstrap(output_cplx_scaled[out])
            conj = self.engine.conjugate(output_cplx_scaled[out])
            output[out] = (self.engine.cc_add(output_cplx_scaled[out], conj))
            output[out+n_out] = (self.engine.imult(self.engine.cc_sub(conj, output_cplx_scaled[out])))     
        return output
    
    def calculate_attention_context(self, l_vt_cplx: np.ndarray[DataStruct], l_at_copies: np.ndarray[DataStruct], scale=1, complex=True, rescale=False):
        """
        @param l_vt_cplx: value, ciphertext array of shape (2,)
        @param l_at_copies: attention prob, ciphertext array of shape (128,)
        """
        lev = l_at_copies[0].level_calc
        if l_vt_cplx[0].level_calc != lev:
            raise ValueError("Level of l_vt_cplx should be the same as l_at_copies")
        
        n_in = 128
        n_out = 2

        ttemp = np.full((n_out,4), None, dtype=DataStruct)
        
        if not complex:
            scale = scale/2
            
        scales = np.full((2**15,),scale, dtype=float)
        scale_pt = self.engine.encode(scales, level=lev)
        masks = self.evaluator.masks['ct_ct_matmul']

    
        for out in range(n_out):            
            temp_ct_ct = self.engine.ct_ct_mult(l_vt_cplx[out], l_at_copies[0], rescale=rescale, relin=False)
            ttemp[out, 0] = self.engine.pt_ct_mult_extended(scale_pt, temp_ct_ct)
            
        for n in range(1, n_in):
            q = (n // 16) % 4  
            j = n % 16
            rot = n
            l_prime = l_at_copies[n]
            
            temp = np.full((n_out,4), None, dtype=DataStruct)

            rrot = (2**11)*j - 16 * rot
            rrot = rrot%(2**15)
            
            if j == 0:
                for out in range(n_out):
                    rot_temp = self.engine.rotate_left(l_vt_cplx[out], -rrot)
                    temp_ct_ct = self.engine.ct_ct_mult(rot_temp, l_prime, rescale=rescale, relin=False)

                    for i in range(1):
                        temp[out, i] = self.engine.pt_ct_mult_extended(masks[i][n], temp_ct_ct)
                    temp[out, 1] = self.engine.pt_ct_mult_extended(masks[1][n], temp_ct_ct)
    
                if q == 1:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[1,0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[1,1])
                    
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[0,0])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[0,1])
                                    
                elif q == 2:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[0,0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[0,1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[1,0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[1,1])
                    
                elif q == 3:
                    ttemp[0,0] = self.engine.cc_add(ttemp[0,0], temp[1,0])
                    ttemp[0,1] = self.engine.cc_add(ttemp[0,1], temp[1,1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[0,0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[0,1])
                    
                elif q == 0:  
                    ttemp[0,0] = self.engine.cc_add(ttemp[0,0], temp[0,0])
                    ttemp[0,1] = self.engine.cc_add(ttemp[0,1], temp[0,1])
                    
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[1,0])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[1,1])


            elif j != 0:            
                
                for out in range(n_out):
                    rot_temp = self.engine.rotate_left(l_vt_cplx[out], -rrot)
                    temp_ct_ct = self.engine.ct_ct_mult(rot_temp, l_prime, rescale=rescale, relin=False)                   
                    for i in range(3):
                        temp[out, i] = self.engine.pt_ct_mult_extended(masks[i][n], temp_ct_ct)
                    temp[out, 3] = self.engine.pt_ct_mult_extended(masks[3][n], temp_ct_ct)
                
                if q == 0:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[1,2])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[1,3])
                    ttemp[0,0] = self.engine.cc_add(ttemp[0,0], temp[0,0])
                    ttemp[0,1] = self.engine.cc_add(ttemp[0,1], temp[0,1])
                    
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[0,2])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[0,3])
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[1,0])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[1,1])

                elif q == 1:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[0,2])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[0,3])
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[1,0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[1,1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[1,2])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[1,3])           
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[0,0])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[0,1])
                    
                elif q == 2:
                    ttemp[0,0] = self.engine.cc_add(ttemp[0,0], temp[1,2])
                    ttemp[0,1] = self.engine.cc_add(ttemp[0,1], temp[1,3])
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[0,0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[0,1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[0,2])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[0,3])
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[1,0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[1,1])
     
                elif q == 3:
                    ttemp[0,0] = self.engine.cc_add(ttemp[0,0], temp[0,2])
                    ttemp[0,1] = self.engine.cc_add(ttemp[0,1], temp[0,3])
                    ttemp[0,0] = self.engine.cc_add(ttemp[0,0], temp[1,0])
                    ttemp[0,1] = self.engine.cc_add(ttemp[0,1], temp[1,1])
                    
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[1,2])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[1,3])
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[0,0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[0,1])
       
        output_cplx = np.full((2,), None, dtype=DataStruct)
        output_cplx_scaled = np.full((2,), None, dtype=DataStruct)
        ttemp_relin = np.full((n_out,4), None, dtype=DataStruct)
        
        for out in range(n_out):
            for i in range(4):
                ttemp_relin[out, i] = self.engine.relinearize(ttemp[out, i])
            ttemp_relin[out, 1] = self.engine.rotate_left(ttemp_relin[out, 1], -2**11)
            ttemp_relin[out, 3] = self.engine.rotate_left(ttemp_relin[out, 3], -2**11)
            ttemp_relin[out, 2] = self.engine.imult(self.engine.conjugate(self.engine.cc_add(ttemp_relin[out, 2], ttemp_relin[out, 3])))
            output_cplx[out] = self.engine.cc_add(self.engine.cc_add(ttemp_relin[out, 0], ttemp_relin[out, 1]), ttemp_relin[out, 2])
            output_cplx_scaled[out] = self.engine.rescale(output_cplx[out])
        return output_cplx_scaled
    
    
    def dense(self, x: np.ndarray[DataStruct]):
        """
        @param x: input, ciphertext array of shape (32,)
        @return: output, ciphertext array of shape (8,)
        """

        lev = x[0].level_calc
        wx = self.evaluator.pt_ct_matmul(self.weights['dense.weight'], x) #Level: l -> l + 2
        if wx.shape[0] != 8:
            raise ValueError("Shape of wx should be (8,)")
                
        mask1 = np.ones((self.engine.num_slots,), dtype=int)
        mask1[np.arange(self.engine.num_slots) % (16) < 6] = 0
        
        for i in range(8):
            temp = self.engine.rotate_left(self.engine.mc_mult(
                mask1, wx[i]), 6) #Level: l + 3, scale: delta
            wx[i] = self.engine.cc_add(self.engine.level_up(wx[i], lev+3), temp)
            wx[i] = self.engine.pc_add(self.weights['dense.bias'][i], wx[i])
            wx[i] = self.engine.cc_add(wx[i], self.engine.conjugate(wx[i]))
        return wx
    

class ThorBertFF(ThorModule):
    def __init__(self, evaluator:ThorLinearEvaluator, weights: dict[np.ndarray], layer_idx: int):
        self.evaluator = evaluator
        self.engine = evaluator.engine
        self.weights = {}
        self.weights['dense1.weight'] = weights[f'bert.encoder.layer.{layer_idx}.intermediate.dense.weight']
        self.weights['dense1.bias'] = weights[f'bert.encoder.layer.{layer_idx}.intermediate.dense.bias']
        self.weights['dense2.weight'] = weights[f'bert.encoder.layer.{layer_idx}.output.dense.weight']
        self.weights['dense2.bias'] = weights[f'bert.encoder.layer.{layer_idx}.output.dense.bias']
        self.weights['LayerNorm.weight'] = weights[f'bert.encoder.layer.{layer_idx}.output.LayerNorm.weight']
        self.weights['LayerNorm.bias'] = weights[f'bert.encoder.layer.{layer_idx}.output.LayerNorm.bias']
        self.keys = ['dense1.weight', 'dense1.bias',
                     'dense2.weight', 'dense2.bias',
                     'LayerNorm.weight', 'LayerNorm.bias'
                     ]

        self.devices = []
        
    def dense1(self, x: np.ndarray): #Matmul + bias
        """
        Input: ciphertext array of shape (64,), complexified
        Returns ciphertext array of shape (2,8)
        """
        #Shape of self.weights['dense1.weight'] is (2, 8, 6, 64)

        if x.shape != (64,):
            raise ValueError("Input of ff dense1 should be (64,)")
        
        wx = np.full((2,8), None, dtype=DataStruct)
        wx[0] = self.evaluator.pt_ct_matmul(self.weights['dense1.weight'][0], x, mode='block_diag_2') #Level: l -> l + 2
        wx[1] = self.evaluator.pt_ct_matmul(self.weights['dense1.weight'][1], x, mode='block_diag_2')
        for i in range(2):
            for j in range(8):
                wx[i,j] = self.engine.pc_add(self.weights['dense1.bias'][i, j], wx[i,j])
                wx[i,j] = self.engine.cc_add(wx[i, j], self.engine.conjugate(wx[i, j]))
        return wx
    
    def dense2(self, x: np.ndarray, debug=False): #Matmul + bias
        """
        Input: ciphertext array of shape (2,8)
        @Return: ciphertext array of shape (8,)
        """
        if x.shape != (2,8):
            raise ValueError("Ouptut of ff1 should be (2,8)")
        l = np.full((2, 64), None, dtype=DataStruct)
        wx = np.full((2, 8), None, dtype=DataStruct)
        x_out = np.full((8,), None, dtype=DataStruct)
        result = np.full((8,), None, dtype=DataStruct)
        for n in range(2):
            for i in range(4):
                l[n, 16*i] = self.engine.cc_add(x[n, i], self.engine.imult(x[n, i+4]))
                for j in range(1, 16):
                    index = 16*i + j 
                    #Change this to hrot later
                    l[n, index] = self.engine.rotate_left(l[n, index-1], 2**11)    
            wx[n] = self.evaluator.pt_ct_matmul(self.weights['dense2.weight'][n], l[n], mode='block_diag_2')
            
        mask = np.ones((self.engine.num_slots,), dtype=int)
        mask[np.arange(self.engine.num_slots) % (16) >= 6] = 0
        for i in range(8):
            x_out[i] = self.engine.cc_add(wx[0,i], wx[1,i])
            x_out[i] = self.engine.cc_add(x_out[i], self.engine.rotate_left(x_out[i], 8))
            result[i] = self.engine.pc_add(self.weights['dense2.bias'][i], x_out[i])
            result[i] = self.engine.cc_add(result[i], self.engine.conjugate(result[i]))
        if debug:
            for i in range(8):
                x_out[i] = self.engine.cc_add(x_out[i], self.engine.conjugate(x_out[i]))
            return result, x_out
        return result

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_PLAIN = os.path.join(_REPO, "Plain")
_ENC_ROOT = os.path.join(_HERE, "encoded_models")
if _PLAIN not in sys.path:
    sys.path.insert(0, _PLAIN)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from encode_weights import encoded_layer_pkl  # noqa: E402


def _thor_paths(dataset: str) -> dict[str, str]:
    enc = os.path.join(_ENC_ROOT, dataset)
    return {
        "dataset": os.path.join(_REPO, "glue_datasets", dataset),
        "encoded_dir": enc,
        "att": os.path.join(enc, "att.pkl"),
        "ff": os.path.join(enc, "ff.pkl"),
        "pooler": os.path.join(enc, "pooler.pkl"),
        "cls": os.path.join(enc, "cls.pkl"),
    }


def _require_assets(
    paths: dict[str, str],
    *,
    layers: list[int] | None = None,
    need_pooler_cls: bool = False,
    gelu_levels: dict[int, int] | None = None,
) -> None:
    missing = []
    for k in ("dataset",):
        if not os.path.exists(paths[k]):
            missing.append(f"{k}={paths[k]}")
    dataset = os.path.basename(paths.get("encoded_dir") or "") or "mrpc"
    if "encoded_models" in (paths.get("encoded_dir") or ""):
        dataset = os.path.basename(paths["encoded_dir"])
    if layers:
        for li in layers:
            att_p = encoded_layer_pkl(dataset, "att", li)
            if not os.path.isfile(att_p):
                missing.append(f"att_L{li}={att_p}")
            g_lv = 1 if gelu_levels is None else int(gelu_levels[li])
            ff_p = encoded_layer_pkl(dataset, "ff", li, gelu_level=g_lv)
            if not os.path.isfile(ff_p):
                missing.append(f"ff_L{li}={ff_p}")
    if need_pooler_cls:
        for k in ("pooler", "cls"):
            if not os.path.exists(paths[k]):
                missing.append(f"{k}={paths[k]}")
    if missing:
        raise FileNotFoundError(
            "Missing THOR assets "
            + ", ".join(missing)
            + ". Encode with: python3 HE/encode_weights.py "
            "(FF: ff_L{n}_low.pkl / ff_L{n}_high.pkl)"
        )


def _drop_he_weight_keys(module, keys: list[str] | tuple[str, ...]) -> None:
    """Drop finished DualRail PT tiles from CPU/CUDA to free RAM/VRAM."""
    import gc

    w = getattr(module, "weights", None)
    if not isinstance(w, dict):
        return
    dropped = 0
    for key in keys:
        arr = w.get(key)
        if arr is None:
            continue
        try:
            for index in np.ndindex(arr.shape):
                cell = arr[index]
                if isinstance(cell, list):
                    for j, item in enumerate(cell):
                        cell[j] = None
                        del item
                    arr[index] = None
                elif torch.is_tensor(cell):
                    arr[index] = None
                else:
                    arr[index] = None
        except Exception:
            pass
        w[key] = None
        dropped += 1
    if dropped:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            used = torch.cuda.memory_allocated(0) / 1024**3
            print(
                f"  [mem] dropped {dropped} weight key(s) "
                f"cuda_alloc={used:.2f} GiB",
                flush=True,
            )
        except Exception:
            pass


def _release_he_module_weights(module) -> None:
    """Release any remaining DualRail weights on a Thor module."""
    keys = list(getattr(module, "keys", []) or [])
    if not keys and isinstance(getattr(module, "weights", None), dict):
        keys = list(module.weights.keys())
    _drop_he_weight_keys(module, keys)
    try:
        module.devices = []
    except Exception:
        pass


def _forward_layer(
    *,
    engine,
    evaluator,
    thor_attention,
    thor_ff,
    layer_idx: int,
    x,
    thor_attention_mask,
    devices,
    bts_hook=None,
    rem_guard=None,
):
    """One encoder layer. Bootstrap sites follow the plan.

    ``rem_guard`` stops the layer when HE remaining levels disagree with the plan.
    """
    from bootstrap import (
        plan_refresh_cts,
        plan_refresh_dualrail8,
        plan_refresh_dualrail_ff,
        refresh_dualrail8,
    )
    from nonlinear.softmax import align_join

    def _as_object_array(seq):
        """Build object ndarray without np.asarray (Liberate CT ⇔ CUDA tensor)."""
        seq = list(seq)
        out = np.empty(len(seq), dtype=object)
        for i, v in enumerate(seq):
            out[i] = v
        return out

    def _he_rem(cts) -> int:
        return int(engine.num_levels) - int(list(cts)[0].level_calc)

    def _probe(site: str, cts, *, note: str = "") -> None:
        if rem_guard is None:
            return
        rem = _he_rem(cts)
        extra = f" ({note})" if note else ""
        print(
            f"  [probe] {site}: remaining={rem}{extra}",
            flush=True,
        )
        rem_guard.check_probe(site, rem)

    thor_attention.to(devices)
    # FF DualRail PTs stay on CPU until after LN1 — halves peak VRAM vs loading both.
    prefix = f"L{int(layer_idx)}"
    print(
        f"HE forward layer_idx={layer_idx} "
        f"bts={getattr(bts_hook, 'mode', 'none')}"
        f"{'/mock' if getattr(engine, '_mock_bootstrap_info', None) else ''}",
        flush=True,
    )

    # Layer entry is always 8 real (L0: encrypt_embedding_real8; L1+: ln2_out).
    if x.shape != (8,):
        raise ValueError(
            f"layer entry must be 8 real packs, got x.shape={x.shape}"
        )

    # 固定 pack 8 real → 4 cplx（非微事件）；残差旁路仍为 ``x``
    x_cplx = np.full((4,), None, dtype=object)
    for i in range(4):
        x_cplx[i] = engine.cc_add(x[i], engine.imult(x[i + 4]))

    # Layer-entry rem（对齐 DP ``__initial__`` / 跨层出口）——须在 QKV before-boot 之前。
    _probe("entry_cplx", x_cplx)

    # Plan site ``linear_qkv`` before：刷 4 cplx；若刷则 unpack 回写残差 ``x``。
    # 必须在 emb-duplicate 之前：否则 unpack 会把 duplicate 写进残差。
    _qkv_ev = f"{prefix}.linear_qkv"
    if bts_hook is not None and bts_hook.should_refresh(_qkv_ev):
        print(
            f"  [smoke_thor_repro] linear_qkv refresh "
            f"pack_mode={bts_hook.pack_mode(_qkv_ev)}",
            flush=True,
        )
        from bootstrap import unpack_cplx4_to_real8

        x_cplx = _as_object_array(
            plan_refresh_cts(
                engine,
                x_cplx,
                bts_hook,
                _qkv_ev,
            )
        )
        x = unpack_cplx4_to_real8(
            engine,
            x_cplx,
            hook=bts_hook,
            event_name=_qkv_ev,
        )

    # L0 embedding 8-real already has %16∈[6,12) filled; L1+ LN layout needs
    # emb-duplicate via cc_add(z, rotate_left(z, -6)) — only on QKV ``x_cplx``.
    if layer_idx != 0:
        for i in range(4):
            x_cplx[i] = engine.cc_add(
                x_cplx[i], engine.rotate_left(x_cplx[i], -6)
            )

    x_cplx_rots = evaluator.make_rotated_copies(x_cplx)
    q_wo_rescale = thor_attention.query(x_cplx_rots)
    k = thor_attention.key(x_cplx_rots)
    v = thor_attention.value(x_cplx_rots)
    _probe("after_qkv", q_wo_rescale, note="Q before rescale")
    _probe("after_qkv_v", v, note="V parked")

    # Plan site ``linear_att_score`` before：刷 Q+K（V 旁路不动）。
    _qk_ev = f"{prefix}.linear_att_score"
    if bts_hook is not None and bts_hook.should_refresh(_qk_ev):
        print(
            f"  [smoke_thor_repro] Q∥K before-score refresh "
            f"event={_qk_ev} pack_mode={bts_hook.pack_mode(_qk_ev)}",
            flush=True,
        )
        _qk = list(q_wo_rescale) + list(k)
        _qk = list(
            plan_refresh_cts(
                engine,
                _qk,
                bts_hook,
                _qk_ev,
            )
        )
        q_wo_rescale = _as_object_array(_qk[:4])
        k = _as_object_array(_qk[4:])

    l_k = evaluator.transpose_upper_to_lower(k)
    _probe("after_transpose_k", l_k)
    l_k_cplx = np.full((4,), None, dtype=object)
    for i in range(4):
        l_k_cplx[i] = engine.cc_add(
            engine.level_up(l_k[i], l_k[i].level_calc + 1),
            engine.imult(evaluator.rotate_internal(l_k[i], 64, mode="att")),
        )
        l_k_cplx[i] = engine.rescale(l_k_cplx[i])
    _probe("after_k_complexify", l_k_cplx)

    q = np.full_like(q_wo_rescale, None, dtype=object)
    for i in range(4):
        q[i] = engine.rescale(q_wo_rescale[i])
    _probe("after_q_rescale", q)
    q_copies = evaluator.make_copies(q)
    _probe("after_make_copies", q_copies)
    sftmx_in = thor_attention.calculate_attention_score(
        l_k_cplx, q_copies, bootstrap=False, scale=1, rescale=False
    )
    _probe("after_att_score", sftmx_in)
    # Q/K linear done — free DualRail PT tiles before Softmax peak CTs.
    _drop_he_weight_keys(
        thor_attention,
        ("query.weight", "query.bias", "key.weight", "key.bias"),
    )

    # Softmax 前 DualRail：仅 plan 站点（score / stockmeyer）。
    if bts_hook is not None:
        for _sev in (
            f"{prefix}.softmax_exp_stockmeyer",
        ):
            if not bts_hook.should_refresh(_sev):
                continue
            print(
                f"  [smoke_thor_repro] pre-Softmax DualRail "
                f"event={_sev} pack_mode={bts_hook.pack_mode(_sev)}",
                flush=True,
            )
            from bootstrap import refresh_dualrail8

            sftmx_in = _as_object_array(
                refresh_dualrail8(
                    engine,
                    list(sftmx_in),
                    bts_hook,
                    _sev,
                )
            )
            break
    _probe("after_bts_score", sftmx_in)

    sftmx_out = thor_attention.softmax(
        x=sftmx_in, attention_mask=thor_attention_mask, rescale=False, debug=False, sk=None
    )
    _probe("after_softmax", sftmx_out)

    # Plan site ``linear_att_context`` before：刷 128 α（主路 inputs；V 旁路见下）。
    _ctx_ev = f"{prefix}.linear_att_context"
    if bts_hook is not None and bts_hook.should_refresh(_ctx_ev):
        print(
            f"  [smoke_thor_repro] linear_att_context refresh "
            f"pack_mode={bts_hook.pack_mode(_ctx_ev)}",
            flush=True,
        )
        sftmx_out = _as_object_array(
            plan_refresh_cts(
                engine,
                list(sftmx_out),
                bts_hook,
                _ctx_ev,
            )
        )

    v_cplx = np.full((2,), None, dtype=object)
    for i in range(2):
        v_cplx[i] = engine.cc_add(v[i], engine.imult(v[i + 2]))
    sftmx_out, v_cplx = align_join(
        engine,
        sftmx_out,
        v_cplx,
        bts_hook=bts_hook,
        bypass_event=f"{prefix}.bypass.v_to_softmax",
    )
    for i in range(2):
        v_cplx[i] = engine.rescale(v_cplx[i])
    sftmx_out_rescale = np.full((128,), None, dtype=object)
    for j in range(128):
        sftmx_out_rescale[j] = engine.rescale(sftmx_out[j])
    att_context = thor_attention.calculate_attention_context(
        v_cplx, sftmx_out_rescale, rescale=False
    )
    _probe("after_att_context", att_context)
    _drop_he_weight_keys(
        thor_attention,
        ("value.weight", "value.bias"),
    )
    _ctx_rem = int(engine.num_levels) - int(att_context[0].level_calc)
    print(
        f"  [smoke_thor_repro] after_att_context rem={_ctx_rem} "
        f"lc={att_context[0].level_calc}",
        flush=True,
    )

    # Plan site ``bridge_rot_attn_dense`` before：刷 2 context（主路 inputs）。
    _rot_ev = f"{prefix}.bridge_rot_attn_dense"
    if bts_hook is not None and bts_hook.should_refresh(_rot_ev):
        print(
            f"  [smoke_thor_repro] bridge_rot_attn_dense refresh "
            f"pack_mode={bts_hook.pack_mode(_rot_ev)}",
            flush=True,
        )
        att_context = _as_object_array(
            plan_refresh_cts(
                engine,
                att_context,
                bts_hook,
                _rot_ev,
            )
        )
    _probe("after_bts_context", att_context)

    att_context_rots = thor_attention.evaluator.make_rotated_copies(att_context)

    # Plan site ``linear_attn_dense`` before：刷 32 rot copies（主路 inputs）。
    _dense_ev = f"{prefix}.linear_attn_dense"
    if bts_hook is not None and bts_hook.should_refresh(_dense_ev):
        print(
            f"  [smoke_thor_repro] linear_attn_dense refresh "
            f"pack_mode={bts_hook.pack_mode(_dense_ev)}",
            flush=True,
        )
        att_context_rots = _as_object_array(
            plan_refresh_cts(
                engine,
                att_context_rots,
                bts_hook,
                _dense_ev,
            )
        )
    dense_output = thor_attention.dense(att_context_rots)
    _probe("after_attn_dense", dense_output)
    _drop_he_weight_keys(
        thor_attention,
        ("dense.weight", "dense.bias"),
    )
    _join_attn = f"{prefix}.join_resid_attn"
    if bts_hook is not None and bts_hook.should_refresh(_join_attn):
        print(
            f"  [smoke_thor_repro] join_resid_attn refresh "
            f"pack_mode={bts_hook.pack_mode(_join_attn)}",
            flush=True,
        )
        dense_output = _as_object_array(
            plan_refresh_dualrail8(
                engine, dense_output, bts_hook, _join_attn
            )
        )
    dense_output, x = align_join(
        engine,
        dense_output,
        x,
        bts_hook=bts_hook,
        bypass_event=f"{prefix}.bypass.resid_attn",
    )
    x_out_sum = np.full((8,), None, dtype=object)
    for i in range(8):
        x_out_sum[i] = engine.add(x[i], dense_output[i])
    ln1_in = x_out_sum
    ln1_out = thor_attention.layernorm(x=ln1_in, sk=None)
    _probe("after_ln1", ln1_out)
    # Attn LayerNorm linear done; free γ/β before FF weights hit GPU.
    _drop_he_weight_keys(
        thor_attention,
        ("LayerNorm.weight", "LayerNorm.bias"),
    )
    thor_ff.to(devices)

    # Plan site ``bridge_rot_ff1`` before：刷 8 LN1（主路 inputs）。
    # FF 残差会合：``join_resid_ff`` + ``bypass.resid_ff``。
    _ff_rot_ev = f"{prefix}.bridge_rot_ff1"
    if bts_hook is not None and bts_hook.should_refresh(_ff_rot_ev):
        print(
            f"  [smoke_thor_repro] bridge_rot_ff1 refresh "
            f"pack_mode={bts_hook.pack_mode(_ff_rot_ev)}",
            flush=True,
        )
        ln1_out = _as_object_array(
            plan_refresh_dualrail8(
                engine,
                ln1_out,
                bts_hook,
                _ff_rot_ev,
            )
        )
    _probe("after_bts_bridge_rot_ff1", ln1_out)

    l = np.full((64,), None, dtype=object)
    mask = np.full((engine.num_slots,), 1, dtype=int)
    mask[np.arange(engine.num_slots) % 16 >= 6] = 0
    for i in range(4):
        temp = engine.cc_add(ln1_out[i], engine.imult(ln1_out[i + 4]))
        temp = engine.mc_mult(mask, temp)
        l[16 * i] = engine.cc_add(temp, engine.rotate_left(temp, -8))
        for j in range(1, 16):
            index = 16 * i + j
            l[index] = engine.rotate_left(l[index - 1], 2**11)
    _probe("after_ff_bridge_mask", l, note="mc_mult on LN1→rot64")

    # Plan site ``linear_ff_dense1`` before：刷 64 rot copies（已是复数侧，直接刷）。
    _ff_d1_ev = f"{prefix}.linear_ff_dense1"
    if bts_hook is not None and bts_hook.should_refresh(_ff_d1_ev):
        print(
            f"  [smoke_thor_repro] linear_ff_dense1 refresh "
            f"pack_mode={bts_hook.pack_mode(_ff_d1_ev)}",
            flush=True,
        )
        l = _as_object_array(
            plan_refresh_cts(
                engine,
                list(l),
                bts_hook,
                _ff_d1_ev,
            )
        )

    gelu_in_wo_bs = thor_ff.dense1(l)
    _probe("after_ff_dense1", gelu_in_wo_bs[0])
    _drop_he_weight_keys(
        thor_ff,
        ("dense1.weight", "dense1.bias"),
    )
    # GeLU f1/f2/reconstruct hooks live inside ``he_gelu_cheb`` / ``gelu_poly_pack``.
    _probe("after_ff_dense1_pre_gelu", gelu_in_wo_bs[0])

    gelu_out = thor_ff.gelu(x=gelu_in_wo_bs)
    _probe("after_gelu", gelu_out[0])

    # Plan site ``linear_ff_dense2`` before：DualRail 刷 GeLU 出口 (2,8)。
    _ff_d2_ev = f"{prefix}.linear_ff_dense2"
    if bts_hook is not None and bts_hook.should_refresh(_ff_d2_ev):
        print(
            f"  [smoke_thor_repro] linear_ff_dense2 refresh "
            f"pack_mode={bts_hook.pack_mode(_ff_d2_ev)}",
            flush=True,
        )
        gelu_out = plan_refresh_dualrail_ff(
            engine,
            gelu_out,
            bts_hook,
            _ff_d2_ev,
        )
        _probe("after_bts_post_gelu", gelu_out[0])

    dense2_out = thor_ff.dense2(gelu_out)
    _probe("after_ff_dense2", dense2_out)
    _drop_he_weight_keys(
        thor_ff,
        ("dense2.weight", "dense2.bias"),
    )
    _join_ff = f"{prefix}.join_resid_ff"
    if bts_hook is not None and bts_hook.should_refresh(_join_ff):
        print(
            f"  [smoke_thor_repro] join_resid_ff refresh "
            f"pack_mode={bts_hook.pack_mode(_join_ff)}",
            flush=True,
        )
        dense2_out = _as_object_array(
            plan_refresh_dualrail8(
                engine, dense2_out, bts_hook, _join_ff
            )
        )
    dense2_out, ln1_out = align_join(
        engine,
        dense2_out,
        ln1_out,
        bts_hook=bts_hook,
        bypass_event=f"{prefix}.bypass.resid_ff",
    )
    ln2_in = np.full((8,), None, dtype=object)
    for i in range(8):
        ln2_in[i] = engine.add(ln1_out[i], dense2_out[i])
    # LN2 DualRail / scale hooks 在 ``he_layernorm_poly``（``ln2_scale`` 起）
    ln2_out = thor_ff.layernorm(x=ln2_in, sk=None)
    _probe("after_ln2", ln2_out)
    # No level_up(→21): next layer / plan decides rem; do not burn CT for old W@21.

    _drop_he_weight_keys(
        thor_ff,
        ("LayerNorm.weight", "LayerNorm.bias"),
    )
    _release_he_module_weights(thor_attention)
    _release_he_module_weights(thor_ff)
    return ln2_out, {
        "x": x,
        "q": q_wo_rescale,
        "sftmx_in": sftmx_in,
        "sftmx_out": sftmx_out,
        "att_context": att_context,
        "ln1_in": ln1_in,
        "ln1_out": ln1_out,
        "gelu_in_wo_bs": gelu_in_wo_bs,
        "gelu_out": gelu_out,
        "dense2_out": dense2_out,
        "ln2_in": ln2_in,
        "ln2_out": ln2_out,
    }

"""
HE BertPooler + classifier（THOR DualRail 布局）。

入口：第 12 层 encoder 的 8 条 real DualRail（ln2_out）。
出口：n_cls 条密文，logit 在 slot 0（与官方 ThorBertClassifier 相同）。

Bts 不进 bts_plan：
  1. 入口 DualRail 8 强制刷一次（encoder 出口 rem 不固定）
  2. tanh：p1 前、p2 后各 ``engine.bootstrap`` 一次（同 THOR he_tanh）

Tanh：ŷ = p2(p1(z/C))，默认 C=5、nolinear/tanh_out 自拟合幂基。
"""
import os
from typing import Any, Sequence

import numpy as np
import torch
from liberate.fhe.data_struct import DataStruct

from tanh_poly import DEFAULT_COEFF_JSON, load_tanh_fit

from bootstrap import refresh_dualrail8
from nonlinear.tanh import _tanh_coeffs, he_tanh_ct


def load_weights(path: str) -> dict:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)

_THOR_ENC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoded_models")

HEAD_ROT_DELTAS: list[int] = [
    32,
    64,
    128,
    1024,
    16384,
    -16,
    -32,
    -64,
    -128,
    -256,
    -512,
    -1024,
]


def encoded_head_pkl(dataset: str, kind: str, *, encoded_dir: str | None = None) -> str:
    if kind not in ("pooler", "cls"):
        raise ValueError(f"kind must be pooler|cls, got {kind!r}")
    root = encoded_dir or os.path.join(_THOR_ENC, dataset)
    return os.path.join(root, f"{kind}.pkl")


def ensure_head_rot_keys(engine, sk) -> None:
    """Classifier / CLS-broadcast 旋转；encoder 加载时未必带齐。"""
    engine.add_rot_keys_from_sk(list(HEAD_ROT_DELTAS), sk)


def _move_weight_arr_to_device(arr: np.ndarray, device: int) -> None:
    for index in np.ndindex(arr.shape):
        cell = arr[index]
        if isinstance(cell, torch.Tensor):
            arr[index] = [cell.to(f"cuda:{device}")]
        elif isinstance(cell, list) and cell and torch.is_tensor(cell[0]):
            arr[index] = [cell[0].to(f"cuda:{device}")]


def _pooler_dense(evaluator, engine, x4: np.ndarray, w: np.ndarray, b: np.ndarray):
    """官方 ThorBertPooler.dense：CLS mask + 广播 + 6 块对角 GEMM + bias。"""
    if x4.shape != (4,):
        raise ValueError(f"pooler dense 输入应为 (4,) 复密文，得到 {x4.shape}")
    lev = int(x4[0].level_calc)
    nslot = int(engine.num_slots)
    mask = np.zeros((nslot,), dtype=np.float64)
    mask[np.arange(nslot) % (2**11) < 6] = 1.0

    x = list(x4)
    for index, ct in enumerate(x):
        ct_masked = engine.mc_mult(mask, ct)
        for i in range(7):
            ct_rot = engine.rotate_left(ct_masked, -16 * (2**i))
            ct_masked = engine.cc_add(ct_masked, ct_rot)
        x[index] = ct_masked

    wx_l = np.full((6,), None, dtype=object)
    for l in range(6):
        temp = engine.pt_ct_mult(w[l, 0], x[0])
        for i in range(1, 4):
            temp = engine.cc_add(temp, engine.pt_ct_mult(w[l, i], x[i]))
        wx_l[l] = engine.rescale(temp)

    temp2 = engine.level_up(wx_l[0], lev + 3)
    for l in range(1, 6):
        rotated = evaluator.rotate_internal(wx_l[l], delta=6 - l, mode="block_diag_2")
        temp2 = engine.cc_add(temp2, rotated)
    temp2 = evaluator.rotsum(temp2, 2**11)
    out = engine.pc_add(b[0], temp2)
    out = engine.cc_add(out, engine.conjugate(out))
    wx = np.empty((1,), dtype=object)
    wx[0] = out
    return wx


def _classifier_dense(engine, x1: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """官方 ThorBertClassifier.dense：逐类内积 + rot-sum → slot 0。"""
    if x1.shape != (1,):
        raise ValueError(f"classifier 输入应为 (1,)，得到 {x1.shape}")
    n_cls = int(w.shape[0])
    wx = np.full((n_cls,), None, dtype=object)
    for i in range(n_cls):
        wx[i] = engine.pt_ct_mult(w[i], x1[0])
        temp = engine.cc_add(wx[i], engine.rotate_left(wx[i], 1))
        temp2 = engine.cc_add(temp, engine.rotate_left(temp, 2))
        wx[i] = engine.cc_add(temp2, engine.rotate_left(temp, 4))
        for j in range(4, 11):
            wx[i] = engine.cc_add(wx[i], engine.rotate_left(wx[i], 2**j))
        wx[i] = engine.rescale(wx[i])
        wx[i] = engine.pc_add(b[i], wx[i])
    return wx


class HeBertHead:
    """8 DualRail ln2 → pooler tanh → classifier logits CTs。"""

    def __init__(
        self,
        evaluator,
        *,
        dataset: str,
        encoded_dir: str | None = None,
        device: int = 0,
        tanh_C: float | None = None,
        tanh_coeff_path: str | None = None,
    ):
        self.evaluator = evaluator
        self.engine = evaluator.engine
        self.device = int(device)
        enc = encoded_dir or os.path.join(_THOR_ENC, dataset)
        pooler_path = encoded_head_pkl(dataset, "pooler", encoded_dir=enc)
        cls_path = encoded_head_pkl(dataset, "cls", encoded_dir=enc)
        if not os.path.isfile(pooler_path):
            raise FileNotFoundError(f"missing pooler.pkl: {pooler_path}")
        if not os.path.isfile(cls_path):
            raise FileNotFoundError(f"missing cls.pkl: {cls_path}")
        pw = load_weights(pooler_path)
        cw = load_weights(cls_path)
        self.w_pool = pw["bert.pooler.dense.weight"]
        self.b_pool = pw["bert.pooler.dense.bias"]
        cls_name = (
            "cls.seq_relationship"
            if "cls.seq_relationship.weight" in cw
            else "classifier"
        )
        self.w_cls = cw[f"{cls_name}.weight"]
        self.b_cls = cw[f"{cls_name}.bias"]
        _move_weight_arr_to_device(self.w_pool, self.device)
        _move_weight_arr_to_device(self.b_pool, self.device)
        _move_weight_arr_to_device(self.w_cls, self.device)
        _move_weight_arr_to_device(self.b_cls, self.device)
        self.tanh_C, self.p1, self.p2 = _tanh_coeffs(tanh_coeff_path, tanh_C)

    def forward(self, packs8: Sequence[Any] | np.ndarray) -> np.ndarray:
        packs = list(packs8)
        if len(packs) != 8:
            raise ValueError(f"head 入口应为 8 条 DualRail，得到 {len(packs)}")
        rem0 = int(self.engine.num_levels) - int(packs[0].level_calc)
        print(
            f"  [he-head] entry rem={rem0} → DualRail8 bootstrap (not in plan)",
            flush=True,
        )
        packs = refresh_dualrail8(
            self.engine, packs, None, "head.entry_dualrail8"
        )
        rem1 = int(self.engine.num_levels) - int(packs[0].level_calc)
        print(f"  [he-head] after entry bts rem={rem1}", flush=True)

        x_cplx = np.empty((4,), dtype=object)
        for i in range(4):
            x_cplx[i] = self.engine.cc_add(packs[i], self.engine.imult(packs[i + 4]))
        pooled = _pooler_dense(
            self.evaluator, self.engine, x_cplx, self.w_pool, self.b_pool
        )
        pooled[0] = self.engine.mult_scalar(pooled[0], 1.0 / self.tanh_C)
        print(
            f"  [he-head] tanh C={self.tanh_C:g} Stockmeyer p2(p1(z/C)) + 2×bts",
            flush=True,
        )
        pooled[0] = he_tanh_ct(self.engine, pooled[0], p1=self.p1, p2=self.p2)
        logits_ct = _classifier_dense(self.engine, pooled, self.w_cls, self.b_cls)
        return logits_ct


def decrypt_logits_slot0(engine, sk, logits_ct: np.ndarray) -> np.ndarray:
    out = []
    for i in range(len(logits_ct)):
        vec = np.asarray(engine.decrode(logits_ct[i], sk, is_real=False)).ravel()
        out.append(float(np.real(vec[0])))
    return np.asarray(out, dtype=np.float64)
