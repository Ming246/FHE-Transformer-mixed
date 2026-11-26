import torch 
import numpy as np
from liberate.fhe.data_struct import DataStruct
from liberate.fhe.presets import types, errors
from liberate.fhe import ckks_engine
from liberate.fhe.encdec import rotate
from liberate.fhe.bootstrapping import ckks_bootstrapping as bs

class CkksEngine(ckks_engine):
    def __init__(self, params, verbose=False):
        """
        CkksEngine is a child class of Liberate ckks_engine for basic CKKS operations.
        """
        super().__init__(**params, verbose=verbose)
        self.rot_keys = np.full((2**15,), None, dtype=DataStruct) #Note that this is right rotation, as Liberate uses right rotation as default
        self.hrot_keys = np.full((2**15,), None, dtype=DataStruct)
        self.pk = None
        self.evk = None
        self.conj_key = None
        self.fft_error = None
        self.one = None
    
    #Precomputation
    def add_pk(self, pk:DataStruct):
        self.pk = pk
        
    def add_rot_keys_from_sk(self, deltas:list[int], sk:DataStruct):
        """
        Add left rotation keys for the given deltas.
        """
        for delta in deltas:
            if delta < 0:
                delta = self.num_slots + delta
            if self.rot_keys[delta] == None:
                self.rot_keys[delta] = self.create_rotation_key(sk, self.num_slots - delta)
            
    def add_hrot_keys_from_sk(self, deltas:list[int], sk:DataStruct):
        """
        Add left hoist rotation keys for the given deltas.
        """
        for delta in deltas:
            if delta < 0:
                delta = self.num_slots + delta
            self.hrot_keys[delta] = self.create_hoisting_rotation_key(sk, self.num_slots - delta)
            
    def add_bs_key(self, bsk:DataStruct):
        self.bs_key = bsk
            
    def add_evk(self, evk:DataStruct):
        self.evk = evk

    def add_gk(self, gk:DataStruct):
        self.gk = gk
        
    def add_conj_key(self, conj_key:DataStruct):
        self.conj_key = conj_key
            
    #Basic Operations
    def encode_and_encrypt(self, m, pk=None, level: int = 0, padding=True) -> DataStruct:
        if pk is None:
            if self.pk is None:
                raise ValueError("Public key is not set")
            pk = self.pk
        return super().encodecrypt(m, pk, level, padding)
    
    def bootstrap(self, ct):
        for device in self.ntt.devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        temp = ct
        ct_bs = bs.bootstrap(self, temp, self.bs_key, self.evk, self.conj_key, self.pk)
        for device in self.ntt.devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        return ct_bs
    
    #Mult
    def mult(self, a, b, evk=None, relin:bool=True, auto_level=False) -> DataStruct:
        if evk is None:
            if self.evk is None:
                raise ValueError("Evaluation key is not set")
            evk = self.evk
        return super().mult(a, b, evk, relin=relin)
    
    def auto_ct_ct_mult(self, ct0: DataStruct, ct1: DataStruct, evk=None, relin=True, rescale=True) -> DataStruct:
        if evk is None:
            if self.evk is None:
                raise ValueError("Evaluation key is not set")
            evk = self.evk
        lct0, lct1 = self.auto_level(ct0, ct1)
        return self.ct_ct_mult(lct0, lct1, evk, relin=relin, rescale=rescale)

    def ct_ct_mult(self, a: DataStruct, b: DataStruct, evk =None, relin=True, rescale=True) -> DataStruct:
        return super().cc_mult(a, b, evk, relin, rescale)
    
    def relinearize(self, ct_triplet: DataStruct, evk: DataStruct=None, is_fast:bool=True) -> DataStruct:
        if evk is None:
            if self.evk is None:
                raise ValueError("Evaluation key is not set")
            evk = self.evk
        return super().relinearize(ct_triplet=ct_triplet, evk=evk, is_fast=is_fast)
    
    def sqrt(self, ct: DataStruct, evk=None, e=0.0001, alpha=0.0001) -> DataStruct:
        if evk is None:
            if self.evk is None:
                raise ValueError("Evaluation key is not set")
            evk = self.evk
        return super().sqrt(ct, self.evk, e, alpha)
    
    def square(self, ct: DataStruct, evk =None, relin=True, is_fast: bool = True):
        if evk is None:
            if self.evk is None:
                raise ValueError("Evaluation key is not set")
            evk = self.evk
        return super().square(ct, evk, relin, is_fast)
    
    def imult(self, x: np.ndarray[DataStruct]) -> np.ndarray[DataStruct]:
        return self.mult_imag(x)
    
    def minus_imult(self, x: np.ndarray[DataStruct]) -> np.ndarray[DataStruct]:
        return self.mult_imag(x, True)
    
    def pt_ct_mult(self, pt, ct) -> DataStruct:
        """
        Plaintext - Ciphertext multiplication.
        """
        
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchmode(origin=ct.origin, to=types.origins["sk"])
            
        pt_tiled = self.ntt.tile_unsigned(pt, ct.level_calc)
        # Transform ntt to prepare for multiplication.
        self.ntt.enter_ntt(pt_tiled, ct.level_calc)
        # Prepare a new ct.
        new_ct = self.clone(ct)


        self.ntt.enter_ntt(new_ct.data[0], ct.level_calc)
        self.ntt.enter_ntt(new_ct.data[1], ct.level_calc)

        new_d0 = self.ntt.mont_mult(pt_tiled, new_ct.data[0], ct.level_calc)
        new_d1 = self.ntt.mont_mult(pt_tiled, new_ct.data[1], ct.level_calc)
        
        self.ntt.intt_exit_reduce(new_d0, ct.level_calc)
        self.ntt.intt_exit_reduce(new_d1, ct.level_calc)

        new_ct.data[0] = new_d0
        new_ct.data[1] = new_d1

        return new_ct
    
    def pt_ct_mult_extended(engine, pt, ct):
        pt_tiled = engine.ntt.tile_unsigned(pt, ct.level_calc)
        engine.ntt.enter_ntt(pt_tiled, ct.level_calc)
        new_ct = engine.clone(ct)
        new_d0 = engine.ntt.mont_mult(pt_tiled, new_ct.data[0], ct.level_calc)
        new_d1 = engine.ntt.mont_mult(pt_tiled, new_ct.data[1], ct.level_calc)
        new_d2 = engine.ntt.mont_mult(pt_tiled, new_ct.data[2], ct.level_calc)
        new_ct.data[0] = new_d0
        new_ct.data[1] = new_d1
        new_ct.data[2] = new_d2
        return new_ct
    
    def mult_int_scalar_triplet(self, ct, scalar, evk=None, relin=True):
        device_len = len(ct.data[0])
        data_len = len(ct.data)
        level = ct.level_calc

        int_scalar = int(scalar)
        mont_scalar = [(int_scalar * self.ctx.R) % qi for qi in self.ctx.q]

        dest = self.ntt.p.destination_arrays[level]

        partitioned_mont_scalar = [[mont_scalar[i] for i in desti] for desti in dest]
        tensorized_scalar = []
        for device_id in range(device_len):
            scal_tensor = torch.tensor(
                partitioned_mont_scalar[device_id],
                dtype=self.ctx.torch_dtype,
                device=self.ntt.devices[device_id]
            )
            tensorized_scalar.append(scal_tensor)

        new_ct = self.clone(ct)
        new_data = new_ct.data

        for i in range(data_len):          
            self.ntt.mont_enter_scalar(new_data[i], tensorized_scalar, level)
            self.ntt.reduce_2q(new_data[i], level)

        return new_ct

    def pc_add(self, pt, ct) -> DataStruct:
        """
        Note that if the level of pt and ct is different, the result will be incorrect.
        """
        pt_tiled = self.ntt.tile_unsigned(pt, ct.level_calc)

        self.ntt.mont_enter_scale(pt_tiled, ct.level_calc)

        new_ct = self.clone(ct)
        self.ntt.mont_enter(new_ct.data[0], ct.level_calc)
        new_d0 = self.ntt.mont_add(pt_tiled, new_ct.data[0], ct.level_calc)
        self.ntt.mont_redc(new_d0, ct.level_calc)
        self.ntt.reduce_2q(new_d0, ct.level_calc)

        new_ct.data[0] = new_d0

        return new_ct
    
    #Rotations
    def rotate_left(self, ct:DataStruct, delta:int=2**60, rot_key=None) -> DataStruct:
        if delta == 0:
            return ct
        elif delta < 0:
            delta = self.num_slots + delta
        if rot_key is None:
            if delta == 2**60:
                raise ValueError("Either delta or rot_key should be given")
            elif self.rot_keys[delta] is None:
                return self.rotate_galois(ct, self.gk, -delta)
            else:
                rot_key = self.rot_keys[delta]
        return self.rotate_single(ct, rot_key)
    
    def conjugate(self, ct:DataStruct, conj_key=None) -> DataStruct:
        if conj_key is None:
            if self.conj_key is None:
                raise ValueError("Conjugation key is not set")
            conj_key = self.conj_key
        return super().conjugate(ct, conj_key)    

    def rotate_hoist(self, ct: DataStruct, delta: DataStruct, rotk=None) -> DataStruct:
        if rotk is None:
            rotk = self.hrot_keys[delta]
        include_special = ct.include_special
        ntt_state = ct.ntt_state
        montgomery_state = ct.montgomery_state
        origin = rotk.origin
        delta = int(origin.split(':')[-1])
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        level = ct.level_calc
        ct_clone = self.clone(ct)
        c0 = ct_clone.data[0]
        c1 = ct_clone.data[1]
        decomposed_part_results = self.decompose(c1, level, exit_ntt=False)
        self.ntt.mont_enter_scalar(c0, [self.mont_PR[i][-len(self.ntt.p.destination_arrays[level][i]):] for i in
                                        range(self.ntt.num_devices)], lvl=level, mult_type=-1)
        self.ntt.mont_enter_scalar(c1, [self.mont_PR[i][-len(self.ntt.p.destination_arrays[level][i]):] for i in
                                        range(self.ntt.num_devices)], lvl=level, mult_type=-1)
        c0 = [torch.cat([c0[device_id],
                            torch.zeros([self.ntt.num_special_primes, self.ctx.N], device=self.ntt.devices[device_id],
                                        dtype=self.ctx.torch_dtype)], dim=0) for device_id in range(self.ntt.num_devices)]
        summed0, summed1 = self.mult_sum(decomposed_part_results, rotk, level)
        self.ntt.intt_exit_reduce(summed0, level, mult_type=-2)
        self.ntt.intt_exit_reduce(summed1, level, mult_type=-2)
        summed0 = self.ntt.mont_add(c0, summed0, level, mult_type=-2)
        summed0 = [rotate(d, delta) for d in summed0]
        summed1 = [rotate(d, delta) for d in summed1]
        self.ntt.make_unsigned(summed0, level, mult_type=-2)
        self.ntt.reduce_2q(summed0, level, mult_type=-2)
        self.ntt.make_unsigned(summed1, level, mult_type=-2)
        self.ntt.reduce_2q(summed1, level, mult_type=-2)
        d0 = self.mod_down(summed0, level)
        d1 = self.mod_down(summed1, level)
        return DataStruct(
            data=(d0, d1),
            include_special=include_special,
            ntt_state=ntt_state,
            montgomery_state=montgomery_state,
            origin=types.origins["ct"],
            level=level,
            hash=self.hash
        )
    
    def rotsum(self, ct:DataStruct, interval:int) -> DataStruct:
        """
        Rotate Sum Operation
        """
        rep = int(np.log2(self.num_slots/interval))
        temp = ct
        for i in range(rep):
            temp = self.cc_add(temp, self.rotate_left(temp, interval*2**i))
        return temp
    
    def cc_add(self, a: DataStruct, b: DataStruct) -> DataStruct:
        if a is None:
            return b
        elif b is None:
            return a
        if a.origin == types.origins["ct"] and b.origin == types.origins["ct"]:
            ct_add = self.cc_add_double(a, b)
        elif a.origin == types.origins["ctt"] and b.origin == types.origins["ctt"]:
            ct_add = self.cc_add_triplet(a, b)
        else:
            raise errors.DifferentTypeError(a=a.origin, b=b.origin)
        return ct_add

import os
from typing import Any

from path_setup import diagnose, ensure_thor_path, liberate_resources_dir


# Matches thirdparty/THOR-main/encode.py and forward.ipynb
THOR_DEFAULT_PARAMS: dict[str, Any] = {
    "logN": 16,
    "scale_bits": 41,
    "num_special_primes": 4,
    "devices": [0],
    "quantum": "pre_quantum",
}

# Smaller engine for primitive smoke (same security model as THOR)
SMOKE_PARAMS: dict[str, Any] = {
    "logN": 15,
    "scale_bits": 40,
    "num_special_primes": 2,
    "devices": [0],
    "quantum": "pre_quantum",
    "num_scales": 8,
}

# Bootstrap rotation bundle (→ add_bs_key); from forward.ipynb
ROTK_DICT_KEYS: list[int] = [
    -32768,
    -16384,
    -1024,
    -512,
    -32,
    -16,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    32,
    64,
    96,
    128,
    160,
    192,
    224,
    256,
    288,
    320,
    352,
    384,
    416,
    448,
    480,
    512,
    1024,
    2048,
    3072,
    4096,
    5120,
    6144,
    7168,
    8192,
    9216,
    10240,
    11264,
    12288,
    13312,
    14336,
    15360,
    16384,
]

# Ordinary left-rotation deltas generated from sk; from forward.ipynb
ROT_DELTAS: list[int] = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    16,
    2048,
    4096,
    6144,
    8192,
    10240,
    12288,
    14336,
    18432,
    20480,
    22528,
    24576,
    26624,
    28672,
    30720,
    256,
    2288,
    4320,
    6352,
    8384,
    10416,
    12448,
    14480,
    16512,
    18544,
    20576,
    22608,
    24640,
    26672,
    28704,
    30736,
    512,
    768,
    1280,
    2544,
    2800,
    3312,
    4576,
    4832,
    5344,
    6608,
    6864,
    8640,
    8896,
    10672,
    # Pooler CLS-broadcast / classifier rot-sum（head_he_ct，encoder 不用）
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

DEFAULT_KEYS0 = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys", "keys0")
)


def create_engine(
    params: dict[str, Any] | None = None,
    *,
    verbose: bool = False,
):
    """
    Construct ``CkksEngine``.

    First call without ``scale_primes.pkl`` triggers Liberate prime generation
    (can take many minutes). Subsequent calls read the cache.
    """
    ensure_thor_path()
    p = dict(THOR_DEFAULT_PARAMS if params is None else params)
    return CkksEngine(p, verbose=verbose)


def create_keys(engine, *, with_evk: bool = True):
    """Secret / public / (optional) evaluation keys; wire onto engine."""
    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    engine.add_pk(pk)
    evk = None
    if with_evk:
        evk = engine.create_evk(sk)
        engine.add_evk(evk)
    return {"sk": sk, "pk": pk, "evk": evk}


def load_thor_keys(
    engine,
    keys0: str | None = None,
    *,
    with_bootstrap: bool = True,
    with_rot_deltas: bool = True,
    rot_deltas: list[int] | None = None,
) -> dict[str, Any]:
    """
    Load official THOR ``keys/keys0`` the same way as ``forward.ipynb``.

    - ``rotk_dict`` → ``add_bs_key`` (bootstrap bundle; NOT ordinary rotate keys)
    - ordinary rotates → ``add_rot_keys_from_sk(deltas, sk)``
    - calls ``bs.create_cts_stc_const(engine)`` when ``with_bootstrap``
    """
    ensure_thor_path()
    keys0 = os.path.abspath(keys0 or DEFAULT_KEYS0)
    if not os.path.isdir(keys0):
        raise FileNotFoundError(f"keys0 not found: {keys0}")

    sk = engine.load(os.path.join(keys0, "sk"))
    pk = engine.load(os.path.join(keys0, "pk"))
    evk = engine.load(os.path.join(keys0, "evk"))
    gk = engine.load(os.path.join(keys0, "gk"))
    conjk = engine.load(os.path.join(keys0, "conjk"))
    engine.add_pk(pk)
    engine.add_evk(evk)
    engine.add_gk(gk)
    engine.add_conj_key(conjk)

    bs_key = None
    if with_bootstrap:
        import gc

        import torch
        from liberate.fhe.bootstrapping import ckks_bootstrapping as bs

        rotk_dict = {}
        n = len(ROTK_DICT_KEYS)
        for i, key in enumerate(ROTK_DICT_KEYS, 1):
            path = os.path.join(keys0, "rotk_dict", str(key))
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing bs rotk: {path}")
            rotk_dict[key] = engine.load(path)
            # Fragmentation peaks while uploading ~55×307MB rotks onto 24GB.
            if i % 5 == 0 or i == n:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if i == 1 or i == n or i % 10 == 0:
                used = (
                    torch.cuda.memory_reserved(0) / (1024**3)
                    if torch.cuda.is_available()
                    else float("nan")
                )
                print(
                    f"  loaded rotk_dict {i}/{n} (key={key}) "
                    f"cuda_reserved≈{used:.1f}GiB",
                    flush=True,
                )
        print("  create_cts_stc_const + add_bs_key ...", flush=True)
        bs.create_cts_stc_const(engine)
        engine.add_bs_key(rotk_dict)
        bs_key = rotk_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if with_rot_deltas:
        engine.add_rot_keys_from_sk(
            list(ROT_DELTAS if rot_deltas is None else rot_deltas), sk
        )

    return {
        "sk": sk,
        "pk": pk,
        "evk": evk,
        "gk": gk,
        "conjk": conjk,
        "bs_key": bs_key,
        "keys0": keys0,
    }


def resources_ready() -> bool:
    return os.path.isfile(
        os.path.join(liberate_resources_dir(), "scale_primes.pkl")
    )


def status() -> dict:
    info = diagnose()
    info["resources_ready"] = resources_ready()
    info["keys0_ready"] = os.path.isfile(os.path.join(DEFAULT_KEYS0, "sk"))
    return info
