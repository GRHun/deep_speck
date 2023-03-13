import numpy as np
from os import urandom
import speck as sp
from math import sqrt, log, log2


# convert_to_binary takes as input an array of ciphertext pairs
# where the first row of the array contains the lefthand side of the ciphertexts,
# the second row contains the righthand side of the ciphertexts,
# the third row contains the lefthand side of the second ciphertexts,
# and so on
# it returns an array of bit vectors containing the same data
def WORD_SIZE():
    return 16

def convert_to_binary(arr):
    X = np.zeros((4 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    # zeros(shape, dtype=float, order='C')
    for i in range(4 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    # 对于二维ndarray，transpose在不指定参数是默认是矩阵转置
    X = X.transpose()
    return (X)

def make_structure(pt0, pt1, diff=(0x211, 0xa04), neutral_bits=[20, 21, 22, 14, 15]):
    """生成明文结构
    @para:  (pt0,pt1) - 明文对
    @example: 
        pt0, pt1 = gen_plain(n)
        pt0a, pt1a, pt0b, pt1b = make_structure(pt0, pt1, diff=diff, neutral_bits=neutral_bits)
    """
    p0 = np.copy(pt0)
    p1 = np.copy(pt1)
    # 拉成一列
    p0 = p0.reshape(-1, 1)
    p1 = p1.reshape(-1, 1)

    # i 需要区分大于16和小于16的情况
    for i in neutral_bits:
        d = 1 << i
        d0 = d >> 16
        d1 = d & 0xffff
        p0 = np.concatenate([p0, p0 ^ d0], axis=1)
        p1 = np.concatenate([p1, p1 ^ d1], axis=1)
    p0b = p0 ^ diff[0]
    p1b = p1 ^ diff[1]
    return (p0, p1, p0b, p1b)

def gen_key(nr):
  """
  generate a Speck key, return expanded key for nr rounds
  """
  key = np.frombuffer(urandom(8), dtype=np.uint16)
  ks = sp.expand_key(key, nr)
  return (ks)

def gen_plain(n):
    """
    生成明文对(pt0, pt1)
    @para:  n - 明文对个数
    """
    pt0 = np.frombuffer(urandom(2*n), dtype=np.uint16)
    pt1 = np.frombuffer(urandom(2*n), dtype=np.uint16)
    return (pt0, pt1)

# * baseline training data generator
def make_train_data(n, nr, diff=(0x0040, 0)):
    """
    生成n个nr轮的训练数据
    @para:  n   - 生成的数据数量(danw)
            nr  - 加密轮数
    """
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    print("===> Y when n =\n",n,Y)  #  例子：n=4 [160  92 216 193]
    # numpy.frombuffer 用于实现动态数组。接受 buffer 输入参数，以流的形式读入转化成 ndarray 对象。
    # numpy.frombuffer(buffer, dtype = float[返回数组的数据类型], count = -1, offset = 0)
    # urandom(n) Return a string of n random bytes suitable for cryptographic use
    # 一个字节存储8位无符号数，储存的数值范围为0-255
    Y = Y & 1  # 取最后一位，Y变成长为n的0，1的numpy.ndarray

    # 随机生成主密钥，满足固定差分diff的明文对（plain0，plain1）
    keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4, -1)  #-1表示列数是自动计算的，划为4行
    print("===> keys when n =\n", n,keys)
    plain0l = np.frombuffer(urandom(2*n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2*n), dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    print("===> plain0l when n =", n,plain0l)
    print("===> plain0r when n =", n,plain0r)

    num_rand_samples = np.sum(Y == 0)
    # 如果Y=0，就用新生成的随机明文代替Plain1（第二个明文）
    plain1l[Y == 0] = np.frombuffer(
        urandom(2*num_rand_samples), dtype=np.uint16)
    plain1r[Y == 0] = np.frombuffer(
        urandom(2*num_rand_samples), dtype=np.uint16)

    # 然后对所有的明文对进行nr轮加密
    ks = sp.expand_key(keys, nr)   # 生成nr个字密钥
    ctdata0l, ctdata0r = sp.encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = sp.encrypt((plain1l, plain1r), ks)
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    return (X, Y)

def gen_challenge(n, nr, diff=(0x211, 0xa04), neutral_bits=[20, 21, 22], keyschedule='real'):
    """
    生成n个明文结构，再密钥0向上解密一轮，最后返回加密了 nr轮的密文结构和加密密钥（list"""
    pt0, pt1 = gen_plain(n)
    # print("====>pt0,pt1\n",pt0,pt1)
    pt0a, pt1a, pt0b, pt1b = make_structure(pt0, pt1, diff=diff, neutral_bits=neutral_bits)
    # print("====>pt0a,pt1a\n",pt0a,pt1a)
    # print("====>pt0b,pt1b\n",pt0b,pt1b)
    pt0a, pt1a = sp.dec_one_round((pt0a, pt1a), 0)
    pt0b, pt1b = sp.dec_one_round((pt0b, pt1b), 0)

    key = gen_key(nr)
    if (keyschedule == 'free'):
        key = np.frombuffer(urandom(2*nr), dtype=np.uint16)
    ct0a, ct1a = sp.encrypt((pt0a, pt1a), key)
    ct0b, ct1b = sp.encrypt((pt0b, pt1b), key)
    return ([ct0a, ct1a, ct0b, ct1b], key)

if __name__=="__main__":
    n=5

    nr = 5
    # make_train_data(n, nr, diff=(0x0040, 0))
    # low_weight = np.array(range(2**WORD_SIZE()), dtype=np.uint16)
    # print(low_weight)
    # num_cand = 3
    # keys = np.random.choice(2**(WORD_SIZE()-2), num_cand, replace=False)
    # print(keys)
    # ct ,key = gen_challenge(n, nr)
    # print("====>ct\n",ct)
    # print("====>ct[0]\n",ct[0])
    # print("===>n",len(ct[0]))
    # print("===>len(ct[0][0]))",len(ct[0][0]))
    # r = np.random.randint(0, 4, 7, dtype=np.uint16)
    # # randint(low, high=None, size=None, dtype=int) Return random integers from low (inclusive) to high (exclusive).
    # # 生成的数值在[low, high)区间
    # r = r << 14
    # print("===>r when num_cand=7\n",r)
    print(gen_plain(3))
