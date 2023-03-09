import pickle
from os import urandom
import numpy as np
from speck import *

WORD_SIZE = 16


def urandom_test():
    # 返回大小为 size 的字节串，它是适合加密使用的随机字节。
    size = 5 
    n = size
    print(urandom(n))
    print(urandom(n*2))
    
def make_train_data_test():
    n = 6  # 生成n个数据
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    print("Y:\t", Y)
    Y_1 = Y & 1
    print("Y & 1:\t",Y_1)
    # print(type(Y_1))
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
    print("keys\t", keys)

    diff=(0x0040,0)
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
    # print("plain0l\n",plain0l,"\nplain0r\n",plain0r,"\nplain1l\n",plain1l,"\nplain1r\n",plain1r)

    num_rand_samples = np.sum(Y==0);
    plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    nr = 5; ks = expand_key(keys, nr);
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
    # X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    # print("\nX",[ctdata0l, ctdata0r, ctdata1l, ctdata1r])

class Bird(object):
    have_feather = True
    way_of_reproduction  = 'egg'
def test_pickle(Bird):
    summer       = Bird()                 # construct an object
    picklestring = pickle.dumps(summer)   # serialize object
    print(picklestring)

def test_gen_small_plain(n):
  pt0 = np.frombuffer(urandom(n),dtype=np.uint8)
  pt1 = np.frombuffer(urandom(n),dtype=np.uint8)
#   print("pt0:\t",pt0)
#   print("pt1:\t",pt1)
  return(pt0, pt1)

def test_gen_plain(n):
  pt0 = np.frombuffer(urandom(2*n),dtype=np.uint16)
  pt1 = np.frombuffer(urandom(2*n),dtype=np.uint16)
  print("pt0:\t",pt0)
  print("pt1:\t",pt1)
#   print(convert_to_binary(pt0))
  return(pt0, pt1)

def make_train_data(n, nr, diff=(0x0040, 0)):
    """
    生成训练数据, 返回(X,Y), 16位的二进制([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    @para:  n   - 生成的数据数量
            nr  - 加密轮数
    """
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    # numpy.frombuffer 用于实现动态数组。接受 buffer 输入参数，以流的形式读入转化成 ndarray 对象。
    # numpy.frombuffer(buffer, dtype = float[返回数组的数据类型], count = -1, offset = 0)
    # urandom(n),返回大小为 n 的字节串，它是适合加密使用的随机字节。
    Y = Y & 1  # 取最后一位，Y变成长为n的0，1的numpy.ndarray

    # 随机生成主密钥，满足固定差分diff的明文对（plain0，plain1）
    keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4, -1)
    plain0l = np.frombuffer(urandom(2*n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2*n), dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]

    num_rand_samples = np.sum(Y == 0)
    # 如果Y=0，就用新生成的随机明文代替Plain1（第二个明文）
    plain1l[Y == 0] = np.frombuffer(
        urandom(2*num_rand_samples), dtype=np.uint16)
    plain1r[Y == 0] = np.frombuffer(
        urandom(2*num_rand_samples), dtype=np.uint16)

    ks = expand_key(keys, nr)   # 生成nr个字密钥
    # 然后对所有的明文对进行nr轮加密
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    return (X, Y)


def test_make_struct(n, neutral_bits=[1, 18, 2], diff=(0x211, 0xa04)):
    p0,p1 = test_gen_plain(n)
    p0 = p0.reshape(-1, 1)
    p1 = p1.reshape(-1, 1)
    print("p0\n",p0)
    print("p1\n",p1)

    # i 需要区分大于16和小于16的情况
    for i in neutral_bits:
        print("\n====>",i)
        d = 1 << i
        d0 = d >> 16
        d1 = d & 0xffff
        print("d0:\t",d0)
        print("d1:\t",d1)
        p0 = np.concatenate([p0, p0 ^ d0], axis=1)
        p1 = np.concatenate([p1, p1 ^ d1], axis=1)
        print("now p0 is:\n",p0)
        print("now p1 is:\n",p1)
    p0b = p0 ^ diff[0]
    p1b = p1 ^ diff[1]
    return (p0, p1, p0b, p1b)

def hw(v):
  res = np.zeros(v.shape,dtype=np.uint8)
  for i in range(16):
    res = res + ((v >> i) & 1)
  return(res)

def test_low_weight():
    low_weight = np.array(range(2**WORD_SIZE), dtype=np.uint16)  # [0 1 2 ... 65533 65534 65535]
    # print(hw(low_weight))
    low_weight = low_weight[hw(low_weight) <= 2]
    print(low_weight)

def test_expand_key(k, t):
    """
    密钥拓展算法
    @para:  k: mian key - 密钥长度为 64 = 4*16,形如key = (0x1918,0x1110,0x0908,0x0100)
            t: num of round 
    """
    ks = [0 for i in range(t)];
    ks[0] = k[len(k)-1];
    l = list(reversed(k[:len(k)-1]));
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i);
    

    # print("k0\t", k)
    # ks = [0 for i in range(t)]
    # print("ks1\t",ks)
    # ks[0] = k[len(k)-1]
    # print("ks2\t",ks)
    # print("k\t",k)
    # print("k1\t",k[:len(k)-1])
    # l = list(reversed(k[:len(k)-1]))
    # print("l\t",l)
    # for i in range(t-1):
    #     l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i)
    # return(ks);
    return(ks);

def test_make_structure(pt0, pt1, diff=(0x211, 0xa04), neutral_bits=[1, 18, 22, 14, 15]):
    """生成明文结构
    @para:  (pt0,pt1) - 明文对
    @example: 
        pt0, pt1 = gen_plain(n)
        pt0a, pt1a, pt0b, pt1b = make_structure(pt0, pt1, diff=diff, neutral_bits=neutral_bits)
    """
    p0 = np.copy(pt0)
    p1 = np.copy(pt1)
    p0 = p0.reshape(-1, 1)
    p1 = p1.reshape(-1, 1)
    print("p0\n",p0)
    print("p1\n",p1)
    # i 需要区分大于16和小于16的情况
    for i in neutral_bits:
        print("\n====>",i)
        d = 1 << i
        d0 = d >> 16
        d1 = d & 0xffff
        p0 = np.concatenate([p0, p0 ^ d0], axis=1)
        p1 = np.concatenate([p1, p1 ^ d1], axis=1)
        print("now p0 is:\n",p0)
        print("now p1 is:\n",p1)
    p0b = p0 ^ diff[0]
    p1b = p1 ^ diff[1]
    return (p0, p1, p0b, p1b)

def test_linalg_norm():
    # np.linalg.norm(
    # x[矩阵], ord=None[范数类型], 
    # axis=None[1表示按行向量处理，求多个行向量的范数], 
    # keepdims=False[不保留二维特性])
    v = np.array([[3,4,],
                [1,1]])
    res = np.linalg.norm(v, axis=1)  # axis=1表示对矩阵的每一行求范数
    print("res",res)

def test():
    pass


if __name__=="__main__":
    # urandom_test()
    # make_train_data_test()
    # test_pickle(Bird)
    # test_neutal()
    # test_low_weight()
    # test_gen_small_plain(n=10)
    # test_gen_plain(n=10)
    # test_expand_key((0x1918,0x1110,0x0908,0x0100), 5)
    
    n = 5
    a, b = test_gen_plain(n)
    print(test_make_structure(a,b))
    # test()
    # test_linalg_norm()
