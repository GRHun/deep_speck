import pickle
from os import urandom
import numpy as np
from speck import *

WORD_SIZE = 16

# 1.urandom
def urandom_test():
    # 返回大小为 size 的字节串，它是适合加密使用的随机字节。
    size = 5 
    n = size
    print(urandom(n))
    print(urandom(n*2))
    

#  2.make_train_data_test
def make_train_data_test():
    n = 5  # 生成10个数据
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
    print("plain0l\n",plain0l,"\nplain0r\n",plain0r,"\nplain1l\n",plain1l,"\nplain1r\n",plain1r)

    num_rand_samples = np.sum(Y==0);
    plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    nr = 5; ks = expand_key(keys, nr);
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
    # X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    print("\nX",[ctdata0l, ctdata0r, ctdata1l, ctdata1r])


# 3.pickle
class Bird(object):
    have_feather = True
    way_of_reproduction  = 'egg'

def test_pickle(Bird):
    summer       = Bird()                 # construct an object
    picklestring = pickle.dumps(summer)   # serialize object
    print(picklestring)

def gen_plain(n):
  pt0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  pt1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  return(pt0, pt1);

# *中立比特测试   
def test_neutal():
    num_struct = 3
    pt0, pt1 = gen_plain(num_struct)
    diff=(0x211, 0xa04)
    neutral_bits = [20,21,22,14,15,23]
    p0 = np.copy(pt0)
    p1 = np.copy(pt1)
    p0 = p0.reshape(-1,1)
    p1 = p1.reshape(-1,1)
  
    for i in neutral_bits:
        d = 1 << i
        d0 = d >> 16
        d1 = d & 0xffff
        p0 = np.concatenate([p0,p0^d0],axis=1)
        p1 = np.concatenate([p1,p1^d1],axis=1)
    p0b = p0 ^ diff[0]; p1b = p1 ^ diff[1]
    return(p0,p1,p0b,p1b)

def test():
    pass

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
    return(ks);

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

if __name__=="__main__":
    # urandom_test()
    # make_train_data_test()
    # test_pickle(Bird)
    # test_neutal()
    # test_low_weight()

    test_expand_key((0x1918,0x1110,0x0908,0x0100), 5)


