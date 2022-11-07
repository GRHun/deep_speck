import speck as sp 
import pickle

#
def mod_add():
    WORD_SIZE=16
    MASK_VAL = 2 ** WORD_SIZE - 1
    c0 = 23
    c1 = 324

    mod_add1 = (c0 + c1) & MASK_VAL
    mod_add2 = (c0+c1) % (2 ** WORD_SIZE)

    print("mod_add1: ",mod_add1)
    print("\nmod_add2: ",mod_add2)

def one_round_differential(in_dif):
    p1 = (12,2)
    p2 = (p1[0]^in_dif[0], p1[1]^in_dif[1])
    k = 0
    out1 = sp.enc_one_round(p1,k)
    out2 = sp.enc_one_round(p2,k)
    print(out1, out2)
    out_diff = (out1[0]^out2[0], out1[1]^out2[1])
    print(hex(out_diff[0]), hex(out_diff[1]))




if __name__=="__main__":
    # mod_add()
    in_dif = (0x0040, 0x0000)
    one_round_differential(in_dif)
