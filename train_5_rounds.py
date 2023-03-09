"""
a basic training script that will train a 10-block deep residual network to distinguish 5-round Speck 
from random data using the basic training pipeline described in the paper"""

import train_nets as tn

tn.train_speck_distinguisher(200,num_rounds=5,depth=10);
