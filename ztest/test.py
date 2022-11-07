import numpy as np
from os import urandom


wdir = './freshly_trained_nets/'

def train_speck_distinguisher(num_epochs, num_rounds=7, depth=1):
    #create the network
    net = make_resnet(depth=depth, reg_param=10**-5);
    net.compile(optimizer='adam',loss='mse',metrics=['acc']);
    #generate training and validation data
    X, Y = sp.make_train_data(10**7,num_rounds);
    X_eval, Y_eval = sp.make_train_data(10**6, num_rounds);
    #set up model checkpoint
    check = make_checkpoint(wdir+'best'+str(num_rounds)+'depth'+str(depth)+'.h5');
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));
    #train and evaluate
    h = net.fit(X,Y,epochs=num_epochs,batch_size=bs,validation_data=(X_eval, Y_eval), callbacks=[lr,check]);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_acc']);
    np.save(wdir+'h'+str(num_rounds)+'r_depth'+str(depth)+'.npy', h.history['val_loss']);
    dump(h.history,open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth)+'.p','wb'));
    print("Best validation accuracy: ", np.max(h.history['val_acc']));
    return(net, h);

def make_train_data(n, nr, diff=(0x0040,0)):
    Y = np.frombuffer(urandom(n), dtype=np.uint8); 
    Y = Y & 1;
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
    num_rand_samples = np.sum(Y==0);
    plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    ks = expand_key(keys, nr);
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
    return(X,Y);

def test_func():
    n = 10  # 生成是个数据
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    print(Y)

if __name__=="__main__":
    test_func()
