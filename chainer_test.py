# coding: utf-8

import os
import random
import re

import numpy as np
import chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L



class NeuralNet(chainer.Chain):
    def __init__(self, n_units, n_out):
        initializer = chainer.initializers.HeNormal()
        # initializer = chainer.initializers.Constant(0.5)
        super().__init__(
            l1=L.Linear(None, n_units, initialW=initializer),
            l2=L.Linear(n_units, n_units, initialW=initializer),
            l3=L.Linear(n_units, n_out, initialW=initializer),
        )
        # super().__init__(
        #     l1=L.Linear(None, n_units ),
        #     l2=L.Linear(n_units, n_units ),
        #     l3=L.Linear(n_units, n_out ),
        # )


    def __call__(self, x):
        # h1 = F.relu(self.l1(x))
        # h2 = F.relu(self.l2(h1))
        activate_func = F.leaky_relu
        h1 = activate_func(self.l1(x))
        h2 = activate_func(self.l2(h1))
        return self.l3(h2)

def check_accuracy(model, xs, ts):
    ys = model(xs)
    loss = F.softmax_cross_entropy(ys, ts)
    ys = np.argmax(ys.data, axis=1)
    cors = (ys == ts)
    num_cors = sum(cors)
    accuracy = num_cors / ts.shape[0]
    return accuracy, loss

def set_random_seed(seed):
    # set Python random seed
    random.seed(seed)

    # set NumPy random seed
    np.random.seed(seed)

    # set Chainer(CuPy) random seed    
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)
    print("seed = ", seed )
    return

    

def train( model, model_save_dir,
           xs, ts, txs, tts ):
    epoch_num = 20
    bn = 100
    # optimizer = chainer.optimizers.SGD(lr=0.01)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    if not os.path.exists( model_save_dir ):
        os.mkdir( model_save_dir )
    best_acc = None    
    itr_num = len(xs) // bn
    for i in range(epoch_num):
        for j in range(itr_num):
            start, end = j * bn, (j + 1) * bn
            x = xs[start:end]
            t = ts[start:end]
            t = Variable(np.array(t, "i"))  # int 値に変換.

            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            
            # model.zerograds()
            model.cleargrads()
            loss.backward()
            optimizer.update()

        accuracy_train, loss_train = check_accuracy(model, xs, ts)
        accuracy_test, loss_test = check_accuracy(model, txs, tts)
        if best_acc == None or best_acc < accuracy_test:
            best_acc = accuracy_test
            save_path = os.path.join( model_save_dir, "my_model_%d.npz" % i )
            chainer.serializers.save_npz( save_path, model, compression=True)
        output_str = \
            "Epoch %d\n  loss(train) = %.8f, accuracy(train) = %.8f\n  loss(test)  = %.8f, accuracy(test)  = %.8f" \
            % (i + 1,
               loss_train.data, accuracy_train, loss_test.data, accuracy_test )

        print( output_str )
        # print(model.l1.W.array[0][300] )
        #exit(1)
    print( "best_acc = ", best_acc)
    return

def predict( model_save_dir, weight_name,
             model, txs, tts ):
    weight_path = os.path.join( model_save_dir, weight_name )
    chainer.serializers.load_npz( weight_path, model )
    accuracy_test, loss_test = check_accuracy( model, txs, tts)
    output_str = \
        "loss = %.8f, accuracy = %.8f" % ( loss_test.data, accuracy_test )
    print( output_str )
    return

def get_best_weight_path( model_save_dir ):
    best_itr = None
    p = re.compile(r"my_model_(\d+)\.npz")
    for fname in os.listdir(model_save_dir):
        m = p.match(fname)
        if m:
            itr = int(m.group(1))
            if best_itr == None or best_itr < itr:
                best_itr = itr
    if best_itr:
        return "my_model_" + str(best_itr) + ".npz"
    else:
        return None

def main( is_train ):
    model_save_dir = "save_dir"
    n_units, n_out = 50, 10
    seed = 1
    set_random_seed(seed)  # network 作成前にシードを設定する必要がある.
    model = NeuralNet( n_units, n_out )
    train_data, test_data = chainer.datasets.get_mnist()
    xs, ts = train_data._datasets
    txs, tts = test_data._datasets
    
    if is_train:
        train( model, model_save_dir,
               xs, ts, txs, tts )
    else:
        weight_name = get_best_weight_path( model_save_dir )
        print( "weight_name = ", weight_name )
        predict( model_save_dir, weight_name,
                 model, txs, tts )
    return

if __name__ == '__main__':
    # is_train = True
    is_train = False
    main(is_train)

