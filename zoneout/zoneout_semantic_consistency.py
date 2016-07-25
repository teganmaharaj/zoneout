import theano
import theano.tensor as T
import numpy
np = numpy
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dim', default=2, type=int)
parser.add_argument('--seq_len', default=10, type=int)
parser.add_argument('--nsteps', default=100000, type=int)
parser.add_argument('--noise_level', default=.01, type=float)
parser.add_argument('--zoneout_prob', default=.1, type=float)
parser.add_argument('--learning_rate', default=.0001, type=float)
args = parser.parse_args()
locals().update(args.__dict__)

learning_curves = []
for zoneout_p in [zoneout_prob, 0]:
    lr = theano.shared(np.float32(learning_rate))
    x = T.matrix()
    zoneout_mask = T.matrix()
    W = theano.shared((-np.eye(dim) + noise_level * np.random.randn(dim,dim)).astype("float32"))
    new_h = x
    for i in range(seq_len):
        old_h = new_h
        new_h = T.dot(old_h, W)
        new_h = old_h * (1 - zoneout_mask[i]) + new_h * zoneout_mask[i]
    cost = ((new_h - x)**2).sum()
    train_fn = theano.function([x, zoneout_mask], cost, updates = {W: W - lr * T.grad(cost, W)})
    cost_fn = theano.function([x, zoneout_mask], cost)

    learning_curve = []
    for n in range(nsteps):
        examples = np.random.randn(32, dim).astype("float32") 
        mask = np.random.binomial(1, 1 - zoneout_p, (seq_len, dim)).astype("float32")
        if n % (nsteps / 100) == 0:
            print cost_fn(examples, mask)
            print W.get_value()
        learning_curve.append(train_fn(examples, mask))
    learning_curves.append(learning_curve)
