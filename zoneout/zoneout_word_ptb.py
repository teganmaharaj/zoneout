import argparse
import time
import os
import sys
import logging

import numpy
import numpy as np

import theano
import theano.tensor as T

from blocks.algorithms import (GradientDescent, StepClipping, CompositeRule,
                               Momentum, Adam, RMSProp, StepRule, Scale)
from blocks.bricks import Tanh, Softmax, Linear, Rectifier
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks.recurrent import LSTM, GatedRecurrent, SimpleRecurrent
from blocks.extensions import FinishAfter, Printing, ProgressBar, SimpleExtension, TrainingExtension, CompositeExtension
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.extensions.saveload import Load
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.monitoring import aggregation, evaluators
from blocks.model import Model
from blocks.roles import WEIGHT, OUTPUT

from bricks.recurrent import MEMORY_CELL
from bricks.encoders import DropMultiLayerEncoder
from bricks.recurrent import DropBidirectionalGraves, DropLSTM, DropGRU, DropSimpleRecurrent
from collections import OrderedDict
#import ctc
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialExampleScheme
from datasets.timit import Timit
from datasets.transformers import Transpose
from extensions import SaveLog, SaveParams
from initialization import NormalizedInitialization, OrthogonalInitialization
from utils import SampleDropsNPWord, zoneouts_syntax_help_string
from theano.compile.nanguardmode import NanGuardMode
import commands


floatX = theano.config.floatX
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

# stolen from https://github.com/jelennal/t1t2/blob/master/main.py#L116
def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], np.random.RandomState) and 
           not (hasattr(node, 'op') or isinstance(node.op, (theano.sandbox.rng_mrg.GPU_mrg_uniform, theano.sandbox.cuda.basic_ops.GpuAllocEmpty)))):
            try:
                has_nans = np.isnan(output[0]).any() or np.isinf(output[0]).any()
            except TypeError:
                has_nans = False
            if not has_nans:
                continue           
            print('*** NaN detected ***')
            theano.printing.debugprint(node, depth=3)
            print(type(node), node.op, type(node.op))
            print('Inputs : %s' % [input[0] for input in fn.inputs])
            print'Input shape',  [input[0].shape for input in fn.inputs]
            print('Outputs: %s' % [output[0] for output in fn.outputs])
            print'Output shape',  [output[0].shape for output in fn.outputs]
            print 'NaN # :', [np.sum(np.isnan(output[0])) for output in fn.outputs]  
            print 'Inf # :', [np.sum(np.isinf(output[0])) for output in fn.outputs]  
            print 'NaN location: ', np.argwhere(np.isnan(output[0])), ', Inf location: ', np.argwhere(np.isinf(output[0]))            
            import pdb; pdb.set_trace()
            raise ValueError


def learning_algorithm(args):
    name = args.algorithm
    learning_rate = float(args.learning_rate)
    momentum = args.momentum
    clipping_threshold = args.clipping
    clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
    if name == 'adam':
        adam = Adam(learning_rate=learning_rate)
        step_rule = CompositeRule([adam, clipping])
        learning_rate = adam.learning_rate
    elif name == 'rms_prop':
        rms_prop = RMSProp(learning_rate=learning_rate)
        step_rule = CompositeRule([clipping, rms_prop])
        learning_rate = rms_prop.learning_rate
    elif name == 'momentum':
        sgd_momentum = Momentum(learning_rate=learning_rate, momentum=momentum)
        step_rule = CompositeRule([clipping, sgd_momentum])
        learning_rate = sgd_momentum.learning_rate
    elif name == 'sgd':
        sgd = Scale(learning_rate=learning_rate)
        step_rule = CompositeRule([clipping, sgd])
        learning_rate = sgd.learning_rate
    else:
        raise NotImplementedError
    return step_rule, learning_rate

def parse_args():
    parser = argparse.ArgumentParser(description='PTB experiment')
    parser.add_argument('--experiment_path', type=str,
                        default='./3LSTM_PTB',
                        help='Location for writing results')
    parser.add_argument('--layer_size', type=int,
                        default=256,
                        help='States dimensions')
    parser.add_argument('--epochs', type=int,
                        default=2000,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', default=2e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_noise', type=float, default=0,
                        help='stdv of weight noise')
    parser.add_argument('--momentum',
                        default=0.99,
                        type=float,
                        help='Momentum for SGD')
    parser.add_argument('--clipping',
                        default=10,
                        type=float,
                        help='Gradient clipping norm')
    parser.add_argument('--decrease_lr_after_epoch',
                        default=None,
                        type=float,
                        help='Epoch after which to start decreasing the learning rate (must separately define lr_decay)')
    parser.add_argument('--lr_decay',
                        default=None,
                        type=float,
                        help='After decrease_lr_after_epoch number of epochs, the lr will be divided by this number')
    parser.add_argument('--test_cost', action='store_true',
                        default=False,
                        help='Report test set cost')
    parser.add_argument('--algorithm', choices=['rms_prop', 'adam',
                                                'momentum', 'sgd'],
                        default='adam',
                        help='Learning algorithm to use')
    parser.add_argument('--initialization', choices=['glorot', 'uniform', 'identity', 'ortho'],
                        default='uniform')
    parser.add_argument('--init-width', type=float,
                        default=None, help='width of the uniform initialization')
    parser.add_argument('--z_prob',
                        type=str,
                        default=None,
                        help='(for rnn with shared mask and for SRNN/GRU) Despite the name, is the update probability. Sorry.' +
                        zoneouts_syntax_help_string)
    parser.add_argument('--z_prob_states',
                        type=str,
                        default=None,
                        help='(for rnn with non-shared mask) Despite the name, is the update probability. Sorry.' +
                        zoneouts_syntax_help_string)
    parser.add_argument('--z_prob_cells',
                        type=str,
                        default=None,
                        help='(for rnn with non-shared mask) Despite the name, is the update probability. Sorry.' +
                        zoneouts_syntax_help_string)
    parser.add_argument('--drop_prob_igates',
                        type=str,
                        default='1',
                        help='(for rnn with drop on input gate ala Recurrent dropout without memory loss paper.' +
                        zoneouts_syntax_help_string)
    parser.add_argument('--ogates_zoneout', action='store_true',
                        default=False,
                        help='Zone out the output gates reusing the masks from the input gate dropout')
    parser.add_argument('--stoch_depth', action='store_true',
                        default=False,
                        help='Use stochastic depth, i.e. zoneout on full cell/state layers')
    parser.add_argument('--share_mask', action='store_true',
                        default=False,
                        help='Use the same mask for cells and states')
    parser.add_argument('--gaussian_drop', action='store_true',
                        default=False,
                        help='Use a Gaussian distribution for the dropout masks')
    parser.add_argument('--rnn_type',
                        default='LSTM',
                        help='options: LSTM, GRU, SRNN')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='(future) only works for 1 atm')
    parser.add_argument('--norm_cost_coeff',
                        type=float,
                        default=0,
                        help='options: LSTM, GRU, SRNN')
    parser.add_argument('--penalty', type=str, default='hids',
                        choices=['hids', 'cells'])
    parser.add_argument('--seq_len', type=int, default='35', help='for chopping up ptb')
    parser.add_argument('--testing', action='store_true', default=False, help='testing?')
    parser.add_argument('--batch_size', type=int, default=20)
    return parser.parse_args()



def train(algorithm, learning_rate, clipping, momentum,
          layer_size, epochs, test_cost, experiment_path,
          initialization, init_width, weight_noise, z_prob,
          z_prob_states, z_prob_cells, drop_prob_igates, ogates_zoneout,
          batch_size, stoch_depth, share_mask, gaussian_drop, rnn_type,
          num_layers, norm_cost_coeff, penalty, testing, seq_len, 
          decrease_lr_after_epoch, lr_decay,
          **kwargs):

    print '.. PTB experiment'
    print '.. arguments:', ' '.join(sys.argv)
    t0 = time.time()

    ###########################################
    #
    # LOAD DATA
    #
    ###########################################

    def onehot(x, numclasses=None):
        """ Convert integer encoding for class-labels (starting with 0 !)
            to one-hot encoding.
            The output is an array whose shape is the shape of the input array
            plus an extra dimension, containing the 'one-hot'-encoded labels.
        """
        if x.shape == ():
            x = x[None]
        if numclasses is None:
            numclasses = x.max() + 1
        result = numpy.zeros(list(x.shape) + [numclasses], dtype="int")
        z = numpy.zeros(x.shape, dtype="int")
        for c in range(numclasses):
            z *= 0
            z[numpy.where(x == c)] = 1
            result[..., c] += z
        return result.astype(theano.config.floatX)

    alphabetsize = 10000
    data = np.load('penntree_char_and_word.npz')
    trainset = data['train_words']
    validset = data['valid_words']
    testset = data['test_words']

    if testing:
        trainset = trainset[:3000]
        validset = validset[:3000]

    if share_mask:
        if not z_prob:
            raise ValueError('z_prob must be provided when using share_mask')
        if z_prob_cells or z_prob_states:
            raise ValueError('z_prob_states and z_prob_cells must not be provided when using share_mask (use z_prob instead)')
        z_prob_cells = z_prob
        # we don't want to actually use these masks, so this is to debug
        z_prob_states = None
    else:
        if z_prob:
            raise ValueError('z_prob is only used with share_mask')
        z_prob_cells = z_prob_cells or '1'
        z_prob_states = z_prob_states or '1'

#    rng = np.random.RandomState(seed)



    ###########################################
    #
    # MAKE STREAMS
    #
    ###########################################

    def prep_dataset(dataset):
        dataset = dataset[:(len(dataset) - (len(dataset) % (seq_len * batch_size)))]
        dataset = dataset.reshape(batch_size, -1, seq_len).transpose((1, 0, 2))


        stream = DataStream(IndexableDataset(indexables=OrderedDict([
            ('data', dataset)])),
            iteration_scheme=SequentialExampleScheme(dataset.shape[0]))
        stream = Transpose(stream, [(1, 0)])
        stream = SampleDropsNPWord(
          stream, z_prob_states, z_prob_cells, drop_prob_igates,
          layer_size, num_layers, False, stoch_depth, share_mask,
          gaussian_drop, alphabetsize)
        stream.sources = ('data',) * 3 + stream.sources + ('zoneouts_states', 'zoneouts_cells', 'zoneouts_igates')
        return (stream,)
    train_stream, = prep_dataset(trainset)
    valid_stream, = prep_dataset(validset)
    test_stream, = prep_dataset(testset)


    ####################


    data = train_stream.get_epoch_iterator(as_dict=True).next()


    ####################


    ###########################################
    #
    # BUILD MODEL
    #
    ###########################################
    print '.. building model'

    x = T.tensor3('data')
    y = x
    zoneouts_states = T.tensor3('zoneouts_states')
    zoneouts_cells = T.tensor3('zoneouts_cells')
    zoneouts_igates = T.tensor3('zoneouts_igates')

    x.tag.test_value = data['data']
    zoneouts_states.tag.test_value = data['zoneouts_states']
    zoneouts_cells.tag.test_value = data['zoneouts_cells']
    zoneouts_igates.tag.test_value = data['zoneouts_igates']

    if init_width and not initialization == 'uniform':
        raise ValueError('Width is only for uniform init, whassup?')

    if initialization == 'glorot':
        weights_init = NormalizedInitialization()
    elif initialization == 'uniform':
        weights_init = Uniform(width=init_width)
    elif initialization == 'ortho':
        weights_init = OrthogonalInitialization()
    else:
        raise ValueError('No such initialization')

    if rnn_type.lower() == 'lstm':
        in_to_hids = [Linear(layer_size if l > 0 else alphabetsize, layer_size*4, name='in_to_hid%d'%l,
                           weights_init=weights_init, biases_init=Constant(0.0)) for l in range(num_layers)]
        recurrent_layers = [DropLSTM(dim=layer_size, weights_init=weights_init, activation=Tanh(), model_type=6, name='rnn%d'%l, ogates_zoneout=ogates_zoneout) for l in range(num_layers)]
    elif rnn_type.lower() == 'gru':
        in_to_hids = [Linear(layer_size if l > 0 else alphabetsize, layer_size*3, name='in_to_hid%d'%l,
                           weights_init=weights_init, biases_init=Constant(0.0)) for l in range(num_layers)]
        recurrent_layers = [DropGRU(dim=layer_size, weights_init=weights_init, activation=Tanh(), name='rnn%d'%l) for l in range(num_layers)]
    elif rnn_type.lower() == 'srnn':  # FIXME!!! make ReLU
        in_to_hids = [Linear(layer_size if l > 0 else alphabetsize, layer_size, name='in_to_hid%d'%l,
                           weights_init=weights_init, biases_init=Constant(0.0)) for l in range(num_layers)]
        recurrent_layers = [DropSimpleRecurrent(dim=layer_size, weights_init=weights_init, activation=Rectifier(), name='rnn%d'%l) for l in range(num_layers)]
    else:
        raise NotImplementedError

    hid_to_out = Linear(layer_size, alphabetsize, name='hid_to_out',
                        weights_init=weights_init, biases_init=Constant(0.0))

    for layer in in_to_hids:
        layer.initialize()
    for layer in recurrent_layers:
        layer.initialize()
    hid_to_out.initialize()

    layer_input = x #in_to_hid.apply(x)

    init_updates = OrderedDict()
    for l, (in_to_hid, layer) in enumerate(zip(in_to_hids, recurrent_layers)):
        rnn_embedding = in_to_hid.apply(layer_input)
        if rnn_type.lower() == 'lstm':
            states_init = theano.shared(np.zeros((batch_size, layer_size), dtype=floatX))
            cells_init = theano.shared(np.zeros((batch_size, layer_size), dtype=floatX))
            states_init.name, cells_init.name = "states_init", "cells_init"
            states, cells = layer.apply(rnn_embedding,
                                        zoneouts_states[:, :, l * layer_size : (l + 1) * layer_size],
                                        zoneouts_cells[:, :, l * layer_size : (l + 1) * layer_size],
                                        zoneouts_igates[:, :, l * layer_size : (l + 1) * layer_size],
                                        states_init,
                                        cells_init)
            init_updates.update([(states_init, states[-1]), (cells_init, cells[-1])])
        elif rnn_type.lower() in ['gru', 'srnn']:
            # untested!
            states_init = theano.shared(np.zeros((batch_size, layer_size), dtype=floatX))
            states_init.name = "states_init"
            states = layer.apply(rnn_embedding, zoneouts_states, zoneouts_igates, states_init)
            init_updates.update([(states_init, states[-1])])
        else:
            raise NotImplementedError
        layer_input = states

    y_hat_pre_softmax = hid_to_out.apply(T.join(0, [states_init], states[:-1]))
    shape_ = y_hat_pre_softmax.shape
    y_hat = Softmax().apply(
        y_hat_pre_softmax.reshape((-1, alphabetsize)))

    ####################


    ###########################################
    #
    # SET UP COSTS AND MONITORS
    #
    ###########################################

    cost = CategoricalCrossEntropy().apply(y.reshape((-1, alphabetsize)), y_hat).copy('cost')

    bpc = (cost/np.log(2.0)).copy(name='bpr')
    perp = T.exp(cost).copy(name='perp')


    cost_train = cost.copy(name='train_cost')
    cg_train = ComputationGraph([cost_train])


    ###########################################
    #
    # NORM STABILIZER
    #
    ###########################################
    norm_cost = 0.

    def _magnitude(x, axis=-1):
        return T.sqrt(T.maximum(T.sqr(x).sum(axis=axis), numpy.finfo(x.dtype).tiny))

    if penalty == 'cells':
        assert VariableFilter(roles=[MEMORY_CELL])(cg_train.variables)
        for cell in VariableFilter(roles=[MEMORY_CELL])(cg_train.variables):
            norms = _magnitude(cell)
            norm_cost += T.mean(T.sum((norms[1:] - norms[:-1])**2, axis=0) / (seq_len - 1))
    elif penalty == 'hids':
        for l in range(num_layers):
            assert 'rnn%d_apply_states'%l in [o.name for o in VariableFilter(roles=[OUTPUT])(cg_train.variables)]
        for output in VariableFilter(roles=[OUTPUT])(cg_train.variables):
            for l in range(num_layers):
                if output.name == 'rnn%d_apply_states'%l:
                    norms = _magnitude(output)
                    norm_cost += T.mean(T.sum((norms[1:] - norms[:-1])**2, axis=0) / (seq_len - 1))

    norm_cost.name = 'norm_cost'
    #cost_valid = cost_train
    cost_train += norm_cost_coeff * norm_cost
    cost_train = cost_train.copy('cost_train') #should this be cost_train.outputs[0]? no.

    cg_train = ComputationGraph([cost_train])


    ###########################################
    #
    # WEIGHT NOISE
    #
    ###########################################

    if weight_noise > 0:
        weights = VariableFilter(roles=[WEIGHT])(cg_train.variables)
        cg_train = apply_noise(cg_train, weights, weight_noise)
        cost_train = cg_train.outputs[0].copy(name='cost_train')

    model = Model(cost_train)

    learning_rate = float(learning_rate)
    clipping = StepClipping(threshold=np.cast[floatX](clipping))
    if algorithm == 'adam':
        adam = Adam(learning_rate=learning_rate)
        learning_rate = adam.learning_rate
        step_rule = CompositeRule([adam, clipping])
    elif algorithm == 'rms_prop':
        rms_prop = RMSProp(learning_rate=learning_rate)
        learning_rate = rms_prop.learning_rate
        step_rule = CompositeRule([clipping, rms_prop])
    elif algorithm == 'momentum':
        sgd_momentum = Momentum(learning_rate=learning_rate, momentum=momentum)
        learning_rate = sgd_momentum.learning_rate
        step_rule = CompositeRule([clipping, sgd_momentum])
    elif algorithm == 'sgd':
        sgd = Scale(learning_rate=learning_rate)
        learning_rate = sgd.learning_rate
        step_rule = CompositeRule([clipping, sgd])
    else:
        raise NotImplementedError
    algorithm = GradientDescent(step_rule=step_rule,
                                cost=cost_train,
                                parameters=cg_train.parameters)
                                # theano_func_kwargs={"mode": theano.compile.MonitorMode(post_func=detect_nan)})

    algorithm.add_updates(init_updates)

    def cond_number(x):
        _, _, sing_vals = T.nlinalg.svd(x, True, True)
        sing_mags = abs(sing_vals)
        return T.max(sing_mags) / T.min(sing_mags)
    def rms(x):
        return (x*x).mean().sqrt()

    whysplode_cond = []
    whysplode_rms = []
    for i, p in enumerate(init_updates):
        v = p.get_value()
        if p.get_value().shape == 2:
            whysplode_cond.append(cond_number(p).copy('ini%d:%s_cond(%s)'%(i, p.name, "x".join(map(str, p.get_value().shape)))))
        whysplode_rms.append(rms(p).copy('ini%d:%s_rms(%s)'%(i, p.name, "x".join(map(str, p.get_value().shape)))))
    for i, p in enumerate(cg_train.parameters):
        v = p.get_value()
        if p.get_value().shape == 2:
            whysplode_cond.append(cond_number(p).copy('ini%d:%s_cond(%s)'%(i, p.name, "x".join(map(str, p.get_value().shape)))))
        whysplode_rms.append(rms(p).copy('ini%d:%s_rms(%s)'%(i, p.name, "x".join(map(str, p.get_value().shape)))))

    observed_vars = [cost_train, cost, bpc, perp, learning_rate,
                     aggregation.mean(algorithm.total_gradient_norm).copy("gradient_norm_mean")] # + whysplode_rms

    parameters = model.get_parameter_dict()
    for name, param in parameters.iteritems():
        observed_vars.append(param.norm(2).copy(name=name + "_norm"))
        observed_vars.append(
            algorithm.gradients[param].norm(2).copy(name=name + "_grad_norm"))
    
    train_monitor = TrainingDataMonitoring(
        variables=observed_vars,
        prefix="train", after_epoch=True
    )

    dev_inits = [p.clone() for p in init_updates]
    cg_dev = ComputationGraph([cost, bpc, perp] + init_updates.values()).replace(zip(init_updates.keys(), dev_inits))
    dev_cost, dev_bpc, dev_perp = cg_dev.outputs[:3]
    dev_init_updates = OrderedDict(zip(dev_inits, cg_dev.outputs[3:]))

    dev_monitor = DataStreamMonitoring(
        variables=[dev_cost, dev_bpc, dev_perp],
        data_stream=valid_stream, prefix="dev",
        updates=dev_init_updates
    )

    # noone does this
    if 'load_path' in kwargs:
        with open(kwargs['load_path']) as f:
            loaded = np.load(f)
            model = Model(cost_train)
            params_dicts = model.get_parameter_dict()
            params_names = params_dicts.keys()
            for param_name in params_names:
                param = params_dicts[param_name]
                # '/f_6_.W' --> 'f_6_.W'
                slash_index = param_name.find('/')
                param_name = param_name[slash_index + 1:]
                if param.get_value().shape == loaded[param_name].shape:
                    print 'Found: ' + param_name
                    param.set_value(loaded[param_name])
                else:
                    print 'Not found: ' + param_name
    
    extensions = []
    extensions.extend([FinishAfter(after_n_epochs=epochs),
                       train_monitor, dev_monitor])
    if test_cost:
        test_inits = [p.clone() for p in init_updates]
        cg_test = ComputationGraph([cost, bpc, perp] + init_updates.values()).replace(zip(init_updates.keys(), test_inits))
        test_cost, test_bpc, test_perp = cg_test.outputs[:3]
        test_init_updates = OrderedDict(zip(test_inits, cg_test.outputs[3:]))

        test_monitor = DataStreamMonitoring(
            variables=[test_cost, test_bpc, test_perp],
            data_stream=test_stream,
            prefix="test",
            updates=test_init_updates
        )
        extensions.extend([test_monitor])

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    log_path = os.path.join(experiment_path, 'log.txt')
    fh = logging.FileHandler(filename=log_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    extensions.append(SaveParams('dev_cost', model, experiment_path,
                                 every_n_epochs=1))
    extensions.append(SaveLog(every_n_epochs=1))
    extensions.append(ProgressBar())
    extensions.append(Printing())

    class RollsExtension(TrainingExtension):
        """ rolls the cell and state activations between epochs so that first batch gets correct initial activations """
        def __init__(self, shvars):
            self.shvars = shvars
        def before_epoch(self):
            for v in self.shvars:
                v.set_value(np.roll(v.get_value(), 1, 0))
    extensions.append(RollsExtension(init_updates.keys() + dev_init_updates.keys() + (test_init_updates.keys() if test_cost else [])))

    class LearningRateSchedule(TrainingExtension):
        """ Lets you set a number to divide learning rate by each epoch + when to start doing that """
        def __init__(self):
            self.epoch_number = 0
        def after_epoch(self):
            self.epoch_number += 1
            if self.epoch_number > decrease_lr_after_epoch:
                learning_rate.set_value(learning_rate.get_value()/lr_decay)
    if bool(lr_decay) != bool(decrease_lr_after_epoch):
        raise ValueError('Need to define both lr_decay and decrease_lr_after_epoch')
    if lr_decay and decrease_lr_after_epoch:
        extensions.append(LearningRateSchedule())


    main_loop = MainLoop(model=model, data_stream=train_stream,
                         algorithm=algorithm, extensions=extensions)
    t1 = time.time()
    print "Building time: %f" % (t1 - t0)

    main_loop.run()
    print "Execution time: %f" % (time.time() - t1)


if __name__ == '__main__':
    args = parse_args()
    argsdict = args.__dict__
    argsdict['code_file'] = sys.argv[0]
    argsdict['last_git_commit'] = commands.getoutput('git log --format="%H" -n 1')
    flags = [flag.lstrip('--') for flag in sys.argv[1:]]
    experiment_path = '_'.join(["results"] + flags)
    i = 0
    while os.path.exists(experiment_path + "." + str(i)):
        i += 1
    experiment_path = experiment_path + "." + str(i)
    os.mkdir(experiment_path)
    print "putting log in %s"%experiment_path
    argsdict['experiment_path'] = experiment_path
    with open (os.path.join(experiment_path,'exp_params.txt'), 'w') as f:
        for key in sorted(argsdict):
            f.write(key+'    '+str(argsdict[key])+'\n')
    train(**args.__dict__)
