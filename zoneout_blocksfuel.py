# This is a minimal gist of the things you'd have to
# add to blocks/fuel (Theano) code to implement zoneout.

# To see this in action, see zoneout_char_ptb.py 
# or zoneout_word_ptb.py or zoneout_text8.py


# Set zoneout probabilites for states and cells
z_prob_states = 0.05
z_prob_cells = 0.5


# Make theano variables for the zoneout masks
zoneouts_states = T.tensor3('zoneout_states')
zoneouts_cells = T.tensor3('zoneout_cells')

# A ZoneoutLSTM which extends blocks' recurrent class
class ZoneoutLSTM(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None, **kwargs):
        self.dim = dim
        self.model_type = model_type

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        children = [self.activation, self.gate_activation]
        kwargs.setdefault('children', []).extend(children)
        super(ZoneoutLSTM, self).__init__(**kwargs)

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells', 'zoneout_states', 'zoneout_cells']:
            return self.dim
        if name == 'mask':
            return 0
        return super(ZoneoutLSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4 * self.dim), name='W_state')
        add_role(self.W_state, WEIGHT)
        self.parameters = [self.W_state]

    def _initialize(self):
        for weights in self.parameters[:1]:
            self.weights_init.initialize(weights, self.rng)

    @recurrent(sequences=['inputs', 'zoneout_states', 'zoneout_cells', 'mask'],
               states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, zoneout_states, zoneout_cells, states, cells, mask=None):
        def slice_last(x, no):
            return x[:, no * self.dim: (no + 1) * self.dim]

        activation = tensor.dot(states, self.W_state) + inputs
        in_gate = self.gate_activation.apply(slice_last(activation, 0))
        forget_gate_input = slice_last(activation, 1)
        forget_gate = self.gate_activation.apply(
            forget_gate_input + tensor.ones_like(forget_gate_input))
        next_cells = (
            forget_gate * cells +
            in_gate * self.activation.apply(slice_last(activation, 2)))
        out_gate = self.gate_activation.apply(slice_last(activation, 3))
        next_states = out_gate * self.activation.apply(next_cells)

        # we always "do zoneout"; can pass masks which effectively turn off zoneout
        # by setting zoneout to 1 (i.e. pass all ones, or pass actual probabilites)
        next_states = next_states * zoneout_states + (1 - zoneout_states) * states
        next_cells = next_cells * zoneout_cells + (1 - zoneout_cells) * cells
      
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells



# A transformer that samples zoneout masks for a given datastream
class SampleZoneouts(Transformer):
    def __init__(self, data_stream, z_prob_states, z_prob_cells,
    	         layer_size, is_for_test, **kwargs):
        super(SampleZoneouts, self).__init__(data_stream, **kwargs)
        self.z_prob_states = 1-z_prob_states
        self.z_prob_cells = 1-z_prob_cells
        self.layer_size = layer_size
        self.is_for_test = is_for_test
        self.produces_examples = False

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        transformed_data = [] # Currently T x B x F
        transformed_data.append(np.swapaxes(data[0], 0, 1))
        T, B, _ = transformed_data[0].shape
        if self.is_for_test:
            zoneouts_states = np.ones((T, B, self.layer_size)) * self.z_prob_states
            zoneouts_cells = np.ones((T, B, self.layer_size)) * self.z_prob_cells
        else:
            zoneouts_states = np.random.binomial(n=1, p=self.z_prob_states,
                                       size=(T, B, self.layer_size))
            zoneouts_cells = np.random.binomial(n=1, p=self.z_prob_cells,
                                       size=(T, B, self.layer_size))

        transformed_data.append(zoneouts_states.astype(floatX))
        transformed_data.append(zoneouts_cells.astype(floatX))
        return transformed_data

# Use the SampleZoneouts transformer to sample masks that are the 
# appropriate size for the data stream, return the stream
def get_stream(which_set, batch_size, length, z_prob_states, z_prob_cells,
               hidden_dim, for_evaluation, num_examples):
    iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size)
    stream = fuel.streams.DataStream.default_stream(which_set, iteration_scheme)
    transformed_stream = SampleZoneouts(stream, z_prob_states, z_prob_cells, 
    	                                layer_size, is_test_time)
    transformed_stream.sources = ('features',  'zoneouts_states', 'zoneouts_cells')
    return transformed_stream


# Construct streams, make ghostbusters joke
# This assumes you have a fuel dataset with sets like 'train', 'val', 'test'
train_stream = get_stream(
        'train', batch_size, seq_len, zoneout_states, zoneout_cells, state_dim, False)
val_stream = get_stream(
        'val', batch_size, seq_len, zoneout_states, zoneout_cells, state_dim, False)
test_stream = get_stream(
        'test', batch_size, seq_len, zoneout_states, zoneout_cells, state_dim, True)
