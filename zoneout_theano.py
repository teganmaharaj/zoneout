# This is a very minimal gist of the things you'd have to
# add to raw Theano code to implement zoneout.

# To see this in action, see zoneout_pmnist.py


# Set zoneout probabilites for states and cells
z_prob_states = 0.05
z_prob_cells = 0.5


# Make theano variables for the zoneout masks
zoneouts_states = T.tensor3('drops_states')
zoneouts_cells = T.tensor3('drops_cells')


# Sample masks (probably when you get a batch from input data)
if is_test_time:
    zoneouts_states = np.ones((T, B, layer_size)) * (1-z_prob_states)
    zoneouts_cells = np.ones((T, B, layer_size)) * (1-z_prob_cells)
else:
    zoneouts_states = np.random.binomial(n=1, p=(1-z_prob_states), size=(T, B, layer_size))
    zoneouts_cells = np.random.binomial(n=1, p=(1-z_prob_cells), size=(T, B, layer_size))


# Pass zoneout masks to the LSTM's step function
# and multiply by them to apply zoneout
def step_fn(x, h_prev, c_prev, zoneouts_states, zoneouts_cells, h, c)

	# Compute everything else in the LSTM

	# If zoneouts=0, h=h (no zoneout); 
	# If zoneouts=1, h=h_prev (total zoneout, not recommended.)
	if zoneout:
		h = h_prev * zoneouts_states + (1 - zoneouts_states) * h
		c = c_prev * zoneouts_cells + (1 - zoneouts_cells) * c

	return h,c
