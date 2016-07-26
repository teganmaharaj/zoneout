# Zoneout

This repo contains the code for replicating the results in [Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations](http://arxiv.org/abs/1606.01305), as well as gists to help implement zoneout in your code (in Theano and Tensorflow).

Zoneout is a regularizer for RNNs. At each timestep, units have a random probability of maintaining their previous value. This can be seen as [dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) using an identity mask instead of zero mask, or like a per-unit version of [stochastic depth](https://arxiv.org/pdf/1603.09382.pdf). 

We set state of the art on character-level Penn Treebank with 1.27 BPC, match state of the art 1.36 BPC on text8, and combine with [recurrent batch normalization](https://arxiv.org/abs/1603.09025) to set state of the art 95.9% accuracy on permuted sequential MNIST. We performed no hyperparameter search to get these results; just used settings/architectures from the previous state-of the art and in some cases searched over zoneout probabilites.

## Replicating experiments from the paper

For details about each dataset, and why we chose them/what they demonstrate about zoneout, and for the exact hyperparameter settings used in each experiment, please see the 'Experiments' section of the paper (or look at the default arguments in each script).

### Permuted sequential MNIST

Baseline (unregularized LSTM)
```
zoneout_pmnist.py
```
Zoneout (zoneout probability 0.15 on both cells and states of an LSTM)
```
zoneout_pmnist.py --z_prob_states=0.15 --z_prob_cells=0.15
```
Zoneout + recurrent batch normalization
```
zoneout_pmnist.py --z_prob_states=0.15 --z_prob_cells=0.15 --batch_normalization
```

### Char-level Penn Treebank

NOTE: Currently zoneout probabilities are (1-zoneout_probability) TO BE FIXED.

Baseline (unregularized LSTM)
```
zoneout_char_ptb.py
```
Weight noise baseline 
```
zoneout_char_ptb.py --weight_noise=0.075
```
Norm stabilizer baseline
```
zoneout_char_ptb.py --norm_cost_coeff=50
```
Input-gate dropout baseline (dropout probability 0.7)
```
zoneout_char_ptb.py --drop_prob_igates=0.7
```
Zoneout (with probability 0.5 of zoning out on cells and 0.05 on hidden states in LSTM)
```
zoneout_char_ptb.py --z_prob_cells=0.5 --z_prob_states=0.05
```

### Word-level Penn Treebank

NOTE: Currently zoneout probabilities are (1-zoneout_probability) TO BE FIXED.

Baseline (unregularized LSTM)
```
zoneout_word_ptb.py
```
Zoneout (with probability 0.5 of zoning out on cells and 0.05 on hidden states in LSTM)
```
zoneout_word_ptb.py --z_prob_cells=0.5 --z_prob_states=0.05
```

### Text8

NOTE: Currently zoneout probabilities are (1-zoneout_probability) TO BE FIXED.

Baseline (unregularized LSTM)
```
zoneout_text8.py
```
Weight noise baseline 
```
zoneout_text8.py --weight_noise=0.075
```
Norm stabilizer baseline
```
zoneout_text8.py --norm_cost_coeff=50
```
Input-gate dropout baseline (dropout probability 0.5)
```
zoneout_text8.py --drop_prob_igates=0.5
```
Zoneout (with probability 0.2 of zoning out on cells and 0.2 on hidden states in LSTM)
```
zoneout_text8.py --z_prob_cells=0.2 --z_prob_states=0.2
```

### Toy sequence to sequence task

This is a simplified version of learning to execute, based on Erik Rehn's tensorflow code. These experiments are not yet in the paper.

Baseline (unregularized LSTM)
```
zoneout_seq2seq.py
```
Zoneout 
```
zoneout_seq2seq.py --z_prob_cells=0.5 --z_prob_states=0.05
```

### Semantic consistency

A toy task demonstrating that networks with zoneout encourage units to retain semantic consistency over time.
```
zoneout_semantic_consistency.py
```

### Gradient propagation

Code mostly from recurrent batch normalization, to demonstrate that networks with zoneout propagate information across timesteps more effectively.
```
zoneout_gradient_propagation.py
```

## Implementing zoneout in your code

The repo contains implementations of zoneout in Theano (both pure Theano, and using the Blocks framework) and Tensorflow. You can adapt the scripts used to run the experiments, described above, or look at the following three gists:

```
zoneout_theano.py
zoneout_blocksfuel.py
zoneout_tensorflow.py
```
These repos may also be useful:

https://github.com/mohammadpz/LSTM_Dropout/

https://github.com/ballasn/recurrent-batch-normalization

## Credits

The permuted sequential MNIST code was mostly written by Mohammad Pezeshki and Nicolas Ballas, modified by Tegan Maharaj, and is based heavily on code written by Tim Cooijmans and Nicolas Ballas for recurrent batch normalization.

The char-PTB code was written mostly by Janos Kramar and Tegan Maharaj, based on code written by Mohammad Pezeshki, based on code from MILA's speech group.
The word-level PTB code was based on this code, extended by Janos Kramar, Tegan Maharaj, and David Krueger. The text8 code was also based on this code, extended by Tegan Maharaj. The sequence to sequence task is tensorflow code from Erik Rehn, with zoneout implemented by David Krueger, Janos Kramar, and Tegan Maharaj. The semantic consistency task and code were developed by David Krueger. The gradient propagation code is from recurrent batch normalization. 

The original idea for zoneout was proposed by Anirudh Goyal in discussion with David Krueger, and inspired by his earlier conversations with Yoshua Bengio. Initial experiements were run by David Krueger, who then involved Tegan Maharaj, Janos Kramar, and Christopher Pal. Mohammad Pezeshki was independently persuing a similar idea, with Nicolas Ballas, Hugo Larochelle, and Aaron Courville, so we combined forces and code. Experiments were run by Tegan Maharaj, Janos Kramar, Mohammad Pezeshki, Nicolas Ballas, David Krueger, and Rosemary Nan Ke. Theory was mostly develped and elaborated by David Krueger, in discussions with Janos Kramar, Tegan Maharaj, Nicolas Ballas, Mohammad Pezeshki, Chris Pal, Aaron Courville, Hugo Larochelle, and Anirudh Goyal. The paper was written and figures/tables produced by David Krueger, Tegan Maharaj, Janos Kramar, Nicolas Ballas,  Mohammad Pezeshki, Rosemary Nan Ke, and Chris Pal. 

We had important contributing discussions with Christopher Beckham, Chiheb Trabelsi, Marcin Moczulski, Caglar Gulcehre, and others at MILA (the Montreal Institute for Learning Algorithms). Blocks expertise by Mohammad Pezeshki and Dima Bahdanau, Theano wizardry by Nicolas Ballas and David Krueger, lots of other code, general linux fu, and dark stats knowledge by Janos Kramar, general knowing what's going on at any given time and background research mostly David Krueger, this repo put together by Tegan Maharaj.

Important contributing ideas to this project include [dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf), [stochastic depth](https://arxiv.org/pdf/1603.09382.pdf), [pseudoensembles](https://arxiv.org/abs/1412.4864), [resnets](https://arxiv.org/abs/1512.03385). Similar ideas were developed independently as [recurrent dropout without memory loss](http://arxiv.org/abs/1603.05118) (dropping the input gate), [swapout](https://arxiv.org/pdf/1605.06465.pdf) (idential to zoneout, but in feed-forward nets, where they call it skip-forward), and by [Pranav Shyam](https://github.com/pranv/lrh/blob/master/about.md). Dropout has been studied in recurrent nets by [Moon et al.](http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf) (same units dropped at every time-step), [Gal](http://arxiv.org/abs/1512.05287) (same input and output gates dropped at every timestep + embedding dropout), [Zaremba et al.](http://arxiv.org/abs/1512.05287) (up the stack, not on recurrent connections), and [Pham et al.](https://arxiv.org/pdf/1312.4569.pdf) (on feed-forward connections only). For full references and more explanation of zoneout, see the paper.

Please feel free to contact the authors with any questions, comments, or suggestions!
