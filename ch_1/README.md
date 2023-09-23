# Tasks and their descriptions


## Sigmoid neurons simulating perceptrons, part I

### Task

Suppose we take all the weights and biases in a network of perceptrons, and multiply them by a positive constant, $c>0$. 
Show that the behaviour of the network doesn't change.

### Python script `ex_0_sigmoid` 
In this script the network of perceptrons is made of 3 inputs, 2 perceptron-neurons on the level 0 and 1 perceptron-neuron on the second level
which makes the final output. All the weigths are generated randomly and are integers. Each perceptron-neuron gives one binary output (0 or 1).

#### Parameters
- `LEN_INPUTS` - number of inputs which are generated randomly 
- `NUM_L0_NEURONS` - number of neurons on the level 0 of perceptron-network 
- `C` - constant which is used to multiply the weights and the biases. Should be positive.

#### Results
After testing and extensive logging the script works correctly. Using the data of 20 runs it can be safely assumped that multipling the weights
by a positive constant `c` gives the same results as original weigths. As I understand it is the case because the output of each perceptron-neuron depends
on the sign (`+` or `-`) of operation and multipling the $w \cdot c$ and $b \cdot c$ won't change the sign of the the operation before step function. 

Visually:
- $f(w \cdot x + b)$
- $f((w \cdot c) \cdot x + (b \cdot c)) = f(c \cdot (w \cdot x + b))$
- $f(w \cdot x + b) = f(c \cdot (w \cdot x + b))$
