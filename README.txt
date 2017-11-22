Implementation of a basic Artificial Neural Network of adjustable size, and an
interface for testing the ANN on MNIST data using different parameters.

Created by Ethan Hoeppner

The program is run through the Main function in Program.cs. This implementation
can work for any number of layers of any size. The numberand size of layers can
be modified by changing the integer array variable LAYER_SIZES at the top of
Program.cs. For instance, if it is set to {728, 90, 10}, the NN will have 3 
layers, the first with size 728, the second with 90, and the third with size
10.

To work with the MNIST data, the first layer must be of size 728, because the
MNIST data exists in the form of 28x28 images. The last layer must be size 10,
because the MNIST data is to be classified into 10 digits.

The NeuralNetwork.cs is a general purpose class describing the function of the
Neural Network model. The Program.cs file, however, contains code specific to
the implementation of the NN with MNIST. The NeuralNetwork.cs file could be
used in any situation where a Neural Network would be an appropriate model.

The Neural Network implemented here uses a logistic activation function on
every layer. The logistic function is quite expensive, so instead of actually
computing the exact value during prediction, there is a function for
approximating the value. This approximation function relies on a list of exact
sigmoid values computed before the approximation is to be used. The list of
values is created by running the static GenerateSigmoidApproximations function
in NeuralNetwork. This must be done before any NeuralNetwork object is used to
make a prediction. The GenerateSigmoidApproximations function takes one int
argument: divisions. This is the number of exact sigmoid values that will be
calculated. The higher the number of divisions, the more accurate the
approximation.

This implementation uses Stochastic Gradient Descent for training. Whenever
a prediction is run on the NN, it stores the activations for each of it's
neurons. When the train function is called on the NN, it uses the most recent
set of activations to modify the weights and biases.