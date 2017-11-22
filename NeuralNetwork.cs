using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkMNIST
{
    class NeuralNetwork
    {
        private const float SIGMOID_APPROXIMATION_DISTANCE = 6;
        private static float[] sigmoidValues;
        private static float sigmoidApproximationDivisions;
        private static float sigmoidDivisionSize;

        private int layerCount;

        private int[] layerSizes;

        private float[][,] weights;
        private float[][] biases;

        private float[][] activations;

        public NeuralNetwork(int[] layerSizes, int initializationSeed)
        {
            layerCount = layerSizes.Length;

            this.layerSizes = new int[layerCount];
            weights = new float[layerCount-1][,];
            biases = new float[layerCount - 1][];

            activations = new float[layerCount][];
            
            for(int i=0;i<layerCount;i++)
            {
                //Get and store the size of the current layer
                int layerSize = layerSizes[i];
                this.layerSizes[i] = layerSize;

                //Initialize activation array
                activations[i] = new float[layerSize];
            }

            Random rng = new Random(initializationSeed);
            //Initialize the weights and biases for each layer
            for(int i=0;i<layerCount-1;i++)
            {
                int layerSize = layerSizes[i];
                int nextLayerSize = layerSizes[i + 1];
                //Initialize weights for this layer with random values between
                //-1 and 1
                weights[i] = new float[layerSize, nextLayerSize];
                for(int i2=0;i2<layerSize;i2++)
                {
                    for(int i3=0;i3<nextLayerSize;i3++)
                    {
                        weights[i][i2,i3] = (float)(rng.NextDouble() * 2 - 1);
                    }
                }

                //Initialize biases for this layer with 0s
                biases[i] = new float[nextLayerSize];
            }
        }

        public float[] Predict(float[] input)
        {
            for(int i=0;i<layerSizes[0];i++)
            {
                activations[0][i] = input[i];
            }

            //Itterate through each layer, calculating the new activation
            for(int layer=0;layer<layerCount-1;layer++)
            {
                float[] currentActivation = activations[layer];
                float[] nextActivation = activations[layer + 1];
                int currentLayerSize = layerSizes[layer];
                int nextLayerSize = layerSizes[layer + 1];

                //Initialize the vector to hold the activation of the new layer

                //Multiply the current activation vector by the weight matrix
                //of the current layer to get the actiation of the next layer
                float[,] currentLayerWeights = weights[layer];
                for (int i=0;i<currentLayerSize;i++)
                {
                    for(int i2=0;i2<nextLayerSize;i2++)
                    {
                        nextActivation[i2] += currentLayerWeights[i,i2] * currentActivation[i];
                    }
                }

                //Add the bias of the layer to the newly calculated activation
                float[] currentBias = biases[layer];
                for(int i=0;i<nextLayerSize;i++)
                {
                    nextActivation[i] += currentBias[i];
                }

                //Apply the sigmoid function to each element to finish
                //activation calculation
                for(int i=0;i<nextLayerSize;i++)
                {
                    nextActivation[i] = ApproximateSigmoid(nextActivation[i]);
                }

                //Replace old activation vector with new activation vector
                currentActivation = nextActivation;
            }

            return activations[layerCount - 1];
        }

        public void train(float[] properOutput, float trainingFactor)
        {
            //Calculate the error of the ouput layer
            float[] error = new float[layerSizes[layerCount - 1]];
            for(int i=0;i<layerSizes[layerCount-1];i++)
            {
                float activation = activations[layerCount - 1][i];
                error[i] = activation * (1 - activation) * (properOutput[i] - activation);
            }

            //Itterate through all layers backwards, updating the weights and
            //biases based on the error of the next layer, and then calculating
            //the error for that layer to be used on the previous layer
            for(int layer=layerCount-1;layer>0;layer--)
            {
                //Retrieve data on the size, weights, biases, and activations
                //of the relevant layer
                int layerSize = layerSizes[layer];
                int previousLayerSize = layerSizes[layer - 1];
                float[,] weightMatrix = weights[layer-1];
                float[] biasVector = biases[layer-1];
                float[] previousActivations = activations[layer - 1];

                //Update the weight matrix based on the previous layer's error
                for(int i=0;i<layerSize;i++)
                {
                    for(int i2=0;i2<previousLayerSize;i2++)
                    {
                        weightMatrix[i2,i] += trainingFactor * error[i] * previousActivations[i2];
                    }
                    biasVector[i] += trainingFactor * error[i];
                }

                //Calculate the error for this layer
                float[] newError = new float[previousLayerSize];
                for(int i=0;i<previousLayerSize;i++)
                {
                    for(int i2=0;i2<layerSize;i2++)
                    {
                        float activation = previousActivations[i2];
                        newError[i] += error[i2] * weightMatrix[i,i2] * activation * (1 - activation);
                    }
                }
                error = newError;
            }
        }

        public static float Sigmoid(float x)
        {
            return 1 / (1 + (float)Math.Exp(-x));
        }

        public static float ApproximateSigmoid(float x)
        {
            //Find how far between the lowest and highest approximated
            //distances this value lies
            float proportion = (x + SIGMOID_APPROXIMATION_DISTANCE) / (2 * SIGMOID_APPROXIMATION_DISTANCE);

            //Find the index of the approximate value below
            int lowerIndex = (int)(proportion / sigmoidDivisionSize);

            //If the value is outside the range of the approximation, just
            //return 0 or 1
            if(lowerIndex<0)
            {
                return 0;
            }
            if(lowerIndex >= sigmoidApproximationDivisions -1)
            {
                return 1;
            }

            //Otherwise, return a linear interpolation between approximations
            //above and below the value
            float overhang = proportion % sigmoidDivisionSize;
            float between = overhang / sigmoidDivisionSize;
            return sigmoidValues[lowerIndex] * (1 - between) + sigmoidValues[lowerIndex + 1] * between;
        }
        
        public static void GenerateSigmoidApproximations(int divisions)
        {
            sigmoidApproximationDivisions = divisions;
            sigmoidValues = new float[divisions];
            sigmoidDivisionSize = 1.0f / (divisions - 1);
            for(int i=0;i<divisions;i++)
            {
                float position = ((float)i) / (divisions - 1) * SIGMOID_APPROXIMATION_DISTANCE * 2 - SIGMOID_APPROXIMATION_DISTANCE;
                sigmoidValues[i] = Sigmoid(position);
            }
        }
    }
}
