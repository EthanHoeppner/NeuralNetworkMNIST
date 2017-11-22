using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkMNIST
{
    class Program
    {
        public const int TRAINING_COUNT = 60000;
        public const int TESTING_COUNT = 10000;

        public const float TRAINING_SPEED = 0.1f;
        public static readonly int[] LAYER_SIZES = { 784, 10 };

        static void Main(string[] args)
        {
            //Generate the values to be used in the sigmoid approximations
            NeuralNetwork.GenerateSigmoidApproximations(1000);

            //Initialize the MNIST data
            MNISTReader reader = new MNISTReader();

            //Initialize the neural network
            NeuralNetwork network = new NeuralNetwork(LAYER_SIZES, 0);

            //Traing the neural network on the MNIST training data
            Console.Out.WriteLine("Training...");
            for(int i=0;i<TRAINING_COUNT;i++)
            {
                float[] image = reader.getTrainingImage();
                int label = reader.getTrainingLabel();
                float[] output = network.Predict(image);
                int guessedLabel = 0;
                float maxOutput = output[0];
                for(int i2=1;i2<10;i2++)
                {
                    if (output[i2] > maxOutput)
                    {
                        maxOutput = output[i2];
                        guessedLabel = i2;
                    }
                }
                if(guessedLabel!=label)
                {
                    float[] properOutput = new float[10];
                    properOutput[label] = 1;
                    network.train(properOutput, TRAINING_SPEED);
                }
                if((int)((10*(float)i)/TRAINING_COUNT)>(int)((10*((float)i-1)) / TRAINING_COUNT))
                {
                    Console.Out.WriteLine((int)((100 * (float)i) / TRAINING_COUNT) + "%");
                }
            }

            //Test the neural network on the MNIST test data to determine the
            //degree to which the training was successful
            Console.Out.WriteLine("Testing...");
            int successes = 0;
            for (int i = 0; i < TESTING_COUNT; i++)
            {
                float[] image = reader.getTestImage();
                int label = reader.getTestLabel();
                float[] output = network.Predict(image);
                int guessedLabel = 0;
                float maxOutput = output[0];
                for (int i2 = 1; i2 < 10; i2++)
                {
                    if (output[i2] > maxOutput)
                    {
                        maxOutput = output[i2];
                        guessedLabel = i2;
                    }
                }
                if (guessedLabel != label)
                {
                    successes++;
                }
                if ((int)((10 * (float)i) / TESTING_COUNT) > (int)((10 * ((float)i - 1)) / TESTING_COUNT))
                {
                    Console.Out.WriteLine((int)((100 * (float)i) / TESTING_COUNT) + "%");
                }
            }
            Console.Out.WriteLine("Success: " + ((float)successes) / TESTING_COUNT);

            Console.Out.WriteLine("Done. Press Enter to exit.");
            Console.In.ReadLine();
        }
    }
}
