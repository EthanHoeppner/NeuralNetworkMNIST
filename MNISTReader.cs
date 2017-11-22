using System.IO;

namespace NeuralNetworkMNIST
{
    class MNISTReader
    {
        private BinaryReader TrainingImages;
        private BinaryReader TrainingLabels;
        private BinaryReader TestImages;
        private BinaryReader TestLabels;

        private int currentTrainingLabel;
        private int currentTestLabel;

        public MNISTReader()
        {
            TrainingImages = new BinaryReader(File.Open("mnistData/train-images.idx3-ubyte", FileMode.Open, FileAccess.Read, FileShare.Read));
            TrainingLabels = new BinaryReader(File.Open("mnistData/train-labels.idx1-ubyte", FileMode.Open, FileAccess.Read, FileShare.Read));
            TestImages = new BinaryReader(File.Open("mnistData/t10k-images.idx3-ubyte", FileMode.Open, FileAccess.Read, FileShare.Read));
            TestLabels = new BinaryReader(File.Open("mnistData/t10k-labels.idx1-ubyte", FileMode.Open, FileAccess.Read, FileShare.Read));

            TrainingImages.ReadInt32();
            TrainingImages.ReadInt32();
            TrainingImages.ReadInt32();
            TrainingImages.ReadInt32();

            TrainingLabels.ReadInt32();
            TrainingLabels.ReadInt32();

            TestImages.ReadInt32();
            TestImages.ReadInt32();
            TestImages.ReadInt32();
            TestImages.ReadInt32();

            TestLabels.ReadInt32();
            TestLabels.ReadInt32();
        }

        public float[] getTrainingImage()
        {
            float[] image = new float[28 * 28];
            for (int i = 0; i < 28 * 28; i++)
            {
                image[i] = ((float)TrainingImages.ReadByte()) / 255;
            }
            currentTrainingLabel = (int)TrainingLabels.ReadByte();
            return image;
        }

        public int getTrainingLabel()
        {
            return currentTrainingLabel;
        }

        public float[] getTestImage()
        {
            float[] image = new float[28 * 28];
            for (int i = 0; i < 28 * 28; i++)
            {
                image[i] = ((float)TestImages.ReadByte()) / 255;
            }
            currentTestLabel = (int)TestLabels.ReadByte();
            return image;
        }

        public int getTestLabel()
        {
            return currentTestLabel;
        }
    }
}
