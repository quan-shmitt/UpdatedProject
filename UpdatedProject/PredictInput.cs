using MathNet.Numerics.LinearAlgebra;
using System;
using System.Linq;

namespace UpdatedProject
{
    internal class PredictInput
    {
        readonly ManageData manageData = new ManageData();

        public void FindNumInPicture(int LayerCount, int threashold)
        {
            Matrix<double> LayerMatrix = manageData.GetImage();


            MLP forwardPass = new MLP();

            CNNLayers cnn = new CNNLayers(LayerMatrix);

            cnn.Forwards(0, LayerCount, threashold);

            forwardPass.Forwards(cnn.MatrixToVector(LayerMatrix), 0, LayerCount);


            int PredictedNum = forwardPass.Cache[forwardPass.Cache.Count() - 1].MaximumIndex();
            Console.WriteLine(forwardPass.Cache[forwardPass.Cache.Count() - 1]);

            Console.WriteLine($"The network predicts this num to be: {PredictedNum}");

        }


    }
}
