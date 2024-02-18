using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;

namespace UpdatedProject
{
    internal class PredictInput
    {
        MLP forwardPass = new MLP();
        ManageData manageData = new ManageData();

        public void FindNumInPicture(int LayerCount)
        {
            Vector<double> LayerVector = manageData.GetImage();

            forwardPass.Forwards(LayerVector, 0, LayerCount);

            int PredictedNum = forwardPass.Cache[forwardPass.Cache.Count() - 1].MaximumIndex();

            Console.WriteLine($"The network predicts this num to be: {PredictedNum}");

        }


    }
}
