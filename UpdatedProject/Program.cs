using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace UpdatedProject
{
    internal class Program
    {
        const int Xweight = 784;
        const int Yweight = 16;


        static void Main(string[] args)
        {
            Datainit();

            Console.WriteLine("enter max image count");
            //int imageMaxIndex = Convert.ToInt32(Console.ReadLine());
            int layer = 2;

            pass(layer);
            
        }



        static void pass(int layer)
        {
            int imageMaxIndex = 2;

            for (int i = 0; i <= imageMaxIndex; i++)
            {
                NetInIt networkGen = new NetInIt(i, layer);


                ForwardPass forwardPass = new ForwardPass();
                forwardPass.Forwards(networkGen.weights,networkGen.LayerVector, networkGen.BiasVector,layer, imageMaxIndex, i);
            }
        }



        static void Datainit()
        {
            Console.WriteLine("started");
            string path = "MNIST\\MNISTDataSet.exe";

            if (!File.Exists("MNIST\\images"))
            {
                Console.WriteLine("started processing");
                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = path,
                    UseShellExecute = true
                };

                try
                {
                    Process process = Process.Start(startInfo);
                    process.WaitForExit();
                    Console.WriteLine("finished");
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                }
            }
        }

    }
}
