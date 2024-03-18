﻿using Nett;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace UpdatedProject
{
    internal static class TOMLHandle
    {
        static TomlTable TOMLFILE;

        public static void GetToml(string filename)
        {
            TOMLFILE = Toml.ReadFile(filename);
        }

        public static string[] GetCNNStruct()
        {
            var algorithmsArray = TOMLFILE.Get<TomlTable>("CNNStruct").Get<TomlArray>("algorithms");

            string[] stringArray = algorithmsArray.Items.Select(item => item.Get<string>()).ToArray();

            return stringArray;
        }
 
        public static int GetKernelSize()
        {
            var KernelSize = TOMLFILE.Get<TomlTable>("CNNStruct").Get<int>("kernelSize");

            return KernelSize;
        }

        public static int GetKernelStep()
        {
            var KernelStep = TOMLFILE.Get<TomlTable>("CNNStruct").Get<int>("kernelStep");

            return KernelStep;
        }


        public static int GetLearningRate()
        {
            var LearningRate = TOMLFILE.Get<TomlTable>("MLPStruct").Get<int>("LearningRate");

            return LearningRate;
        }

        public static int GetHiddenLayerCount()
        {
            var HiddenLayerCount = TOMLFILE.Get<TomlTable>("MLPStruct").Get<int>("HiddenLayerCount");

            return HiddenLayerCount;
        }

        public static string[] GetOutputClasses()
        {
            var OutputClasses = TOMLFILE.Get<TomlTable>("MLPStruct").Get<TomlArray>("OutputCLasses");

            string[] StringArray = OutputClasses.Items.Select(item =>item.Get<string>()).ToArray();

            return StringArray;
        }

        public static int GetScaleFactor()
        {
            var ScaleFactor = TOMLFILE.Get<TomlTable>("CNNStruct").Get<int>("ScaleFactor");

            return ScaleFactor;
        }

        public static int GetPoolSize()
        {
            var PoolSize = TOMLFILE.Get<TomlTable>("CNNStruct").Get<int>("PoolSize");

            return PoolSize;
        }


    }
}