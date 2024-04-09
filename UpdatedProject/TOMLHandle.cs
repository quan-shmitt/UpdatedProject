using Nett;
using System.Linq;

namespace UpdatedProject
{
    internal static class TOMLHandle
    {
        static TomlTable TOMLFILE;

        public static int LayerCount;

        public static void GetToml(string filename)
        {
            TOMLFILE = Toml.ReadFile(filename);
            LayerCount = GetHiddenLayerCount().Length;
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

        public static int GetCNNLayerCount()
        {
            var CNNLayerCount = TOMLFILE.Get<TomlTable>("CNNStruct").Get<int>("CNNRepeats");

            return CNNLayerCount;
        }


        public static int GetLearningRate()
        {
            var LearningRate = TOMLFILE.Get<TomlTable>("MLPStruct").Get<int>("LearningRate");

            return LearningRate;
        }

        public static int[] GetHiddenLayerCount()
        {
            var HiddenLayerCount = TOMLFILE.Get<TomlTable>("MLPStruct").Get<TomlArray>("HiddenLayerCount");

            int[] HiddenCount = HiddenLayerCount.Items.Select(item => item.Get<int>()).ToArray();

            return HiddenCount;
        }

        public static string[] GetOutputClasses()
        {
            var OutputClasses = TOMLFILE.Get<TomlTable>("MLPStruct").Get<TomlArray>("OutputCLasses");

            string[] StringArray = OutputClasses.Items.Select(item => item.Get<string>()).ToArray();

            return StringArray;
        }

        public static int GetOutputLayerCount()
        {
            var OutputClasses = TOMLFILE.Get<TomlTable>("MLPStruct").Get<TomlArray>("OutputClasses");

            return OutputClasses.Length;
        }

        public static int[] GetTargetResolution()
        {
            var TargetRes = TOMLFILE.Get<TomlTable>("CNNStruct").Get<TomlArray>("TargetResolution");

            int[] Scale = TargetRes.Items.Select(item => item.Get<int>()).ToArray();

            return Scale;
        }

        public static int GetPoolSize()
        {
            var PoolSize = TOMLFILE.Get<TomlTable>("CNNStruct").Get<int>("PoolSize");

            return PoolSize;
        }


    }
}
