using CsvHelper;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace TP5
{
    class Program
    {
        private static string weightsFile = "weights.csv";
        private static Matrix<double>[] currentWeights;
        static List<Vector<double>> ParseCSV(string path, int recordLines)
        {
            List<Vector<double>> trainingInput = new List<Vector<double>>();
            using (var reader = new StreamReader(path))
            {
                while (!reader.EndOfStream)
                {
                    var line = "";
                    for (var i = 0; i < recordLines; i++)
                        line += "," + reader.ReadLine();
                    var values = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries)
                        .Select(val => Double.Parse(val));
                    trainingInput.Add(Vector<double>.Build.Dense(values.ToArray()));
                }
            }
            return trainingInput;
        }

        public static void PersistWeights(Matrix<double>[] weights, string file)
        {
            using var stream = new StreamWriter(file);
            foreach(var matrix in weights)
            {
                var line = string.Join(' ', matrix
                    .ToRowMajorArray()
                    .Select(d => d.ToString()));
                stream.WriteLine(line);
            }
        }

        public static Matrix<double>[] RestoreWeights(int[] layers, string file)
        {
            using var stream = new StreamReader(file);
            var matrices = new List<Matrix<double>>();
            for(int i = 0; i < layers.Length - 1; i++)
            {
                var strings = stream.ReadLine().Split(' ');
                var fields = strings.Select(s => double.Parse(s));
                matrices.Add(Matrix<double>.Build.DenseOfRowMajor(layers[i+1], layers[i] + 1, fields));
            }
            return matrices.ToArray();
        }

        /// <summary>
        /// Autoencoders
        /// </summary>
        /// <param name="config">Path to the configuration file</param>
        static void Main(string config)
        {
            Configuration configuration = Configuration.FromYamlFile(config);
            if(configuration.ExerciseTest == 1)
            {
                FirstExcercise(configuration);
            }
            else
            {
                Ej2(configuration.Epochs,configuration.LearningRate,configuration.Batch,configuration.MomentumAlpha);
            }
        }

        static void FirstExcercise(Configuration config)
        {

            Func<double, double>  activationFunction;
            Func<double, double> activationFunctionD;
            switch (config.Activation)
            {
                case "linear":
                    activationFunction = val => val;
                    activationFunctionD = val => 1;
                    break;
                case "nonlinear":
                    activationFunction = val => 1.0 / (1.0 + Math.Exp(-val));
                    activationFunctionD = val => (1.0 / (1.0 + Math.Exp(-val))) * (1 - (1.0 / (1.0 + Math.Exp(-val))));
                    break;
                default:
                    Console.WriteLine("Activation function must be linear or nonlinear");
                    return;
            }
            var actFunctions = new Func<double, double>[config.Layers.Length];
            var actDFunctions = new Func<double, double>[config.Layers.Length];
            Array.Fill(actFunctions, activationFunction);
            Array.Fill(actDFunctions, activationFunctionD);

            var perceptron = new MultiLayerPerceptron(config.Layers.ToArray(), config.LearningRate, actFunctions, actDFunctions, config.AdaptiveLearningRate, config.Momentum);
            if (config.Momentum)
                perceptron.Alpha = config.MomentumAlpha;
            if(config.TrainingInput == null || config.TrainingInput == "font_set")
            {
                var fonts = Configuration.FONT_SET.Select(row => Vector<double>.Build.DenseOfEnumerable(row.Select(n => n / 31.0)));
                perceptron.Learn(fonts.ToArray(), fonts.ToArray(), new Vector<double>[] { }, new Vector<double>[] { }, config.Batch, config.MinError, config.Epochs);
                // ----------------------------------------------- RESOLUCION 1.A.4 -------------
                // obtengo el espacio laente de a
                //var c = perceptron.MapGet(fonts.ToArray()[3]);
                //Console.WriteLine(c);
                // obtengo el espacio laente de a
                //var o = perceptron.MapGet(fonts.ToArray()[15]);
                //Console.WriteLine(o);
                //double[] newp = new double[] { (c[0] + o[0]) / 2.0, (c[1] + o[1]) / 2.0 };

                //obtengo el espacio laente de a
                var z = perceptron.MapGet(fonts.ToArray()[26], (int)Math.Floor(config.Layers.Length / 2.0));
                Console.WriteLine(z);
                //obtengo el espacio laente de a
                var t = perceptron.MapGet(fonts.ToArray()[20], (int)Math.Floor(config.Layers.Length / 2.0));
                Console.WriteLine(t);
                double[] newp = new double[] { (z[0] + t[0]) / 2.0, (z[1] + t[1]) / 2.0 };

                Console.WriteLine(newp[0]);
                var acpy = perceptron.MapTry(Vector<double>.Build.DenseOfArray(newp),(int)Math.Floor(config.Layers.Length / 2.0));
                Console.WriteLine(acpy);
                foreach (double bin in acpy)
                {
                    var bin2 = bin * 32;
                    int m = (int)Math.Round(bin2);
                    string str = Convert.ToString(m, 2); ;
                    Console.WriteLine(str);
                }
                // --------------------------------------------------------------------------------------
                var error = fonts.Aggregate(0.0, (sum, v) => sum + perceptron.CalculateError(new Vector<double>[] { v }, new Vector<double>[] { v }));
            }
            else
            {
                var input = ParseCSV(config.TrainingInput, 1);
                perceptron.Learn(input.ToArray(), input.ToArray(), new Vector<double>[] { }, new Vector<double>[] { }, config.Batch, config.MinError, config.Epochs);
                var error = input.Aggregate(0.0, (sum, v) => sum + perceptron.CalculateError(new Vector<double>[] { v }, new Vector<double>[] { v }));
            }
            return;
        }

        /// <summary>
        /// Autoencoder
        /// </summary>
        /// <param name="learningRate">Learning rate</param>
        /// <param name="alpha">Momentum </param>
        /// <param name="epochs">Number of epochs</param>
        /// <param name="batch">Batch size</param>
        static void Ej2(int epochs, double learningRate = 0.01, int batch = 16, double alpha = 0.8, bool train = true)
        {
            Func<double, double> activationFunction, activationFunctionD;
            activationFunction = SpecialFunctions.Logistic;
            activationFunctionD = val => SpecialFunctions.Logistic(val) * (1 - SpecialFunctions.Logistic(val));
            int[] layers = new int[] { 3888, 128, 2, 128, 3888 };
            var actFunctions = new Func<double, double>[layers.Length - 1];
            var actDFunctions = new Func<double, double>[layers.Length - 1];
            Array.Fill(actFunctions, activationFunction);
            Array.Fill(actDFunctions, activationFunctionD);

            var images = ImageHelper.LoadImagesFromDirectory("Images/Microsoft");
            var perceptron = new MultiLayerPerceptron(layers, learningRate, actFunctions, actDFunctions, false, true);
            if (File.Exists(weightsFile))
                perceptron.W = RestoreWeights(layers, "weights.csv");
            else
                perceptron.InitializeRandom();
            perceptron.Alpha = alpha;

            if(train)
            {
                Console.CancelKeyPress += (sender, e) =>
                {
                    Console.WriteLine("Saving...");
                    PersistWeights(currentWeights, weightsFile);
                };
                currentWeights = perceptron.W;
                perceptron.UpdatedWeights += w => currentWeights = w;

                perceptron.Learn(images, images, new Vector<double>[] { }, new Vector<double>[] { }, batch, 1, epochs);

                PersistWeights(perceptron.W, weightsFile);
                return;
            }

            var encoder = new MultiLayerPerceptron(layers[0..3], 0, actFunctions[0..3], actDFunctions[0..3], false, false);
            encoder.W = perceptron.W[0..3];
            var encodings = images.Select(img => encoder.Map(img).ToArray()).ToArray();

            var output = perceptron.Map(images[0]);
            ImageHelper.ImageFromVector(36, 36, output).Save("smile.png");
        }
    }
}
