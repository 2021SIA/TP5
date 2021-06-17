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
        static List<Vector<double>> ParseTSV(string path, int recordLines)
        {
            List<Vector<double>> trainingInput = new List<Vector<double>>();
            using (var reader = new StreamReader(path))
            {
                while (!reader.EndOfStream)
                {
                    var line = "";
                    for (var i = 0; i < recordLines; i++)
                        line += " " + reader.ReadLine();
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

        static void Main(string[] args)
        {
            FirstExcercise();
        }

        static void FirstExcercise()
        {
            Func<double, double>  activationFunction = val=> 1.0 / (1.0 + Math.Exp(-val));
            Func<double, double> activationFunctionD = val => (1.0 / (1.0 + Math.Exp(-val))) * (1 - (1.0 / (1.0 + Math.Exp(-val))));
            //Func<double, double> activationFunction = val => val;
            //Func<double, double> activationFunctionD = val => 1;
            int[] layers = new int[] { /*7*/4, 2,4, 7 };
            var actFunctions = new Func<double, double>[layers.Length];
            var actDFunctions = new Func<double, double>[layers.Length];
            Array.Fill(actFunctions, activationFunction);
            Array.Fill(actDFunctions, activationFunctionD);

            var perceptron = new MultiLayerPerceptron(layers, 0.1, actFunctions, actDFunctions, false, true);
            var fontRaw = new double[32][] {
                new double[] {0x0e, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0f},   // 0x40, @
                new double[] {0x04, 0x0a, 0x11, 0x11, 0x1f, 0x11, 0x11},   // 0x41, A
                new double[] {0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e},   // 0x42, B
                new double[] {0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e},   // 0x43, C
                new double[] {0x1e, 0x09, 0x09, 0x09, 0x09, 0x09, 0x1e},   // 0x44, D
                new double[] {0x1f, 0x10, 0x10, 0x1c, 0x10, 0x10, 0x1f},   // 0x45, E
                new double[] {0x1f, 0x10, 0x10, 0x1f, 0x10, 0x10, 0x10},   // 0x46, F
                new double[] {0x0e, 0x11, 0x10, 0x10, 0x13, 0x11, 0x0f},   // 0x37, G
                new double[] {0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11},   // 0x48, H
                new double[] {0x0e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e},   // 0x49, I
                new double[] {0x1f, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0c},   // 0x4a, J
                new double[] {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11},   // 0x4b, K
                new double[] {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f},   // 0x4c, L
                new double[] {0x11, 0x1b, 0x15, 0x11, 0x11, 0x11, 0x11},   // 0x4d, M
                new double[] {0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11},   // 0x4e, N
                new double[] {0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e},   // 0x4f, O
                new double[] {0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10},   // 0x50, P
                new double[] {0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d},   // 0x51, Q
                new double[] {0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11},   // 0x52, R
                new double[] {0x0e, 0x11, 0x10, 0x0e, 0x01, 0x11, 0x0e},   // 0x53, S
                new double[] {0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},   // 0x54, T
                new double[] {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e},   // 0x55, U
                new double[] {0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04},   // 0x56, V
                new double[] {0x11, 0x11, 0x11, 0x15, 0x15, 0x1b, 0x11},   // 0x57, W
                new double[] {0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11},   // 0x58, X
                new double[] {0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04},   // 0x59, Y
                new double[] {0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f},   // 0x5a, Z
                new double[] {0x0e, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0e},   // 0x5b, [
                new double[] {0x10, 0x10, 0x08, 0x04, 0x02, 0x01, 0x01},   // 0x5c, \\
                new double[] {0x0e, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0e},   // 0x5d, ]
                new double[] {0x04, 0x0a, 0x11, 0x00, 0x00, 0x00, 0x00},   // 0x5e, ^
                new double[] {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f}   // 0x5f, _
            };
            var fonts = fontRaw.Select(row => Vector<double>.Build.DenseOfEnumerable(row.Select(n => n/32.0)));
            perceptron.Learn(fonts.ToArray(), fonts.ToArray(), new Vector<double>[] { }, new Vector<double>[] { }, 1, 0, 10000);

            // ----------------------------------------------- RESOLUCION 1.A.4 -------------
            // obtengo el espacio laente de a
            //var c = perceptron.MapGet(fonts.ToArray()[3]);
            //Console.WriteLine(c);
            // obtengo el espacio laente de a
            //var o = perceptron.MapGet(fonts.ToArray()[15]);
            //Console.WriteLine(o);
            //double[] newp = new double[] { (c[0] + o[0]) / 2.0, (c[1] + o[1]) / 2.0 };

            //obtengo el espacio laente de a
            var z = perceptron.MapGet(fonts.ToArray()[26]);
            Console.WriteLine(z);
            //obtengo el espacio laente de a
            var t = perceptron.MapGet(fonts.ToArray()[20]);
            Console.WriteLine(t);
            double[] newp = new double[] { (z[0] + t[0]) / 2.0, (z[1] + t[1]) / 2.0 };

            Console.WriteLine(newp[0]);
            var acpy = perceptron.MapTry(Vector<double>.Build.DenseOfArray(newp));
            Console.WriteLine(acpy);
            foreach (double bin in acpy)
            {
                var bin2 = bin * 32;
                int m = (int)Math.Round(bin2);
                string str = Convert.ToString(m, 2); ;
                Console.WriteLine(str);
            }
            // --------------------------------------------------------------------------------------
            var error = fonts.Aggregate(0.0,(sum, v) => sum + perceptron.CalculateError(new Vector<double>[] { v }, new Vector<double>[] { v }));
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
