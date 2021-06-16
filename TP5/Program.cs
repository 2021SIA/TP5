using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace TP5
{
    class Program
    {

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

        static void Main(string[] args)
        {
            Func<double, double> activationFunction = val => val;
            Func<double, double> activationFunctionD = val => 1;
            int[] layers = new int[] { /*7*/ 3, 2, 3, 7 };
            var actFunctions = new Func<double, double>[layers.Length];
            var actDFunctions = new Func<double, double>[layers.Length];
            Array.Fill(actFunctions, activationFunction);
            Array.Fill(actDFunctions, activationFunctionD);

            var perceptron = new MultiLayerPerceptron(layers, 0.01, actFunctions, actDFunctions, false);
            var fontRaw = new double[32][] {
                   new double[] {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // 0x20, space
                   new double[] {0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04},   // 0x21, !
                   new double[] {0x09, 0x09, 0x12, 0x00, 0x00, 0x00, 0x00},   // 0x22, "
                   new double[] {0x0a, 0x0a, 0x1f, 0x0a, 0x1f, 0x0a, 0x0a},   // 0x23, #
                   new double[] {0x04, 0x0f, 0x14, 0x0e, 0x05, 0x1e, 0x04},   // 0x24, $
                   new double[] {0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13},   // 0x25, %
                   new double[] {0x04, 0x0a, 0x0a, 0x0a, 0x15, 0x12, 0x0d},   // 0x26, &
                   new double[] {0x04, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00},   // 0x27, '
                   new double[] {0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02},   // 0x28, (
                   new double[] {0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08},   // 0x29, )
                   new double[] {0x04, 0x15, 0x0e, 0x1f, 0x0e, 0x15, 0x04},   // 0x2a, *
                   new double[] {0x00, 0x04, 0x04, 0x1f, 0x04, 0x04, 0x00},   // 0x2b, +
                   new double[] {0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x08},   // 0x2c, ,
                   new double[] {0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00},   // 0x2d, -
                   new double[] {0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c},   // 0x2e, .
                   new double[] {0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10},   // 0x2f, /
                   new double[] {0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e},   // 0x30, 0
                   new double[] {0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e},   // 0x31, 1
                   new double[] {0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f},   // 0x32, 2
                   new double[] {0x0e, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0e},   // 0x33, 3
                   new double[] {0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02},   // 0x34, 4
                   new double[] {0x1f, 0x10, 0x1e, 0x01, 0x01, 0x11, 0x0e},   // 0x35, 5
                   new double[] {0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e},   // 0x36, 6
                   new double[] {0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},   // 0x37, 7
                   new double[] {0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e},   // 0x38, 8
                   new double[] {0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c},   // 0x39, 9
                   new double[] {0x00, 0x0c, 0x0c, 0x00, 0x0c, 0x0c, 0x00},   // 0x3a, :
                   new double[] {0x00, 0x0c, 0x0c, 0x00, 0x0c, 0x04, 0x08},   // 0x3b, ;
                   new double[] {0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02},   // 0x3c, <
                   new double[] {0x00, 0x00, 0x1f, 0x00, 0x1f, 0x00, 0x00},   // 0x3d, =
                   new double[] {0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08},   // 0x3e, >
                   new double[] {0x0e, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04}   // 0x3f, ?
            };
            var fonts = fontRaw.Select(row => Vector<double>.Build.DenseOfEnumerable(row.Select(n => n/31.0)));
            perceptron.Learn(fonts.ToArray(), fonts.ToArray(), new Vector<double>[] { }, new Vector<double>[] { }, 1, 0, 10000);
            //Vector<Double> v = Vector<double>.Build.Dense(2, i => i/2);
            //Console.WriteLine(v);
            //var arr = perceptron.MapTry(v);
            //Console.WriteLine(arr);
            //foreach (double bin in arr)
            //{
            //    Console.WriteLine(bin);
            //    long m = BitConverter.DoubleToInt64Bits(bin);
            //    string str = Convert.ToString(m, 2);;
            //    Console.WriteLine(str);
            //}
            
            var error = fonts.Aggregate(0.0,(sum, v) => sum + perceptron.CalculateError(new Vector<double>[] { v }, new Vector<double>[] { v }));
            return;
        }
    }
}
