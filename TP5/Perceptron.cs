using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Text;

namespace TP5
{
    public interface Perceptron
    {
        List<double> Learn(Vector<double>[] trainingInput, Vector<double>[] trainingOutput, Vector<double>[] testInput, Vector<double>[] testOutput, int batch, double minError, int epochs);
        Vector<double> Map(Vector<double> input);
        double CalculateError(Vector<double>[] input, Vector<double>[] desiredOutput);
    }
}
