﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;
using Accord.Math.Optimization;
using MathNet.Numerics.Distributions;
using System.Diagnostics;

namespace TP5
{
    public class MultiLayerPerceptron : Perceptron
    {
        public double LearningRate { get; set; }
        public bool AdaptiveLearningRate { get; set; }
        private int[] layers;
        public Matrix<double>[] W;
        private Func<double, double>[] g;
        private Func<double, double>[] gprime;
        private bool momentum;
        public double Alpha = 0.8;


        public event Action<Matrix<double>[]> UpdatedWeights;


        public MultiLayerPerceptron(int[] layers, double learningRate, Func<double, double>[] g, Func<double, double>[] gprime, bool adaptiveLearningRate, bool momentum)
        {
            this.W = new Matrix<double>[layers.Length];
            this.layers = layers;
            this.LearningRate = learningRate;
            this.AdaptiveLearningRate = adaptiveLearningRate;
            this.g = g;
            this.gprime = gprime;
            this.momentum = momentum;
        }
        public double CalculateError(Vector<double>[] input, Vector<double>[] desiredOutput) => CalculateError(input, desiredOutput, W);
        private double CalculateError(Vector<double>[] input, Vector<double>[] desiredOutput, Matrix<double>[] w)
        {
            double sum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                double dif = (desiredOutput[i] - Map(input[i], w)).L2Norm();
                sum += dif * dif;
            }
            double error = sum * 0.5d;
            Console.WriteLine($"Error: {error}");
            return error;
        }

        public void InitializeRandom()
        {
            for (int i = 0; i < layers.Length - 1; i++)
            {
                W[i] = CreateMatrix.Random<double>(layers[i+1], layers[i] + 1, new ContinuousUniform(-1, 1));
            }
        }

        public Vector<double> Map(Vector<double> input) => Map(input, W);
        private Vector<double> Map(Vector<double> input, Matrix<double>[] w)
        {
            
            
            Vector<double> V = input;
            for (int k = 0; k < layers.Length - 1; k++)
            {
                //Agrego el valor 1 al principio de cada salida intermedia.
                if (k > 0 || (w[k].ColumnCount != V.Count))
                {
                    var withOne = Vector<double>.Build.Dense(V.Count + 1);
                    withOne[0] = 1;
                    V.CopySubVectorTo(withOne, 0, 1, V.Count);
                    V = withOne;
                }
                V = (w[k] * V).Map(g[k]);
            }
            return V;
        }

        //dado el espacio latente te devuelve la letra que representa
        public Vector<double> MapTry(Vector<double> input)
        {
            Matrix<double>[] w = W;
            Vector<double> V = input;
            for (int k = 2; k < layers.Length; k++)
            {
                //Agrego el valor 1 al principio de cada salida intermedia.
                if (k > 0 || (w[k].ColumnCount != V.Count))
                    V = Vector<double>.Build.DenseOfEnumerable(new double[] { 1 }.Concat(V));
                V = (w[k] * V).Map(g[k]);
            }
            return V;
        }

        // obtiene el espacio latente de una letra dada
        public Vector<double> MapGet(Vector<double> input)
        {
            Matrix<double>[] w = W;
            Vector<double> V = input;
            for (int k = 0; k < 2; k++)
            {
                //Agrego el valor 1 al principio de cada salida intermedia.
                if (k > 0 || (w[k].ColumnCount != V.Count))
                    V = Vector<double>.Build.DenseOfEnumerable(new double[] { 1 }.Concat(V));
                V = (w[k] * V).Map(g[k]);
            }
            return V;
        }
        private double optimizing(Vector<double>[] input,
            Vector<double>[] V,
            Matrix<double>[] w,
            int M,
            Vector<double>[] h,
            Vector<double>[] delta,
            Vector<double>[] trainingOutput,
            Matrix<double>[] deltaW,
            int batch,
            int[] rand)
        {
            Func<double, double> function = x => loop(input, V, M, h, delta, trainingOutput, deltaW, batch, x, rand, w.Select(wi => Matrix<double>.Build.DenseOfMatrix(wi)).ToArray());
            BrentSearch search = new BrentSearch(function, 0, 1);
            bool success = search.Minimize();   
            double min = search.Solution;
            
            return min;
        }


        

        private double loop(
            Vector<double>[] input,
            Vector<double>[] V,
            int M,
            Vector<double>[] h,
            Vector<double>[] delta,
            Vector<double>[] trainingOutput,
            Matrix<double>[] deltaW,
            int batch,
            double lr,
            int[] rand,
            Matrix<double>[] w)
        {
            int j;
            double error_min = -1;
            double error;
            for (j = 0; j < input.Length; j++)
            {
                int index = rand[j];
                V[0] = input[index];
                for(int k = 0; k < M; k++)
                {
                    h[k] = w[k] * V[k];
                    Vector<double> activationOutput = h[k].Map(g[k]);
                    //Agrego el valor 1 al principio de cada salida intermedia.
                    V[k + 1] = k + 1 < M ? Vector<double>.Build.DenseOfEnumerable(new double[]{1}.Concat(activationOutput)) : activationOutput;
                }
                delta[M - 1] = h[M - 1].Map(gprime[M - 1]).PointwiseMultiply(trainingOutput[index] - V[M]);
                for (int k = M - 1; k > 0; k--)
                {
                    var aux = w[k].TransposeThisAndMultiply(delta[k]);
                    //Salteo el primer valor ya que corresponde al delta de la neurona extra para el umbral (siempre tiene que tener activacion igual a 1).
                    aux = Vector<double>.Build.DenseOfEnumerable(aux.Skip(1));
                    delta[k - 1] = h[k - 1].Map(gprime[k - 1]).PointwiseMultiply(aux);

                }
                
                for (int k = 0; k < M; k++)
                {
                    if(deltaW[k] != null)
                        deltaW[k] += lr * delta[k].OuterProduct(V[k]);
                    else
                        deltaW[k] = lr * delta[k].OuterProduct(V[k]);
                }

                if(j % batch == 0)
                {
                    for (int k = 0; k < M; k++)
                        w[k] += deltaW[k];
                    error = CalculateError(input, trainingOutput, w);
                    if (error_min == -1 || error < error_min)
                        error_min = error;
                    //Reinicio los deltaW para el proximo lote.
                    Array.Fill(deltaW, null);
                }
            }
            if (j % batch != 0)
            {
                for (int k = 0; k < M; k++)
                    w[k] += deltaW[k];
                error = CalculateError(input, trainingOutput, w);
                if (error < error_min)
                    error_min = error;
            }
            return error_min;
        }

        public void Learn(
            Vector<double>[] trainingInput, 
            Vector<double>[] trainingOutput,
            Vector<double>[] testInput,
            Vector<double>[] testOutput,
            int batch, 
            double minError, 
            int epochs)
        {
            Vector<double>[] input = new Vector<double>[trainingInput.Length];
            //Agrego el valor 1 al principio del input.
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = Vector<double>.Build.DenseOfEnumerable(new double[]{1}.Concat(trainingInput[i]));
            }
            int M = layers.Length - 1;
            Matrix<double>[] w = new Matrix<double>[M];
            for(int i = 0; i < M; i++)
            {
                w[i] = W[i].Clone();
            }
            var w_min = w;
            Vector<double>[] V = new Vector<double>[M + 1];
            Vector<double>[] delta = new Vector<double>[M];
            Matrix<double>[] deltaW = new Matrix<double>[M];
            Vector<double>[] h = new Vector<double>[M];
            double error;
            double error_min = double.MaxValue;

            Stopwatch sw = new Stopwatch();
            for(int i = 0; i < epochs && error_min > minError; i++)
            {
                sw.Restart();
                Console.WriteLine($"Epoch: {i}");

                int[] rand = Combinatorics.GeneratePermutation(input.Length);
                double lr = AdaptiveLearningRate ? optimizing(input, V, w, M, h, delta, trainingOutput, deltaW, batch, rand) : LearningRate;
                int j;
                for (j = 0; j < input.Length; j++)
                {
                    int index = rand[j];
                    V[0] = input[index];
                    for (int k = 0; k < M; k++)
                    {
                        h[k] = w[k] * V[k];
                        Vector<double> activationOutput = h[k].Map(g[k]);
                        //Agrego el valor 1 al principio de cada salida intermedia.

                        if (k + 1 < M)
                        {
                            var withOne = Vector<double>.Build.Dense(activationOutput.Count + 1);
                            withOne[0] = 1;
                            activationOutput.CopySubVectorTo(withOne, 0, 1, activationOutput.Count);
                            V[k + 1] = withOne;
                        }
                        else
                            V[k + 1] = activationOutput;
                        
                    }
                    delta[M - 1] = h[M - 1].Map(gprime[M - 1]).PointwiseMultiply(trainingOutput[index] - V[M]);
                    
                    for (int k = M - 1; k > 0; k--)
                    {
                        var aux = w[k].TransposeThisAndMultiply(delta[k]);
                        //Salteo el primer valor ya que corresponde al delta de la neurona extra para el umbral (siempre tiene que tener activacion igual a 1).
                        aux = aux.SubVector(1, aux.Count - 1);
                        delta[k - 1] = h[k - 1].Map(gprime[k - 1]).PointwiseMultiply(aux);

                    }

                    for (int k = 0; k < M; k++)
                    {
                        if (deltaW[k] != null)
                            deltaW[k] = momentum ? lr * delta[k].OuterProduct(V[k]) + Alpha * deltaW[k] : lr * delta[k].OuterProduct(V[k]) + deltaW[k];
                        else
                            deltaW[k] = lr * delta[k].OuterProduct(V[k]);
                    }

                    if (j % batch == 0)
                    {
                        for (int k = 0; k < M; k++)
                            w[k] += deltaW[k];
                        error = CalculateError(input, trainingOutput, w);
                        if (error < error_min)
                        {
                            error_min = error;
                            w_min = w;
                            UpdatedWeights(w_min);
                        }
                        //Reinicio los deltaW para el proximo lote.
                        Array.Fill(deltaW, null);
                    }
                }
                if (j % batch != 0)
                {
                    for (int k = 0; k < M; k++)
                        w[k] += deltaW[k];
                    error = CalculateError(input, trainingOutput, w);
                    if (error < error_min)
                    {
                        error_min = error;
                        w_min = w;
                        UpdatedWeights(w_min);
                    }
                }
                sw.Stop();
                Console.WriteLine($"Epoch {i} finished in {sw.Elapsed.TotalSeconds}s");
            }
            W = w_min;
        }
    }
}
