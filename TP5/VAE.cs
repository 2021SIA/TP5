using System;
using System.Collections.Generic;
using System.Text;

namespace TP5
{
    public class VAE
    {
        public MultiLayerPerceptron Encoder { get; set; }
        public MultiLayerPerceptron Decoder { get; set; }

        public VAE(MultiLayerPerceptron encoder, MultiLayerPerceptron decoder)
        {
            this.Encoder = encoder;
            this.Decoder = decoder;
        }
    }
}
