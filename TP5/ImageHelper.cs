using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Linq;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Drawing.Imaging;

namespace TP5
{
    public class ImageHelper
    {
        private static byte[] ImageToArray(Bitmap bmp)
        {
            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadWrite, bmp.PixelFormat);
            IntPtr ptr = bmpData.Scan0;
            int bytes = Math.Abs(bmpData.Stride) * bmp.Height;
            byte[] rgbValues = new byte[bytes];
            Marshal.Copy(ptr, rgbValues, 0, bytes);
            return rgbValues;
        }

        private static byte[] ImageToArray2(Bitmap bmp)
        {
            return Enumerable.Range(0, bmp.Width)
                .SelectMany(x => Enumerable.Range(0, bmp.Height)
                    .Select(y => (x, y)))
                .Select(point => bmp.GetPixel(point.x, point.y))
                .SelectMany(color => new byte[] { color.R, color.G, color.B })
                .ToArray();
        }

        public static Bitmap ImageFromVector(int width, int height, Vector<double> vector)
        {
            var bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            for(int i = 0, idx = 0; i < width; i++)
                for(int j = 0; j < height; j++)
                {
                    int r = (int)(vector[idx++] * 255);
                    int g = (int)(vector[idx++] * 255);
                    int b = (int)(vector[idx++] * 255);
                    bitmap.SetPixel(i, j, Color.FromArgb(r, g, b));
                }
            return bitmap;
        }

        public static Vector<double>[] LoadImagesFromDirectory(string directory)
        {
            ImageConverter converter = new ImageConverter();
            VectorBuilder<double> vectorBuilder = Vector<double>.Build;
            return Directory
                .GetFiles(directory, "*.png")
                .Select(path => new Bitmap(path))
                .Select(ImageToArray2)
                .Select(arr => vectorBuilder.DenseOfEnumerable(arr.Select(pixel => pixel / 255.0)))
                .ToArray();
        }
    }
}
