using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using YoloDotNet.Models;

namespace YoloDotNet.Utility
{
    public class SegmentationProcessor
    {
        public static SKBitmap CreateGrayscaleBitmap(int width, int height)
        {
            // Crea una bitmap in scala di grigi
            SKImageInfo info = new SKImageInfo(width, height, SKColorType.Gray8, SKAlphaType.Opaque);
            SKBitmap bitmap = new SKBitmap(info);
            return bitmap;
        }

        public static SKBitmap CreateColorBitmap(int width, int height)
        {
            // Crea una bitmap in scala di grigi
            SKImageInfo info = new SKImageInfo(width, height, SKColorType.Bgra8888, SKAlphaType.Opaque);
            SKBitmap bitmap = new SKBitmap(info);
            return bitmap;
        }

        public static SKColor MultiplyColor(SKColor color, float factor, bool factorToAlpha  = false)
        {
            // Moltiplica ciascun componente del colore per il fattore specificato
            byte r = (byte)Math.Clamp(color.Red * factor, 0, 255);
            byte g = (byte)Math.Clamp(color.Green * factor, 0, 255);
            byte b = (byte)Math.Clamp(color.Blue * factor, 0, 255);
            byte a = factorToAlpha ? (byte)Math.Clamp(color.Alpha * factor, 0, 255) : color.Alpha; // Mantieni il valore alfa originale oppure moltiplica per factor

            // Restituisce il nuovo colore SKColor
            return new SKColor(r, g, b, a);
        }

        public static SKImage ProcessSegmentation(SKBitmap bitmap, Segmentation segmentation)
        {
            // Ottieni i dati dei pixel come array di byte
            IntPtr pixelsPtr = bitmap.GetPixels();
            int width = bitmap.Width;
            int height = bitmap.Height;

            unsafe
            {
                byte* pixelData = (byte*)pixelsPtr.ToPointer();

                // Itera sui pixel segmentati e assegna valori in scala di grigi
                foreach (var pixel in segmentation.SegmentedPixels)
                {
                    int x = pixel.X;
                    int y = pixel.Y;
                    byte confidenceValue = (byte)(pixel.Confidence * 255); // Converti il valore di confidenza in un valore di scala di grigi (0-255)

                    if (x >= 0 && x < width && y >= 0 && y < height)
                    {
                        int index = y * width + x;
                        pixelData[index] = confidenceValue;
                    }
                }
            }

            // Crea e restituisci una SKImage dalla bitmap processata
            return SKImage.FromBitmap(bitmap);
        }

        public static SKImage ProcessSegmentation(SKImage image, Segmentation segmentation)
        {
            
            // Ottieni le dimensioni dell'immagine
            int width = image.Width;
            int height = image.Height;

            // Crea una nuova bitmap in scala di grigi basata sulle dimensioni dell'immagine
            SKBitmap bitmap = CreateGrayscaleBitmap(width, height);

            if (segmentation == null)
            {
                //Log.Error("");
                return SKImage.FromBitmap(bitmap);
            }

            // Ottieni i dati dei pixel come array di byte
            IntPtr pixelsPtr = bitmap.GetPixels();

            unsafe
            {
                byte* pixelData = (byte*)pixelsPtr.ToPointer();

                // Itera sui pixel segmentati e assegna valori in scala di grigi
                foreach (var pixel in segmentation.SegmentedPixels)
                {
                    int x = pixel.X;
                    int y = pixel.Y;
                    byte confidenceValue = (byte)(pixel.Confidence * 255); // Converti il valore di confidenza in un valore di scala di grigi (0-255)

                    if (x >= 0 && x < width && y >= 0 && y < height)
                    {
                        int index = y * width + x;
                        pixelData[index] = confidenceValue;
                    }
                }
            }

            // Build SKImage from bitmap
            return SKImage.FromBitmap(bitmap);
        }

        public static SKImage ProcessSegmentations(SKImage image, IEnumerable<Segmentation> segmentations, IEnumerable<SKColor> colors, bool confidenceToColor = true, bool confidenceToAlpha = false)
        {
            // Ottieni le dimensioni dell'immagine
            int width = image.Width;
            int height = image.Height;

            List<SKColor> sKColors = new List<SKColor>();
            int numColors = 0;
            //set default color is non provided
            if (colors == null || colors.Count() == 0)
            {
                sKColors.Add(SKColors.White);
                numColors = 1;
            }
            else
            {
                sKColors = colors.ToList();
                numColors = colors.Count();
            }

            //Build color bitmam
            SKBitmap colorBitmap = CreateColorBitmap(width, height);

            if (segmentations == null || segmentations.Count() == 0)
            {
                //Log.Error("");
                return SKImage.FromBitmap(colorBitmap);
            }

            //Get pixels byte pointer
            IntPtr pixelsPtr = colorBitmap.GetPixels();

            unsafe
            {
                uint* pixelData = (uint*)pixelsPtr.ToPointer();

                int segmentationIndex = 0;
                foreach (var segmentation in segmentations)
                {
                    int colorindex = segmentationIndex % numColors;
                    SKColor color = sKColors[colorindex];

                    //Pixel iteration
                    foreach (var pixel in segmentation.SegmentedPixels)
                    {
                        int x = pixel.X;
                        int y = pixel.Y;

                        //Moltiplica il valore di confidenza per il colore
                        SKColor pixelColor = confidenceToColor ? MultiplyColor(color, (float)pixel.Confidence, confidenceToAlpha) : color;

                        if (x >= 0 && x < width && y >= 0 && y < height)
                        {
                            int index = y * width + x;
                            pixelData[index] = (uint)pixelColor;
                        }
                    }
                    segmentationIndex++;
                }
            }
            // Build SKImage from bitmap
            return SKImage.FromBitmap(colorBitmap);
        }
    }
}
