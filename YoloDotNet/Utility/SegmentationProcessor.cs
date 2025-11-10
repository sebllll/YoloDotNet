using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using YoloDotNet.Models;
using Stride.Graphics;
using Stride.Core.Mathematics;

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

        public static Color4 MultiplyColor(Color4 color, float factor, bool factorToAlpha = false)
        {
            float r = Math.Clamp(color.R * factor, 0f, 1f);
            float g = Math.Clamp(color.G * factor, 0f, 1f);
            float b = Math.Clamp(color.B * factor, 0f, 1f);
            float a = factorToAlpha ? Math.Clamp(color.A * factor, 0f, 1f) : color.A;
            return new Color4(r, g, b, a);
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

                if (segmentation?.BitPackedPixelMask != null && segmentation.BitPackedPixelMask.Length > 0)
                {
                    var bbox = segmentation.BoundingBox;
                    for (int y = 0; y < bbox.Height; y++)
                    {
                        for (int x = 0; x < bbox.Width; x++)
                        {
                            int bitIndex = y * bbox.Width + x;
                            int byteIndex = bitIndex / 8;
                            int bitInByte = bitIndex % 8;

                            if ((segmentation.BitPackedPixelMask[byteIndex] & (1 << bitInByte)) != 0)
                            {
                                int absoluteX = bbox.Left + x;
                                int absoluteY = bbox.Top + y;
                                byte confidenceValue = 255; // No per-pixel confidence, use full value

                                if (absoluteX >= 0 && absoluteX < width && absoluteY >= 0 && absoluteY < height)
                                {
                                    int index = absoluteY * width + absoluteX;
                                    pixelData[index] = confidenceValue;
                                }
                            }
                        }
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

                if (segmentation.BitPackedPixelMask != null && segmentation.BitPackedPixelMask.Length > 0)
                {
                    var bbox = segmentation.BoundingBox;
                    for (int y = 0; y < bbox.Height; y++)
                    {
                        for (int x = 0; x < bbox.Width; x++)
                        {
                            int bitIndex = y * bbox.Width + x;
                            int byteIndex = bitIndex / 8;
                            int bitInByte = bitIndex % 8;

                            if ((segmentation.BitPackedPixelMask[byteIndex] & (1 << bitInByte)) != 0)
                            {
                                int absoluteX = bbox.Left + x;
                                int absoluteY = bbox.Top + y;
                                byte confidenceValue = 255; // No per-pixel confidence, use full value

                                if (absoluteX >= 0 && absoluteX < width && absoluteY >= 0 && absoluteY < height)
                                {
                                    int index = absoluteY * width + absoluteX;
                                    pixelData[index] = confidenceValue;
                                }
                            }
                        }
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

            //Build color bitmap
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
                    if (segmentation?.BitPackedPixelMask == null || segmentation.BitPackedPixelMask.Length == 0)
                    {
                        segmentationIndex++;
                        continue;
                    }

                    int colorindex = segmentationIndex % numColors;
                    SKColor color = sKColors[colorindex];

                    //Moltiplica il valore di confidenza per il colore
                    SKColor pixelColor = confidenceToColor ? MultiplyColor(color, (float)segmentation.Confidence, confidenceToAlpha) : color;

                    var bbox = segmentation.BoundingBox;
                    for (int y = 0; y < bbox.Height; y++)
                    {
                        for (int x = 0; x < bbox.Width; x++)
                        {
                            int bitIndex = y * bbox.Width + x;
                            int byteIndex = bitIndex / 8;
                            int bitInByte = bitIndex % 8;

                            if ((segmentation.BitPackedPixelMask[byteIndex] & (1 << bitInByte)) != 0)
                            {
                                int absoluteX = bbox.Left + x;
                                int absoluteY = bbox.Top + y;

                                if (absoluteX >= 0 && absoluteX < width && absoluteY >= 0 && absoluteY < height)
                                {
                                    int index = absoluteY * width + absoluteX;
                                    pixelData[index] = (uint)pixelColor;
                                }
                            }
                        }
                    }
                    segmentationIndex++;
                }
            }

            // Build SKImage from bitmap
            return SKImage.FromBitmap(colorBitmap);
        }

        public static Texture SegmentationsToTexture2D(GraphicsDevice device, int width, int height, IEnumerable<Segmentation> segmentations, IEnumerable<Color4> colors, bool confidenceToColor = true, bool confidenceToAlpha = false)
        {
            List<Color4> strideColors = new List<Color4>();
            int numColors = 0;
            //set default color is non provided
            if (colors == null || !colors.Any())
            {
                strideColors.Add(Color4.White); 
                numColors = 1;
            }
            else
            {
                strideColors = colors.ToList(); 
                numColors = strideColors.Count();
            }

            // Initialize pixel data array (defaults to transparent black)
            uint[] pixelData = new uint[width * height];

            if (segmentations == null || !segmentations.Any())
            {
                // Return a texture initialized with default pixelData (transparent black)
                return Texture.New2D<uint>(device, width, height, PixelFormat.R8G8B8A8_UNorm, pixelData, TextureFlags.ShaderResource, GraphicsResourceUsage.Default);
            }
            
            int segmentationIndex = 0;
            foreach (var segmentation in segmentations)
            {
                if (segmentation?.BitPackedPixelMask == null || segmentation.BitPackedPixelMask.Length == 0)
                {
                    segmentationIndex++;
                    continue;
                }

                int colorindex = segmentationIndex % numColors;
                Color4 baseColor = strideColors[colorindex];

                // Apply confidence to color and/or alpha using Color4
                Color4 pixelColorWithConfidence = baseColor;
                if (confidenceToColor)
                {
                    // Use the Color4 overload of MultiplyColor
                    pixelColorWithConfidence = MultiplyColor(baseColor, (float)segmentation.Confidence, confidenceToAlpha);
                }
                else if (confidenceToAlpha) // Apply confidence to alpha only if confidenceToColor is false
                {
                    float newAlpha = Math.Clamp(baseColor.A * (float)segmentation.Confidence, 0f, 1f);
                    pixelColorWithConfidence = new Color4(baseColor.R, baseColor.G, baseColor.B, newAlpha);
                }
                // else: pixelColorWithConfidence remains baseColor as initialized

                var bbox = segmentation.BoundingBox;
                for (int y = 0; y < bbox.Height; y++)
                {
                    for (int x = 0; x < bbox.Width; x++)
                    {
                        int bitIndex = y * bbox.Width + x;
                        int byteIndex = bitIndex / 8;
                        int bitInByte = bitIndex % 8;

                        if ((segmentation.BitPackedPixelMask[byteIndex] & (1 << bitInByte)) != 0)
                        {
                            int absoluteX = bbox.Left + x;
                            int absoluteY = bbox.Top + y;

                            if (absoluteX >= 0 && absoluteX < width && absoluteY >= 0 && absoluteY < height)
                            {
                                int index = absoluteY * width + absoluteX;
                                // Convert Color4 to uint (RGBA for Stride Texture R8G8B8A8_UNorm)
                                pixelData[index] = (uint)pixelColorWithConfidence.ToRgba();
                            }
                        }
                    }
                }
                segmentationIndex++;
            }

            // Create Stride Texture2D from pixel data
            // The uint[] data is RGBA (R at LSB), which for R8G8B8A8_UNorm means R,G,B,A byte order in memory.
            var texture = Texture.New2D<uint>(device, width, height, PixelFormat.R8G8B8A8_UNorm, pixelData, TextureFlags.ShaderResource, GraphicsResourceUsage.Default);
            
            return texture;
        }
    }
}
