using System;
using System.Collections.Generic;
using System.Linq;
using Stride.Graphics;
using Stride.Core.Mathematics;

namespace YoloDotNet.Utility
{
    public static class TextureHelpers
    {
        // Simple compositor: assumes segmentations are already filtered/cropped/scaled upstream.
        public static Texture TextureFromSegmentations(
            GraphicsDevice device,
            int width,
            int height,
            IEnumerable<Segmentation> segmentations,
            bool doRGB = false,
            Color4 tint = default)
        {
            if (segmentations is null)
                segmentations = Enumerable.Empty<Segmentation>();

            // Default tint: opaque white
            if (tint == default)
            {
                tint = new Color4(1.0f, 1.0f, 1.0f, 1.0f);
            }

            var pixelFormat = doRGB ? PixelFormat.R8G8B8A8_UNorm : PixelFormat.R8_UNorm;
            var bytesPerPixel = doRGB ? 4 : 1;
            var finalMaskData = new byte[width * height * bytesPerPixel];

            // Precompute premultiplied color bytes for RGBA path
            byte aByte = (byte)(Math.Clamp(tint.A, 0f, 1f) * 255);
            byte rByte = (byte)(Math.Clamp(tint.R, 0f, 1f) * 255);
            byte gByte = (byte)(Math.Clamp(tint.G, 0f, 1f) * 255);
            byte bByte = (byte)(Math.Clamp(tint.B, 0f, 1f) * 255);
            byte rPremul = (byte)(rByte * aByte / 255);
            byte gPremul = (byte)(gByte * aByte / 255);
            byte bPremul = (byte)(bByte * aByte / 255);

            foreach (var seg in segmentations)
            {
                if (seg?.BitPackedPixelMask is null || seg.BitPackedPixelMask.Length == 0)
                    continue;

                var bbox = seg.BoundingBox;
                int left = Math.Clamp(bbox.Left, 0, width - 1);
                int top = Math.Clamp(bbox.Top, 0, height - 1);

                // Clamp width/height to canvas
                int boxWidth = Math.Max(0, Math.Min(bbox.Width, width - left));
                int boxHeight = Math.Max(0, Math.Min(bbox.Height, height - top));
                if (boxWidth == 0 || boxHeight == 0)
                    continue;

                var mask = seg.BitPackedPixelMask;

                // Unpack row-major bit-packed mask written with bbox.Width stride
                for (int y = 0; y < boxHeight; y++)
                {
                    for (int x = 0; x < boxWidth; x++)
                    {
                        int i = y * bbox.Width + x; // Important: use original bbox.Width for bit layout
                        int byteIndex = i >> 3;
                        int bitIndex = i & 7;

                        if (byteIndex >= mask.Length)
                            break;

                        if ((mask[byteIndex] & (1 << bitIndex)) == 0)
                            continue;

                        int tx = left + x;
                        int ty = top + y;
                        if ((uint)tx >= (uint)width || (uint)ty >= (uint)height)
                            continue;

                        int destIndex = (ty * width + tx) * bytesPerPixel;

                        if (doRGB)
                        {
                            // Max-over on alpha to combine overlaps (mirrors TransferResizedMask)
                            if (aByte > finalMaskData[destIndex + 3])
                            {
                                finalMaskData[destIndex + 0] = rPremul; // R
                                finalMaskData[destIndex + 1] = gPremul; // G
                                finalMaskData[destIndex + 2] = bPremul; // B
                                finalMaskData[destIndex + 3] = aByte;   // A
                            }
                        }
                        else
                        {
                            // Grayscale: write max value
                            if (aByte > finalMaskData[destIndex])
                            {
                                finalMaskData[destIndex] = aByte;
                            }
                        }
                    }
                }
            }

            return Texture.New2D(
                device,
                width,
                height,
                pixelFormat,
                finalMaskData,
                TextureFlags.ShaderResource,
                GraphicsResourceUsage.Immutable);
        }
    }
}
