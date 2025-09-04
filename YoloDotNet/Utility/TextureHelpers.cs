using System;
using System.Collections.Generic;
using System.Linq;
using Stride.Graphics;
using Stride.Core.Mathematics;

namespace YoloDotNet.Utility
{
    public static class TextureHelpers
    {
        // Compositor: assumes segmentations are already filtered/cropped/scaled upstream.
        // Renders into a possibly larger output texture (outputTexWidth/outputTexHeight).
        public static Texture TextureFromSegmentations(
            GraphicsDevice device,
            int width,
            int height,
            IEnumerable<Segmentation> segmentations,
            bool doRGB = false,
            Color4 tint = default,
            bool useSegmentationColor = true,
            int outputTexWidth = 0,
            int outputTexHeight = 0)
        {
            if (segmentations is null)
                segmentations = Enumerable.Empty<Segmentation>();

            if (tint == default)
                tint = new Color4(1.0f, 1.0f, 1.0f, 1.0f);

            int outW = outputTexWidth > 0 ? outputTexWidth : width;
            int outH = outputTexHeight > 0 ? outputTexHeight : height;

            var pixelFormat = doRGB ? PixelFormat.R8G8B8A8_UNorm : PixelFormat.R8_UNorm;
            var bytesPerPixel = doRGB ? 4 : 1;
            var finalMaskData = new byte[outW * outH * bytesPerPixel];

            foreach (var seg in segmentations)
            {
                if (seg?.BitPackedPixelMask is null || seg.BitPackedPixelMask.Length == 0)
                    continue;

                var bbox = seg.BoundingBox;
                if (bbox.Width <= 0 || bbox.Height <= 0)
                    continue;

                // Choose color: per-segmentation or global tint
                var chosen = useSegmentationColor ? seg.Color : tint;
                if (chosen == default)
                    chosen = new Color4(1f, 1f, 1f, 1f);

                // Premultiplied bytes
                byte aByte = (byte)(Math.Clamp(chosen.A, 0f, 1f) * 255);
                byte rPremul = (byte)(Math.Clamp(chosen.R, 0f, 1f) * aByte);
                byte gPremul = (byte)(Math.Clamp(chosen.G, 0f, 1f) * aByte);
                byte bPremul = (byte)(Math.Clamp(chosen.B, 0f, 1f) * aByte);

                // Clip the draw rect against the output canvas only
                int drawLeft = Math.Max(0, bbox.Left);
                int drawTop = Math.Max(0, bbox.Top);
                int drawRight = Math.Min(outW, bbox.Right);
                int drawBottom = Math.Min(outH, bbox.Bottom);

                if (drawRight <= 0 || drawBottom <= 0 || drawLeft >= outW || drawTop >= outH)
                    continue;

                // Compute where to start reading from the mask if bbox is partially outside
                int startXInMask = Math.Max(0, -bbox.Left);
                int startYInMask = Math.Max(0, -bbox.Top);

                var mask = seg.BitPackedPixelMask;
                int maskStride = bbox.Width;

                for (int ty = drawTop; ty < drawBottom; ty++)
                {
                    int yInMask = startYInMask + (ty - drawTop);
                    for (int tx = drawLeft; tx < drawRight; tx++)
                    {
                        int xInMask = startXInMask + (tx - drawLeft);

                        int i = yInMask * maskStride + xInMask;
                        int byteIndex = i >> 3;
                        int bitIndex = i & 7;

                        if ((mask[byteIndex] & (1 << bitIndex)) == 0)
                            continue;

                        int destIndex = (ty * outW + tx) * bytesPerPixel;

                        if (doRGB)
                        {
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
                            if (aByte > finalMaskData[destIndex])
                                finalMaskData[destIndex] = aByte;
                        }
                    }
                }
            }

            return Texture.New2D(
                device,
                outW,
                outH,
                pixelFormat,
                finalMaskData,
                TextureFlags.ShaderResource,
                GraphicsResourceUsage.Immutable);
        }
    }
}
