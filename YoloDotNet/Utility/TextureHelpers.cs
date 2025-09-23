using System;
using System.Collections.Generic;
using System.Linq;
using Stride.Graphics;
using Stride.Core.Mathematics;

namespace YoloDotNet.Utility
{
    public static class TextureHelpers
    {
        // Optimized binary mask renderer.
        // confidenceToAlpha:
        //   false -> alpha = chosen.A
        //   true  -> alpha = chosen.A * seg.Confidence (still uniform per segmentation; mask is binary)
        public static Texture TextureFromSegmentations(
            GraphicsDevice device,
            int width,
            int height,
            IEnumerable<Segmentation> segmentations,
            bool doRGB = false,
            Color4 tint = default,
            bool useSegmentationColor = true,
            bool confidenceToAlpha = false,
            int outputTexWidth = 0,
            int outputTexHeight = 0)
        {
            segmentations ??= Enumerable.Empty<Segmentation>();

            if (tint == default)
                tint = new Color4(1f, 1f, 1f, 1f);

            int outW = outputTexWidth > 0 ? outputTexWidth : width;
            int outH = outputTexHeight > 0 ? outputTexHeight : height;

            var format = doRGB ? PixelFormat.R8G8B8A8_UNorm : PixelFormat.R8_UNorm;
            int bpp = doRGB ? 4 : 1;
            byte[] dst = new byte[outW * outH * bpp];

            foreach (var seg in segmentations)
            {
                if (seg?.BitPackedPixelMask is null || seg.BitPackedPixelMask.Length == 0)
                    continue;

                var bbox = seg.BoundingBox;
                int bw = bbox.Width;
                int bh = bbox.Height;
                if (bw <= 0 || bh <= 0)
                    continue;

                // Clip to output
                int left = Math.Max(0, bbox.Left);
                int top = Math.Max(0, bbox.Top);
                int right = Math.Min(outW, bbox.Right);
                int bottom = Math.Min(outH, bbox.Bottom);
                if (left >= right || top >= bottom)
                    continue;

                int startXInMask = Math.Max(0, -bbox.Left);
                int startYInMask = Math.Max(0, -bbox.Top);

                var chosen = useSegmentationColor ? seg.Color : tint;
                if (chosen == default)
                    chosen = new Color4(1f, 1f, 1f, 1f);

                float aBase = Math.Clamp(chosen.A, 0f, 1f);
                if (aBase <= 0f)
                    continue;

                float segConf = confidenceToAlpha ? (float)Math.Clamp(seg.Confidence, 0.0, 1.0) : 1f;
                float alphaFactor = aBase * segConf;
                if (alphaFactor <= 0f)
                    continue;

                byte aByte = (byte)(alphaFactor * 255f);

                byte rPremul = 0, gPremul = 0, bPremul = 0;
                if (doRGB)
                {
                    float r = Math.Clamp(chosen.R, 0f, 1f);
                    float g = Math.Clamp(chosen.G, 0f, 1f);
                    float b = Math.Clamp(chosen.B, 0f, 1f);
                    rPremul = (byte)(r * aByte);
                    gPremul = (byte)(g * aByte);
                    bPremul = (byte)(b * aByte);
                }

                var mask = seg.BitPackedPixelMask;
                int maskStride = bw;

                for (int ty = top; ty < bottom; ty++)
                {
                    int yMask = startYInMask + (ty - top);
                    int baseDst = ty * outW + left;
                    int rowMaskStart = yMask * maskStride;

                    for (int tx = left; tx < right; tx++)
                    {
                        int xMask = startXInMask + (tx - left);
                        int idx = rowMaskStart + xMask;

                        int byteIndex = idx >> 3;
                        int bitShift = idx & 7;
                        if ((mask[byteIndex] & (1 << bitShift)) == 0)
                            continue;

                        int destIndex = (baseDst + (tx - left)) * bpp;

                        if (doRGB)
                        {
                            // Max alpha: overwrite only if higher alpha (binary uniform aByte so single compare)
                            if (aByte > dst[destIndex + 3])
                            {
                                dst[destIndex + 0] = rPremul;
                                dst[destIndex + 1] = gPremul;
                                dst[destIndex + 2] = bPremul;
                                dst[destIndex + 3] = aByte;
                            }
                        }
                        else
                        {
                            if (aByte > dst[destIndex])
                                dst[destIndex] = aByte;
                        }
                    }
                }
            }

            return Texture.New2D(
                device,
                outW,
                outH,
                format,
                dst,
                TextureFlags.ShaderResource,
                GraphicsResourceUsage.Immutable);
        }
    }
}
