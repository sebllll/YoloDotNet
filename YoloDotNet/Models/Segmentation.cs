// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using Stride.Core.Mathematics;

namespace YoloDotNet.Models
{
    public class Segmentation : TrackingInfo, IDetection
    {
        /// <summary>
        /// Label information associated with the detected object.
        /// </summary>
        public LabelModel Label { get; init; } = new();

        /// <summary>
        /// Confidence score of the detected object.
        /// </summary>
        public double Confidence { get; init; }

        // Store the original box once; render-time BoundingBox is computed = base + offset
        private SKRectI _baseBoundingBox;

        /// <summary>
        /// Rectangle defining the region of interest (bounding box) of the detected object.
        /// Computed from the base box plus current Offset to avoid mutating state on every Offset change.
        /// </summary>
        public SKRectI BoundingBox
        {
            get => new SKRectI(
                _baseBoundingBox.Left + _offset.X,
                _baseBoundingBox.Top + _offset.Y,
                _baseBoundingBox.Right + _offset.X,
                _baseBoundingBox.Bottom + _offset.Y);
            init => _baseBoundingBox = value;
        }

        // Bit-packed binary mask (1 = present, 0 = absent)
        public byte[] BitPackedPixelMask { get; set; } = [];

        /// <summary>
        /// Enumerates all set pixels (bit=1) in the mask.
        /// Confidence is uniform = Segmentation.Confidence (binary mask has no per-pixel grades).
        /// Optimized to skip zero bytes.
        /// </summary>
        /// <returns>An enumerable of tuples, each containing the absolute X and Y coordinates and the detection confidence for a pixel in the mask.</returns>
        public IEnumerable<(int X, int Y, double Confidence)> UnpackMask()
        {
            var mask = BitPackedPixelMask;
            if (mask is null || mask.Length == 0)
                yield break;

            int width = _baseBoundingBox.Width;
            int height = _baseBoundingBox.Height;
            if (width <= 0 || height <= 0)
                yield break;

            double conf = Confidence;

            // Row-major scan
            for (int y = 0; y < height; y++)
            {
                int rowBitStart = y * width;
                int bit = 0;

                // Process full bytes
                int fullBytes = width >> 3;        // width / 8
                int remainder = width & 7;         // width % 8
                int byteIndex = rowBitStart >> 3;

                // Handle aligned full bytes
                for (int b = 0; b < fullBytes; b++, byteIndex++)
                {
                    byte m = mask[byteIndex];
                    if (m == 0)
                    {
                        bit += 8;
                        continue;
                    }

                    // Expand only set bits
                    // Check each bit (LSB first as encoded)
                    if ((m & 0x01) != 0) yield return (_baseBoundingBox.Left + bit + _offset.X, _baseBoundingBox.Top + y + _offset.Y, conf);
                    if ((m & 0x02) != 0) yield return (_baseBoundingBox.Left + bit + 1 + _offset.X, _baseBoundingBox.Top + y + _offset.Y, conf);
                    if ((m & 0x04) != 0) yield return (_baseBoundingBox.Left + bit + 2 + _offset.X, _baseBoundingBox.Top + y + _offset.Y, conf);
                    if ((m & 0x08) != 0) yield return (_baseBoundingBox.Left + bit + 3 + _offset.X, _baseBoundingBox.Top + y + _offset.Y, conf);
                    if ((m & 0x10) != 0) yield return (_baseBoundingBox.Left + bit + 4 + _offset.X, _baseBoundingBox.Top + y + _offset.Y, conf);
                    if ((m & 0x20) != 0) yield return (_baseBoundingBox.Left + bit + 5 + _offset.X, _baseBoundingBox.Top + y + _offset.Y, conf);
                    if ((m & 0x40) != 0) yield return (_baseBoundingBox.Left + bit + 6 + _offset.X, _baseBoundingBox.Top + y + _offset.Y, conf);
                    if ((m & 0x80) != 0) yield return (_baseBoundingBox.Left + bit + 7 + _offset.X, _baseBoundingBox.Top + y + _offset.Y, conf);

                    bit += 8;
                }

                // Remainder bits (0..7)
                if (remainder > 0)
                {
                    byte m = mask[byteIndex];
                    for (int r = 0; r < remainder; r++)
                    {
                        if ((m & (1 << r)) != 0)
                        {
                            int xLocal = bit + r;
                            yield return (_baseBoundingBox.Left + xLocal + _offset.X,
                                          _baseBoundingBox.Top + y + _offset.Y,
                                          conf);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Color per Segmentation when rendering masks. Can be set from outside.
        /// </summary>
        public Color4 Color { get; set; } = Color4.White;

        private Int2 _offset;

        /// <summary>
        /// Offset used when rendering segmentations on larger canvases.
        /// Changing this value does not mutate the base bounding box; BoundingBox is computed from base + offset.
        /// </summary>
        public Int2 Offset
        {
            get => _offset;
            set => _offset = value;
        }
    }
}
