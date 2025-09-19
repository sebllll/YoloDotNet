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

        /// <summary>
        /// Bit-packed mask where each bit represents a pixel with confidence above a threshold (1 = present, 0 = absent).
        /// </summary>
        public byte[] BitPackedPixelMask { get; set; } = [];

        /// <summary>
        /// Unpacks the bit-packed pixel mask and yields each pixel's location and confidence.
        /// </summary>
        /// <returns>An enumerable of tuples, each containing the absolute X and Y coordinates and the detection confidence for a pixel in the mask.</returns>
        public IEnumerable<(int X, int Y, double Confidence)> UnpackMask()
        {
            if (BitPackedPixelMask == null || BitPackedPixelMask.Length == 0)
            {
                yield break;
            }

            var (width, height) = (_baseBoundingBox.Width, _baseBoundingBox.Height);
            if (width <= 0 || height <= 0)
            {
                yield break;
            }

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int bitIndex = y * width + x;
                    int byteIndex = bitIndex / 8;
                    int bitInByte = bitIndex % 8;

                    if (byteIndex < BitPackedPixelMask.Length && (BitPackedPixelMask[byteIndex] & (1 << bitInByte)) != 0)
                    {
                        yield return (
                            _baseBoundingBox.Left + x + _offset.X,
                            _baseBoundingBox.Top + y + _offset.Y,
                            this.Confidence);
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
