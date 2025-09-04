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

        /// <summary>
        /// Rectangle defining the region of interest (bounding box) of the detected object.
        /// </summary>
        public SKRectI BoundingBox { get; init; }

        /// <summary>
        /// Bit-packed mask where each bit represents a pixel with confidence above a threshold (1 = present, 0 = absent).
        /// Can be unpacked to an <see cref="SKBitmap"/> using the <c>UnpackToBitmap</c> extension method.
        /// </summary>
        public byte[] BitPackedPixelMask { get; set; } = [];

        /// <summary>
        /// Color per Segmentation when rendering masks.
        /// Can be set from outside.
        /// </summary>
        public Color4 Color { get; set; } = Color4.White;
    }
}
