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
