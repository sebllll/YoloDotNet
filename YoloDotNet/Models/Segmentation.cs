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

        // Backing fields for bounding boxes
        private SKRectI _boundingBox;
        private SKRectI _baseBoundingBox;

        /// <summary>
        /// Rectangle defining the region of interest (bounding box) of the detected object.
        /// </summary>
        public SKRectI BoundingBox
        {
            get => _boundingBox;
            init
            {
                _boundingBox = value;
                _baseBoundingBox = value;
            }
        }

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

        private Int2 _offset;

        /// <summary>
        /// Offset used when rendering segmentations on larger canvases.
        /// Setting this will also offset the BoundingBox accordingly.
        /// </summary>
        public Int2 Offset
        {
            get => _offset;
            set
            {
                _offset = value;
                _boundingBox = new SKRectI(
                    _baseBoundingBox.Left + value.X,
                    _baseBoundingBox.Top + value.Y,
                    _baseBoundingBox.Right + value.X,
                    _baseBoundingBox.Bottom + value.Y
                );
            }
        }
    }
}
