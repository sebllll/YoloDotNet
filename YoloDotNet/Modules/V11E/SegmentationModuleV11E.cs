// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using Stride.Graphics;
using Stride.Core.Mathematics;
using System.Reflection.Metadata.Ecma335;

namespace YoloDotNet.Modules.V11E
{
    internal class SegmentationModuleV11E : ISegmentationModule
    {
        private readonly YoloCore _yoloCore;
        private readonly SegmentationModuleV8 _segmentationModuleV8 = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public SegmentationModuleV11E(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // YOLOv11E uses the YOLOv8 model architecture
            _segmentationModuleV8 = new SegmentationModuleV8(_yoloCore);
        }

        public List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
            => _segmentationModuleV8.ProcessImage(image, confidence, pixelConfidence, iou);

        public (List<SKRectI>, Texture) ProcessPersonMaskAsTexture(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou, int labelIndex, bool CropToBB, Color4 tint)
            //using IDisposableReadOnlyCollection<OrtValue>? ortValues = _yoloCore.Run(imageData, width, height);
            => _segmentationModuleV8.ProcessPersonMaskAsTexture(device, imageData, width, height, confidence, pixelConfidence, iou, labelIndex, CropToBB, tint);
        #region Helper methods

        public void Dispose()
        {
            _segmentationModuleV8?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion

    }
}
