// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using Stride.Graphics;
using Stride.Core.Mathematics;

namespace YoloDotNet.Modules.V8E
{
    internal class SegmentationModuleV8E : ISegmentationModule
    {
        private readonly YoloCore _yoloCore;
        private readonly SegmentationModuleV8 _segmentationModuleV8 = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public SegmentationModuleV8E(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // YOLOv8E uses the YOLOv8 model architecture
            _segmentationModuleV8 = new SegmentationModuleV8(_yoloCore);
        }

        public List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
            => _segmentationModuleV8.ProcessImage(image, confidence, pixelConfidence, iou);

        public List<Segmentation> ProcessImageData(byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou, int labelIndex, bool cropToBB, double scaleBB, Func<ObjectResult, bool>? bboxFilter)
            => _segmentationModuleV8.ProcessImageData(imageData, width, height, confidence, pixelConfidence, iou, labelIndex, cropToBB, scaleBB, bboxFilter);

        public (ObjectResult[], Texture) ProcessMaskAsTexture(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou, int labelIndex, bool CropToBB, Color4 tint, double scaleBB, bool doRGB, Func<ObjectResult, bool>? bboxFilter)
            => _segmentationModuleV8.ProcessMaskAsTexture(device, imageData, width, height, confidence, pixelConfidence, iou, labelIndex, CropToBB, tint, scaleBB, doRGB, bboxFilter);
        
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
