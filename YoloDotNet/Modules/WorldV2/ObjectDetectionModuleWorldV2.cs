// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.WorldV2
{
    internal class ObjectDetectionModuleWorldV2 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;
        private readonly ObjectDetectionModuleV8 _objectDetectionModuleV8 = default!;

        public ObjectDetectionModuleWorldV2(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            // Yolo uses the YOLOv8 model architecture.
            _objectDetectionModuleV8 = new ObjectDetectionModuleV8(_yoloCore);
        }

        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
            => _objectDetectionModuleV8.ProcessImage(image, confidence, pixelConfidence, iou);

        public List<ObjectDetection> ProcessImageData(byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou)
            => _objectDetectionModuleV8.ProcessImageData(imageData, width, height, confidence, pixelConfidence, iou);

        public void Dispose()
        {
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}