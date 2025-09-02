<<<<<<< HEAD
﻿// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V11
{
    internal class SegmentationModuleV11 : ISegmentationModule
    {
=======
﻿using Stride.Graphics;

namespace YoloDotNet.Modules.V11
{
    public class SegmentationModuleV11 : ISegmentationModule
    {
        public event EventHandler VideoStatusEvent = delegate { };
        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };

>>>>>>> SegmentationsToTexture2D
        private readonly YoloCore _yoloCore;
        private readonly SegmentationModuleV8 _segmentationModuleV8 = default!;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public SegmentationModuleV11(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

<<<<<<< HEAD
            // Yolov11 uses the YOLOv8 model architecture.
            _segmentationModuleV8 = new SegmentationModuleV8(_yoloCore);
        }

        public List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
            => _segmentationModuleV8.ProcessImage(image, confidence, pixelConfidence, iou);

        #region Helper methods

        public void Dispose()
        {
=======
            // Yolov11 has the same model input/output shapes as Yolov8  
            // Use Yolov8 module  
            _segmentationModuleV8 = new SegmentationModuleV8(_yoloCore);

            SubscribeToVideoEvents();
        }

        public List<Segmentation> ProcessImage(SKImage image, double confidence, double pixelConfidence, double iou)
            => _segmentationModuleV8.ProcessImage(image, confidence, pixelConfidence, iou);

        public List<Segmentation> ProcessImage(byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou)
            => _segmentationModuleV8.ProcessImage(imageData, width, height, confidence, pixelConfidence, iou);

        public Dictionary<int, List<Segmentation>> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou)
            => _yoloCore.RunVideo(options, confidence, pixelConfidence, iou, ProcessImage);

        public Texture ProcessPersonMaskAsTexture(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou)
            => _segmentationModuleV8.ProcessPersonMaskAsTexture(device, imageData, width, height, confidence, pixelConfidence, iou);

        public (List<SKRectI>, Texture) ProcessPersonMaskAsTextureAndBB(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou)
            => _segmentationModuleV8.ProcessPersonMaskAsTextureAndBB(device, imageData, width, height, confidence, pixelConfidence, iou);

        public (List<SKRectI>, Texture) ProcessPersonMaskAsTextureAndBBFull(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou)
    => _segmentationModuleV8.ProcessPersonMaskAsTextureAndBBFull(device, imageData, width, height, confidence, pixelConfidence, iou);


        #region Helper methods  

        private void SubscribeToVideoEvents()
        {
            _yoloCore.VideoProgressEvent += (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            _yoloCore.VideoCompleteEvent += (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            _yoloCore.VideoStatusEvent += (sender, e) => VideoStatusEvent?.Invoke(sender, e);
        }

        public void Dispose()
        {
            _yoloCore.VideoProgressEvent -= (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            _yoloCore.VideoCompleteEvent -= (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            _yoloCore.VideoStatusEvent -= (sender, e) => VideoStatusEvent?.Invoke(sender, e);

>>>>>>> SegmentationsToTexture2D
            _segmentationModuleV8?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion

    }
}
