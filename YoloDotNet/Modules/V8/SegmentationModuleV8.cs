// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using Microsoft.ML.OnnxRuntime;
using SkiaSharp;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System;
using YoloDotNet.Core;
using YoloDotNet.Models;
using YoloDotNet.Modules.Interfaces;
using System.Runtime.InteropServices;
using Stride.Graphics;
using System.Linq;
using Buffer = System.Buffer;

namespace YoloDotNet.Modules.V8
{
    internal class SegmentationModuleV8 : ISegmentationModule
    {
        private readonly YoloCore _yoloCore;
        private readonly ObjectDetectionModuleV8 _objectDetectionModule;
        private readonly float _scalingFactorW;
        private readonly float _scalingFactorH;
        private readonly int _maskWidth;
        private readonly int _maskHeight;
        private readonly int _elements;
        private readonly int _channelsFromOutput0;
        private readonly int _channelsFromOutput1;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        // Represents a fixed-size float buffer of 32 elements for mask weights.
        // Uses the InlineArray attribute to avoid heap allocations entirely.
        // This structure is stack-allocated when used inside methods or structs,
        // making it ideal for high-performance scenarios where per-frame allocations must be avoided.
        [InlineArray(32)]
        internal struct MaskWeights32
        {
            private float _mask;
        }

        public SegmentationModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
            _objectDetectionModule = new ObjectDetectionModuleV8(_yoloCore);

            // Get model input width and height
            var inputWidth = _yoloCore.OnnxModel.Input.Width;
            var inputHeight = _yoloCore.OnnxModel.Input.Height;

            // Get model pixel mask widh and height
            _maskWidth = _yoloCore.OnnxModel.Outputs[1].Width;
            _maskHeight = _yoloCore.OnnxModel.Outputs[1].Height;

            _elements = _yoloCore.OnnxModel.Labels.Length + 4; // 4 = the boundingbox dimension (x, y, width, height)
            _channelsFromOutput0 = _yoloCore.OnnxModel.Outputs[0].Channels;
            _channelsFromOutput1 = _yoloCore.OnnxModel.Outputs[1].Channels;

            // Calculate scaling factor for downscaling boundingboxes to segmentation pixelmask proportions
            _scalingFactorW = (float)_maskWidth / inputWidth;
            _scalingFactorH = (float)_maskHeight / inputHeight;
        }

        public List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
        {
            var (ortValues, imageSize) = _yoloCore.Run(image);

            return RunSegmentation(imageSize, ortValues, confidence, pixelConfidence, iou);
        }

        private List<Segmentation> RunSegmentation(SKSizeI imageSize, IDisposableReadOnlyCollection<OrtValue> ortValues, double confidence, double pixelConfidence, double iou)
            SubscribeToVideoEvents();
        }

        public List<Segmentation> ProcessImage(SKImage image, double confidence, double pixelConfidence, double iou)
        {
            using IDisposableReadOnlyCollection<OrtValue>? ortValues = _yoloCore.Run(image);
            return RunSegmentation(image.Width, image.Height, ortValues, confidence, pixelConfidence, iou);
        }

        public List<Segmentation> ProcessImage(byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou)
        {
            using IDisposableReadOnlyCollection<OrtValue>? ortValues = _yoloCore.Run(imageData, width, height);
            return RunSegmentation(width, height, ortValues, confidence, pixelConfidence, iou);
        }

        public Texture ProcessPersonMaskAsTexture(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou)
        {
            using IDisposableReadOnlyCollection<OrtValue>? ortValues = _yoloCore.Run(imageData, width, height);
            return RunPersonSegmentationToTexture(device, width, height, ortValues, confidence, pixelConfidence, iou);
        }

        public (List<SKRectI>, Texture) ProcessPersonMaskAsTextureAndBB(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou)
        {
            using IDisposableReadOnlyCollection<OrtValue>? ortValues = _yoloCore.Run(imageData, width, height);
            return RunPersonSegmentationToTextureAndBBFull(device, width, height, ortValues, confidence, pixelConfidence, iou);
        }

        public (List<SKRectI>, Texture) ProcessPersonMaskAsTextureAndBBFull(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou)
        {
            using IDisposableReadOnlyCollection<OrtValue>? ortValues = _yoloCore.Run(imageData, width, height);
            return RunPersonSegmentationToTextureAndBBFull(device, width, height, ortValues, confidence, pixelConfidence, iou);
        }

        public Dictionary<int, List<Segmentation>> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou)
            => _yoloCore.RunVideo(options, confidence, pixelConfidence, iou, (image, conf, pixConf, iouVal) => ProcessImage(image, conf, pixConf, iouVal));

        #region Segmentation

        /// <summary>
        /// <para><strong>Current segmentation process overview:</strong></para>
        /// <list type="number">
        ///     <item>
        ///         <description>Perform regular object detection to obtain the bounding boxes.</description>
        ///     </item>
        ///     <item>
        ///         <description>Rescale each bounding box to fit the object within the ONNX model dimensions for segmentation (default: 160x160 for YOLO).</description>
        ///     </item>
        ///     <item>
        ///         <description>Calculate pixel mask weights for the rescaled bounding box.</description>
        ///     </item>
        ///     <item>
        ///         <description>Apply the mask weights to the pixels within the rescaled bounding box area.</description>
        ///     </item>
        ///     <item>
        ///         <description>Crop the bounding box with the applied mask weights.</description>
        ///     </item>
        ///     <item>
        ///         <description>Rescale the cropped bounding box back to its original size.</description>
        ///     </item>
        ///     <item>
        ///         <description>Iterate through all pixels and collect those with confidence values greater than the threshold.</description>
        ///     </item>
        /// </list>
        /// </summary>
        /// <param name="image">The input image for segmentation.</param>
        /// <param name="ortValues">A read-only collection of OrtValue objects used for segmentation.</param>
        /// <param name="confidence">The confidence threshold for object detection.</param>
        /// <param name="iou">The Intersection over Union (IoU) threshold for excluding bounding boxes.</param>
        /// <returns>A list of Segmentation objects corresponding to the input bounding boxes.</returns> 
        private List<Segmentation> RunSegmentation(int imageWidth, int imageHeight, IDisposableReadOnlyCollection<OrtValue> ortValues, double confidence, double pixelConfidence, double iou)
        {
            var ortSpan0 = ortValues[0].GetTensorDataAsSpan<float>();
            var ortSpan1 = ortValues[1].GetTensorDataAsSpan<float>();

            var boundingBoxes = _objectDetectionModule.ObjectDetection(imageSize, ortSpan0, confidence, iou);
            using var dummyImage = SKImage.Create(new SKImageInfo(imageWidth, imageHeight));
            var boundingBoxes = _objectDetectionModule.ObjectDetection(dummyImage, ortSpan0, confidence, iou);
            var pixels = new ConcurrentBag<Pixel>();
            var croppedImage = new SKBitmap();
            var resizedBitmap = new SKBitmap();
            using var segmentedBitmap = new SKBitmap(_yoloCore.OnnxModel.Outputs[1].Width, _yoloCore.OnnxModel.Outputs[1].Height, SKColorType.Gray8, SKAlphaType.Opaque);
            using var paint = new SKPaint { FilterQuality = SKFilterQuality.Low, IsAntialias = false };

            var elements = _yoloCore.OnnxModel.Labels.Length + 4; // 4 = the boundingbox dimension (x, y, width, height)

            var inputWidth = _yoloCore.OnnxModel.Input.Width;
            var inputHeight = _yoloCore.OnnxModel.Input.Height;
            var scalingFactorW = (float)_yoloCore.OnnxModel.Outputs[1].Width / inputWidth;
            var scalingFactorH = (float)_yoloCore.OnnxModel.Outputs[1].Height / inputHeight;

            var pixelThreshold = (float)pixelConfidence; // ImageConfig.SEGMENTATION_PIXEL_THRESHOLD;

            foreach (var box in boundingBoxes)
            {
                var pixelMaskInfo = new SKImageInfo(box.BoundingBox.Width, box.BoundingBox.Height, SKColorType.Gray8, SKAlphaType.Opaque);
                var downScaledBoundingBox = DownscaleBoundingBoxToSegmentationOutput(box.BoundingBoxUnscaled);

                // 1) Get weights
                var maskWeights = GetMaskWeightsFromBoundingBoxArea(box, ortSpan0);

                // 2) Apply pixelmask based on mask-weights to canvas
                using var pixelMaskBitmap = new SKBitmap(_maskWidth, _maskHeight, SKColorType.Gray8, SKAlphaType.Opaque);
                ApplySegmentationPixelMask(pixelMaskBitmap, box.BoundingBoxUnscaled, ortSpan1, maskWeights);

                // 3) Crop downscaled boundingbox from the pixelmask canvas
                using var cropped = new SKBitmap();
                pixelMaskBitmap.ExtractSubset(cropped, downScaledBoundingBox);

                // 4) Upscale cropped pixelmask to original boundingbox size. For smother edges, use an appropriate resampling method!
                using var resizedCrop = new SKBitmap(pixelMaskInfo);

                // Use AVX2-optimized upscaling if supported; otherwise, fall back to SkiaSharp's ScalePixels.
                if (Avx2.IsSupported)
                    Avx2LinearResizer.ScalePixels(cropped, resizedCrop);
                else
                    cropped.ScalePixels(resizedCrop, ImageConfig.SegmentationResamplingOptions);

                // 5) Pack the upscaled pixel mask into a compact bit array (1 bit per pixel)
                // for cleaner, memory-efficient storage of the mask in the detection box.
                box.BitPackedPixelMask = PackUpscaledMaskToBitArray(resizedCrop, pixelConfidence);
            }

            // Clean up
            ortValues[0]?.Dispose();
            ortValues[1]?.Dispose();
            ortValues?.Dispose();

            var outTexture = Texture.New2D(device, imageWidth, imageHeight, PixelFormat.R8_UNorm,
                                           finalMaskData, TextureFlags.ShaderResource,
                                           GraphicsResourceUsage.Immutable, TextureOptions.None);

            return (skRectList, outTexture);
        }

        private MaskWeights32 GetMaskWeightsFromBoundingBoxArea(ObjectResult box, ReadOnlySpan<float> ortSpan0)
        {
            MaskWeights32 maskWeights = default;

            var maskOffset = box.BoundingBoxIndex + (_channelsFromOutput0 * _elements);

            for (var m = 0; m < _channelsFromOutput1; m++, maskOffset += _channelsFromOutput0)
                maskWeights[m] = ortSpan0[maskOffset];

            return maskWeights;
        }

        private SKRectI DownscaleBoundingBoxToSegmentationOutput(SKRect box)
        {
            int left = (int)Math.Floor(box.Left * _scalingFactorW);
            int top = (int)Math.Floor(box.Top * _scalingFactorH);
            int right = (int)Math.Ceiling(box.Right * _scalingFactorW);
            int bottom = (int)Math.Ceiling(box.Bottom * _scalingFactorH);

            // Clamp to mask bounds (important!)
            left = Math.Clamp(left, 0, _maskWidth - 1);
            top = Math.Clamp(top, 0, _maskHeight - 1);
            right = Math.Clamp(right, 0, _maskWidth - 1);
            bottom = Math.Clamp(bottom, 0, _maskHeight - 1);

            return new SKRectI(left, top, right, bottom);
        }

        unsafe void ApplySegmentationPixelMask(SKBitmap bitmap, SKRect bbox, ReadOnlySpan<float> outputOrtSpan, MaskWeights32 maskWeights)
        {
            var scaledBoundingBox = DownscaleBoundingBoxToSegmentationOutput(bbox);

            int startX = Math.Max(0, (int)scaledBoundingBox.Left);
            int endX = Math.Min(_maskWidth - 1, (int)scaledBoundingBox.Right);
            int startY = Math.Max(0, (int)scaledBoundingBox.Top);
            int endY = Math.Min(_maskHeight - 1, (int)scaledBoundingBox.Bottom);

            //var thresholdF = (float)threshold;
            int stride = bitmap.RowBytes;
            byte* ptr = (byte*)bitmap.GetPixels().ToPointer();

            for (int y = startY; y <= endY; y++)
            {
                byte* row = ptr + y * stride;

                for (int x = startX; x <= endX; x++)
                {
                    float pixelWeight = 0;
                    int offset = x + y * _maskWidth;

                    for (int p = 0; p < 32; p++, offset += _maskWidth * _maskHeight)
                        pixelWeight += outputOrtSpan[offset] * maskWeights[p];

                    pixelWeight = YoloCore.Sigmoid(pixelWeight);
                    row[x] = (byte)(pixelWeight * 255); // write directly to Gray8 bitmap
                }
            }
                        byte* pixelData = (byte*)pixelsPtr.ToPointer();
                        pixelData[y * output1Width + x] = YoloCore.CalculatePixelLuminance(YoloCore.Sigmoid(pixelWeight));
                }
            }
        }

        unsafe private void ApplyMaskToSegmentedPixelsFull(SKBitmap segmentedBitmap, int output1Width, int output1Height, int output1Channels, ReadOnlySpan<float> ortSpan1, float[] maskWeights)
        {
            IntPtr pixelsPtr = segmentedBitmap.GetPixels();
            byte* pixelData = (byte*)pixelsPtr.ToPointer();

            for (int y = 0; y < output1Height; y++)
            {
                for (int x = 0; x < output1Width; x++)
                {
                    float pixelWeight = 0;
                    var offset = x + y * output1Width;
                    for (var p = 0; p < output1Channels; p++, offset += output1Width * output1Height)
                        pixelWeight += ortSpan1[offset] * maskWeights[p];

                    pixelData[y * output1Width + x] = YoloCore.CalculatePixelLuminance(YoloCore.Sigmoid(pixelWeight));
                }
            }
        }

        unsafe private void CropSegmentedBoundingBox(SKBitmap croppedImaged, SKBitmap segmentedBitmap, SKRectI scaledBoundingBox, int output1Width)
        {
            IntPtr croppedPixelsPtr = croppedImaged.GetPixels();
            IntPtr pixelsPtr = segmentedBitmap.GetPixels();

            Parallel.For(0, scaledBoundingBox.Height, _yoloCore.parallelOptions, y =>
            {
                int srcIndex = (scaledBoundingBox.Top + y) * output1Width + scaledBoundingBox.Left;
                int dstIndex = y * scaledBoundingBox.Width;

                byte* srcPixelData = (byte*)pixelsPtr.ToPointer();
                byte* dstPixelData = (byte*)croppedPixelsPtr.ToPointer();
                Buffer.MemoryCopy(srcPixelData + srcIndex, dstPixelData + dstIndex, scaledBoundingBox.Width, scaledBoundingBox.Width);
            });
        }

        unsafe private byte[] PackUpscaledMaskToBitArray(SKBitmap resizedBitmap, double confidenceThreshold)
        {
            IntPtr resizedPtr = resizedBitmap.GetPixels();
            byte* resizedPixelData = (byte*)resizedPtr.ToPointer();

            var totalPixels = resizedBitmap.Width * resizedBitmap.Height;
            var bytes = new byte[CalculateBitMaskSize(totalPixels)];

            // Use bit-packing to efficiently store 8 pixels per byte (1 bit per pixel), 
            // significantly reducing memory usage compared to storing each pixel individually.
            for (int i = 0; i < totalPixels; i++)
            {
                var pixel = resizedPixelData[i];

                var confidence = YoloCore.CalculatePixelConfidence(pixel);

                if (confidence > confidenceThreshold)
                {
                    // Map this pixel's index to its bit in the byte array:
                    // - byteIndex: the byte containing this pixel's bit (1 byte = 8 pixels)
                    // - bitIndex: the bit position within that byte (0-7)
                    int byteIndex = i >> 3;     // Same as i / 8 (fast using bit shift)
                    int bitIndex = i & 0b0111;  // Same as i % 8 (fast using bit mask)

                    // Set the bit to 1 to indicate the pixel is present (confidence > threshold)
                    // Bits remain 0 by default to indicate absence (confidence <= threshold)
                    bytes[byteIndex] |= (byte)(1 << bitIndex);
                }
            }

            return bytes;
        }

        private static int CalculateBitMaskSize(int totalPixels) => (totalPixels + 7) / 8;

        public void Dispose()
        {
            _objectDetectionModule?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}