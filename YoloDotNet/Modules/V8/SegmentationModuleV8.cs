// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using Stride.Graphics;
using System.ComponentModel.Design;

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
        {
            var ortSpan0 = ortValues[0].GetTensorDataAsSpan<float>();
            var ortSpan1 = ortValues[1].GetTensorDataAsSpan<float>();

            var boundingBoxes = _objectDetectionModule.ObjectDetection(imageSize, ortSpan0, confidence, iou);

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
                    cropped.ScalePixels(resizedCrop, ImageConfig.SegmentationResamplingFilterQuality);

                // 5) Pack the upscaled pixel mask into a compact bit array (1 bit per pixel)
                // for cleaner, memory-efficient storage of the mask in the detection box.
                box.BitPackedPixelMask = PackUpscaledMaskToBitArray(resizedCrop, pixelConfidence);
            }

            // Clean up
            ortValues[0]?.Dispose();
            ortValues[1]?.Dispose();
            ortValues?.Dispose();

            return [.. boundingBoxes.Select(x => (Segmentation)x)];
        }

        public (List<SKRectI>, Texture) ProcessPersonMaskAsTexture(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou, int labelIndex, bool CropToBB)
        {
            using IDisposableReadOnlyCollection<OrtValue>? ortValues = _yoloCore.Run(imageData, width, height);
            return RunPersonSegmentationToTexture(device, width, height, ortValues, confidence, pixelConfidence, iou, labelIndex, CropToBB);
        }


        private (List<SKRectI>, Texture) RunPersonSegmentationToTexture(GraphicsDevice device, int imageWidth, int imageHeight, IDisposableReadOnlyCollection<OrtValue> ortValues, double confidence, double pixelConfidence, double iou, int labelIndex, bool CropToBB)
        {
            var ortSpan0 = ortValues[0].GetTensorDataAsSpan<float>();
            var ortSpan1 = ortValues[1].GetTensorDataAsSpan<float>();

            using var dummyImage = SKImage.Create(new SKImageInfo(imageWidth, imageHeight));
            var boundingBoxes = _objectDetectionModule.ObjectDetection(new SKSizeI(imageWidth, imageHeight), ortSpan0, confidence, iou);

            var personBoxes = boundingBoxes.Where(box => box.Label.Index == labelIndex);

            // If no bounding boxes are found, return distinct texture + empty list
            if (!personBoxes.Any())
            {
                var distinctData = new byte[imageWidth * imageHeight];
                // Fill distinctData with a unique value if needed
                var emptyTexture = Texture.New2D(device, imageWidth, imageHeight, PixelFormat.R8_UNorm,
                                                 distinctData, TextureFlags.ShaderResource,
                                                 GraphicsResourceUsage.Immutable, TextureOptions.None);

                return (new List<SKRectI>(), emptyTexture);
            }

            // Prepare final mask
            var finalMaskData = new byte[imageWidth * imageHeight];
            float pixelThreshold = (float)pixelConfidence;

            using var segmentedBitmap = new SKBitmap(_yoloCore.OnnxModel.Outputs[1].Width,
                                                     _yoloCore.OnnxModel.Outputs[1].Height,
                                                     SKColorType.Gray8, SKAlphaType.Opaque);

            int elements = _yoloCore.OnnxModel.Labels.Length + 4;

            var skRectList = new List<SKRectI>();

            foreach (var box in personBoxes)
            {
                skRectList.Add(box.BoundingBox); // Collect bounding box from original space

                var maskWeights = CollectMaskWeightsFromBoundingBoxArea(
                    box,
                    _yoloCore.OnnxModel.Outputs[0].Channels,
                    _yoloCore.OnnxModel.Outputs[1].Channels,
                    elements,
                    ortSpan0);

                using (var canvas = new SKCanvas(segmentedBitmap))
                {
                    canvas.Clear(SKColors.White);
                }

                if (CropToBB)
                {
                    ApplyMaskToSegmentedPixels(segmentedBitmap,
                                               _yoloCore.OnnxModel.Outputs[1].Width,
                                               _yoloCore.OnnxModel.Outputs[1].Height,
                                               _yoloCore.OnnxModel.Outputs[1].Channels,
                                               box.BoundingBox,
                                               ortSpan1,
                                               maskWeights);
                }
                else
                { 
                    ApplyMaskToSegmentedPixelsFull(segmentedBitmap,
                                               _yoloCore.OnnxModel.Outputs[1].Width,
                                               _yoloCore.OnnxModel.Outputs[1].Height,
                                               _yoloCore.OnnxModel.Outputs[1].Channels,
                                               ortSpan1,
                                               maskWeights);
                }

                TransferResizedMaskFull(segmentedBitmap, finalMaskData, imageWidth, imageHeight, pixelThreshold);
            }

            ortValues[0].Dispose();
            ortValues[1].Dispose();

            var outTexture = Texture.New2D(device, imageWidth, imageHeight, PixelFormat.R8_UNorm,
                                           finalMaskData, TextureFlags.ShaderResource,
                                           GraphicsResourceUsage.Immutable, TextureOptions.None);

            return (skRectList, outTexture);
        }




        private unsafe void TransferResizedMaskFull(SKBitmap sourceMask, byte[] destinationMask, int destinationWidth, int destinationHeight, float pixelThreshold)
        {
            if (destinationWidth <= 0 || destinationHeight <= 0) return;

            float scaleX = (float)sourceMask.Width / destinationWidth;
            float scaleY = (float)sourceMask.Height / destinationHeight;

            int sourceWidth = sourceMask.Width;

            IntPtr sourcePtr = sourceMask.GetPixels();
            byte* sourcePixelData = (byte*)sourcePtr.ToPointer();

            for (int y = 0; y < destinationHeight; y++)
            {
                for (int x = 0; x < destinationWidth; x++)
                {
                    int sourceX = (int)(x * scaleX);
                    int sourceY = (int)(y * scaleY);

                    byte pixelValue = sourcePixelData[sourceY * sourceWidth + sourceX];
                    float confidenceValue = YoloCore.CalculatePixelConfidence(pixelValue);

                    if (confidenceValue > pixelThreshold)
                    {
                        int destIndex = y * destinationWidth + x;
                        byte newMaskValue = (byte)(confidenceValue * 255);

                        if (newMaskValue > destinationMask[destIndex])
                        {
                            destinationMask[destIndex] = newMaskValue;
                        }
                    }
                }
            }
        }



        private unsafe void TransferResizedMask(SKBitmap sourceMask, ObjectResult box, SKRectI sourceMaskBounds, byte[] destinationMask, int destinationWidth, float pixelThreshold)
        {
            var destBox = box.BoundingBox;
            if (destBox.Width <= 0 || destBox.Height <= 0) return;

            float scaleX = (float)sourceMaskBounds.Width / destBox.Width;
            float scaleY = (float)sourceMaskBounds.Height / destBox.Height;

            int sourceWidth = sourceMask.Width;
            int destinationHeight = destinationMask.Length / destinationWidth;

            IntPtr sourcePtr = sourceMask.GetPixels();
            byte* sourcePixelData = (byte*)sourcePtr.ToPointer();

            for (int y = 0; y < destBox.Height; y++)
            {
                int destPixelY = destBox.Top + y;
                if (destPixelY < 0 || destPixelY >= destinationHeight) continue;

                for (int x = 0; x < destBox.Width; x++)
                {
                    int destPixelX = destBox.Left + x;
                    if (destPixelX < 0 || destPixelX >= destinationWidth) continue;

                    int sourceX = sourceMaskBounds.Left + (int)(x * scaleX);
                    int sourceY = sourceMaskBounds.Top + (int)(y * scaleY);

                    byte pixelValue = sourcePixelData[sourceY * sourceWidth + sourceX];
                    float confidenceValue = YoloCore.CalculatePixelConfidence(pixelValue);

                    if (confidenceValue > pixelThreshold)
                    {
                        int destIndex = destPixelY * destinationWidth + destPixelX;
                        byte newMaskValue = (byte)(confidenceValue * 255);

                        if (newMaskValue > destinationMask[destIndex])
                        {
                            destinationMask[destIndex] = newMaskValue;
                        }
                    }
                }
            }
        }


        private static float[] CollectMaskWeightsFromBoundingBoxArea(ObjectResult box, int channelsFromOutput0, int channelsFromOutput1, int elements, ReadOnlySpan<float> ortSpan1)
        {
            var maskWeights = new float[channelsFromOutput1];
            var maskOffset = box.BoundingBoxIndex + (channelsFromOutput0 * elements);

            for (var m = 0; m < channelsFromOutput1; m++, maskOffset += channelsFromOutput0)
                maskWeights[m] = ortSpan1[maskOffset];

            return maskWeights;
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
                    pixelData[y * output1Width + x] = pixelData[y * output1Width + x];
                }
            }
        }

        unsafe private void ApplyMaskToSegmentedPixels(SKBitmap segmentedBitmap, int output1Width, int output1Height, int output1Channels, SKRectI scaledBoundingBox, ReadOnlySpan<float> ortSpan1, float[] maskWeights)
        {
            IntPtr pixelsPtr = segmentedBitmap.GetPixels();

            for (int y = 0; y < output1Height; y++)
            {
                for (int x = 0; x < output1Width; x++)
                {
                    if (x < scaledBoundingBox.Left || x > scaledBoundingBox.Right || y < scaledBoundingBox.Top || y > scaledBoundingBox.Bottom)
                        continue;

                    float pixelWeight = 0;
                    var offset = x + y * output1Width;
                    for (var p = 0; p < output1Channels; p++, offset += output1Width * output1Height)
                        pixelWeight += ortSpan1[offset] * maskWeights[p];

                    byte* pixelData = (byte*)pixelsPtr.ToPointer();
                    pixelData[y * output1Width + x] = YoloCore.CalculatePixelLuminance(YoloCore.Sigmoid(pixelWeight));
                }
            }
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