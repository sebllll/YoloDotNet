// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using Stride.Graphics;
using Stride.Core.Mathematics;
using System.IO;
using System.Runtime.InteropServices;

namespace YoloDotNet.Modules.V8
{
    /// <summary>
    /// Handles YOLOv8-based segmentation tasks.
    /// </summary>
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

        /// <summary>
        /// Represents a fixed-size float buffer of 32 elements for mask weights.
        /// This structure is stack-allocated, avoiding heap allocations for performance.
        /// </summary>
        [InlineArray(32)]
        internal struct MaskWeights32
        {
            private float _mask;
        }

        public SegmentationModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;
            _objectDetectionModule = new ObjectDetectionModuleV8(_yoloCore);

            var inputWidth = _yoloCore.OnnxModel.Input.Width;
            var inputHeight = _yoloCore.OnnxModel.Input.Height;

            _maskWidth = _yoloCore.OnnxModel.Outputs[1].Width;
            _maskHeight = _yoloCore.OnnxModel.Outputs[1].Height;

            _elements = _yoloCore.OnnxModel.Labels.Length + 4;
            _channelsFromOutput0 = _yoloCore.OnnxModel.Outputs[0].Channels;
            _channelsFromOutput1 = _yoloCore.OnnxModel.Outputs[1].Channels;

            _scalingFactorW = (float)_maskWidth / inputWidth;
            _scalingFactorH = (float)_maskHeight / inputHeight;
        }

        /// <summary>
        /// Processes an image to produce a list of segmentation results.
        /// </summary>
        public List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
        {
            var (ortValues, imageSize) = _yoloCore.Run(image);
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

        public List<Segmentation> ProcessImageData(byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou, int labelIndex, bool cropToBB, double scaleBB, Func<ObjectResult, bool>? bboxFilter)
        {
            using var ortValues = _yoloCore.Run(imageData, width, height);
            var ortSpan0 = ortValues[0].GetTensorDataAsSpan<float>();
            var ortSpan1 = ortValues[1].GetTensorDataAsSpan<float>();

            var imageSize = new SKSizeI(width, height);
            var boundingBoxes = _objectDetectionModule.ObjectDetection(imageSize, ortSpan0, confidence, iou);

            if (labelIndex != -1)
            {
                boundingBoxes = [.. boundingBoxes.Where(box => box.Label.Index == labelIndex)];
            }

            if (bboxFilter is not null)
            {
                boundingBoxes = [.. boundingBoxes.Where(bboxFilter)];
            }

            foreach (var box in boundingBoxes)
            {
                // Compute the final unscaled bbox used for mask evaluation (optional scaling), then clamp to image bounds
                var unscaled = box.BoundingBoxUnscaled;
                if (cropToBB && scaleBB != 1.0)
                {
                    float cx = unscaled.MidX;
                    float cy = unscaled.MidY;
                    float nw = unscaled.Width * (float)scaleBB;
                    float nh = unscaled.Height * (float)scaleBB;
                    unscaled = new SKRect(cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2);
                }

                // Clamp to image extents (integer rect)
                int left = Math.Clamp((int)Math.Floor(unscaled.Left), 0, width - 1);
                int top = Math.Clamp((int)Math.Floor(unscaled.Top), 0, height - 1);
                int right = Math.Clamp((int)Math.Ceiling(unscaled.Right), 0, width - 1);
                int bottom = Math.Clamp((int)Math.Ceiling(unscaled.Bottom), 0, height - 1);

                // Guard against degenerate rects
                if (right < left) right = left;
                if (bottom < top) bottom = top;

                var finalBox = new SKRectI(left, top, right, bottom);

                // This rect is used to crop from the segmentation output
                var downScaledBoundingBox = DownscaleBoundingBoxToSegmentationOutput(new SKRect(left, top, right, bottom));

                // 1) Get weights from output0
                var maskWeights = GetMaskWeightsFromBoundingBoxArea(box, ortSpan0);

                // 2) Apply pixel mask to canvas limited to the (possibly scaled) bbox
                using var pixelMaskBitmap = new SKBitmap(_maskWidth, _maskHeight, SKColorType.Gray8, SKAlphaType.Opaque);
                ApplySegmentationPixelMask(pixelMaskBitmap, new SKRect(left, top, right, bottom), ortSpan1, maskWeights);

                // 3) Crop the (downscaled) bbox region from the canvas
                using var cropped = new SKBitmap();
                pixelMaskBitmap.ExtractSubset(cropped, downScaledBoundingBox);

                // 4) Upscale cropped pixel mask to the final bbox size (ensures packed mask aligns with finalBox)
                var pixelMaskInfo = new SKImageInfo(finalBox.Width, finalBox.Height, SKColorType.Gray8, SKAlphaType.Opaque);
                using var resizedCrop = new SKBitmap(pixelMaskInfo);
#if NET8_0_OR_GREATER
                if (Avx2.IsSupported)
                    Avx2LinearResizer.ScalePixels(cropped, resizedCrop);
                else
#endif
                    cropped.ScalePixels(resizedCrop, ImageConfig.SegmentationResamplingFilterQuality);

                // 5) Pack to compact bit array (threshold = pixelConfidence)
                box.BitPackedPixelMask = PackUpscaledMaskToBitArray(resizedCrop, pixelConfidence);

                // Ensure the returned Segmentation reflects the transformed bbox
                box.BoundingBox = finalBox;
            }

            ortValues[0]?.Dispose();
            ortValues[1]?.Dispose();

            return [.. boundingBoxes.Select(x => (Segmentation)x)];
        }

        /// <summary>
        /// Processes an image to generate a texture containing segmentation masks.
        /// </summary>
        public (ObjectResult[], Texture) ProcessMaskAsTexture(
            GraphicsDevice device,
            byte[] imageData,
            int width,
            int height,
            double confidence,
            double pixelConfidence,
            double iou, 
           int labelIndex,
           bool cropToBB,
           Color4 tint,
           double scaleBB,
           bool doRGB,
           Func<ObjectResult, bool>? bboxFilter)
        {
            if (tint == default)
            {
                tint = new Color4(1.0f, 1.0f, 1.0f, 1.0f);
            }

            using var ortValues = _yoloCore.Run(imageData, width, height);
            var ortSpan0 = ortValues[0].GetTensorDataAsSpan<float>();
            var ortSpan1 = ortValues[1].GetTensorDataAsSpan<float>();

            var boundingBoxes = _objectDetectionModule.ObjectDetection(new SKSizeI(width, height), ortSpan0, confidence, iou);

            if (labelIndex != -1)
            {
                boundingBoxes = [.. boundingBoxes.Where(box => box.Label.Index == labelIndex)];
            }

            if (bboxFilter is not null)
            {
                boundingBoxes = [.. boundingBoxes.Where(bboxFilter)];
            }

            var pixelFormat = doRGB ? PixelFormat.R8G8B8A8_UNorm : PixelFormat.R8_UNorm;
            var bytesPerPixel = doRGB ? 4 : 1;

            if (boundingBoxes.Length == 0)
            {
                var emptyData = new byte[width * height * bytesPerPixel];
                var emptyTexture = Texture.New2D(device, width, height, pixelFormat, emptyData, TextureFlags.ShaderResource, GraphicsResourceUsage.Immutable);
                return ([], emptyTexture);
            }

            var finalMaskData = new byte[width * height * bytesPerPixel];
            var colorType = doRGB ? SKColorType.Rgba8888 : SKColorType.Gray8;
            var alphaType = doRGB ? SKAlphaType.Premul : SKAlphaType.Opaque;

            using var segmentedBitmap = new SKBitmap(_maskWidth, _maskHeight, colorType, alphaType);
            //var skRectList = new List<SKRectI>();


            foreach (var box in boundingBoxes)
            {
                //skRectList.Add(box.BoundingBox);

                var maskWeights = CollectMaskWeightsFromBoundingBoxArea(box, ortSpan0);

                segmentedBitmap.Erase(SKColors.Transparent);

                SKRectI? cropRect = null;
                if (cropToBB)
                {
                    var unscaledBox = box.BoundingBoxUnscaled;
                    if (scaleBB != 1.0)
                    {
                        float centerX = unscaledBox.MidX;
                        float centerY = unscaledBox.MidY;
                        float newWidth = unscaledBox.Width * (float)scaleBB;
                        float newHeight = unscaledBox.Height * (float)scaleBB;
                        unscaledBox = new SKRect(centerX - newWidth / 2, centerY - newHeight / 2, centerX + newWidth / 2, centerY + newHeight / 2);
                    }
                    cropRect = DownscaleBoundingBoxToSegmentationOutput(unscaledBox);
                }

                ApplyMaskToSegmentedPixels(segmentedBitmap, ortSpan1, maskWeights, tint, doRGB, cropRect);
                TransferResizedMask(segmentedBitmap, finalMaskData, width, height, (float)pixelConfidence);

                // Generate and assign per-box bit-packed mask at original bounding box size
                var pixelMaskInfo = new SKImageInfo(box.BoundingBox.Width, box.BoundingBox.Height, SKColorType.Gray8, SKAlphaType.Opaque);
                var downScaledBoundingBox = DownscaleBoundingBoxToSegmentationOutput(box.BoundingBoxUnscaled);

                using var pixelMaskBitmap = new SKBitmap(_maskWidth, _maskHeight, SKColorType.Gray8, SKAlphaType.Opaque);
                var maskWeights32 = GetMaskWeightsFromBoundingBoxArea(box, ortSpan0);
                ApplySegmentationPixelMask(pixelMaskBitmap, box.BoundingBoxUnscaled, ortSpan1, maskWeights32);

                using var cropped = new SKBitmap();
                pixelMaskBitmap.ExtractSubset(cropped, downScaledBoundingBox);

                using var resizedCrop = new SKBitmap(pixelMaskInfo);
#if NET8_0_OR_GREATER
                if (Avx2.IsSupported)
                    Avx2LinearResizer.ScalePixels(cropped, resizedCrop);
                else
#endif
                    cropped.ScalePixels(resizedCrop, ImageConfig.SegmentationResamplingFilterQuality);

                box.BitPackedPixelMask = PackUpscaledMaskToBitArray(resizedCrop, pixelConfidence);
            }

            ortValues[0]?.Dispose();
            ortValues[1]?.Dispose();

            var outTexture = Texture.New2D(device, width, height, pixelFormat, finalMaskData, TextureFlags.ShaderResource, GraphicsResourceUsage.Immutable);
            return (boundingBoxes, outTexture);
        }

        /// <summary>
        /// Transfers and resizes a mask from a source bitmap to a destination byte array.
        /// </summary>
        private unsafe void TransferResizedMask(SKBitmap sourceMask, byte[] destinationMask, int destinationWidth, int destinationHeight, float pixelThreshold)
        {
            if (destinationWidth <= 0 || destinationHeight <= 0) return;

            float scaleX = (float)sourceMask.Width / destinationWidth;
            float scaleY = (float)sourceMask.Height / destinationHeight;

            int sourceWidth = sourceMask.Width;
            int sourceBytesPerPixel = sourceMask.BytesPerPixel;
            int destBytesPerPixel = destinationMask.Length / (destinationWidth * destinationHeight);
            byte* sourcePixelData = (byte*)sourceMask.GetPixels().ToPointer();

            for (int y = 0; y < destinationHeight; y++)
            {
                for (int x = 0; x < destinationWidth; x++)
                {
                    int sourceX = (int)(x * scaleX);
                    int sourceY = (int)(y * scaleY);
                    int sourceIndex = (sourceY * sourceWidth + sourceX) * sourceBytesPerPixel;

                    byte alphaValue = (sourceBytesPerPixel == 4) ? sourcePixelData[sourceIndex + 3] : sourcePixelData[sourceIndex];
                    if ((float)alphaValue / 255.0f > pixelThreshold)
                    {
                        int destIndex = (y * destinationWidth + x) * destBytesPerPixel;
                        if (destBytesPerPixel == 4) // RGBA
                        {
                            if (alphaValue > destinationMask[destIndex + 3])
                            {
                                destinationMask[destIndex + 0] = sourcePixelData[sourceIndex + 0]; // R
                                destinationMask[destIndex + 1] = sourcePixelData[sourceIndex + 1]; // G
                                destinationMask[destIndex + 2] = sourcePixelData[sourceIndex + 2]; // B
                                destinationMask[destIndex + 3] = alphaValue;                       // A
                            }
                        }
                        else // Grayscale
                        {
                            if (alphaValue > destinationMask[destIndex])
                            {
                                destinationMask[destIndex] = alphaValue;
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Collects mask weights from the model's output tensor for a specific bounding box.
        /// </summary>
        private float[] CollectMaskWeightsFromBoundingBoxArea(ObjectResult box, ReadOnlySpan<float> ortSpan)
        {
            var maskWeights = new float[_channelsFromOutput1];
            var maskOffset = box.BoundingBoxIndex + (_channelsFromOutput0 * _elements);

            for (var m = 0; m < _channelsFromOutput1; m++, maskOffset += _channelsFromOutput0)
                maskWeights[m] = ortSpan[maskOffset];

            return maskWeights;
        }

        /// <summary>
        /// Applies a colored and weighted mask to a bitmap, optionally cropped to a bounding box.
        /// </summary>
        private unsafe void ApplyMaskToSegmentedPixels(SKBitmap segmentedBitmap, ReadOnlySpan<float> ortSpan, float[] maskWeights, Color4 color, bool doRGB, SKRectI? cropRect = null)
        {
            int width = segmentedBitmap.Width;
            int height = segmentedBitmap.Height;
            int startY = cropRect?.Top ?? 0;
            int endY = cropRect?.Bottom ?? height - 1;
            int startX = cropRect?.Left ?? 0;
            int endX = cropRect?.Right ?? width - 1;

            if (doRGB)
            {
                uint* pixelData = (uint*)segmentedBitmap.GetPixels().ToPointer();
                byte rByte = (byte)(color.R * 255);
                byte gByte = (byte)(color.G * 255);
                byte bByte = (byte)(color.B * 255);
                byte aByte = (byte)(color.A * 255);

                for (int y = startY; y <= endY; y++)
                {
                    for (int x = startX; x <= endX; x++)
                    {
                        float pixelWeight = 0;
                        var offset = x + y * width;
                        for (var p = 0; p < _channelsFromOutput1; p++, offset += width * height)
                            pixelWeight += ortSpan[offset] * maskWeights[p];

                        byte alpha = (byte)(YoloCore.Sigmoid(pixelWeight) * aByte);
                        uint r = (uint)(rByte * alpha / 255);
                        uint g = (uint)(gByte * alpha / 255);
                        uint b = (uint)(bByte * alpha / 255);

                        pixelData[y * width + x] = ((uint)alpha << 24) | (b << 16) | (g << 8) | r;
                    }
                }
            }
            else
            {
                byte* pixelData = (byte*)segmentedBitmap.GetPixels().ToPointer();
                byte aByte = (byte)(color.A * 255);

                for (int y = startY; y <= endY; y++)
                {
                    for (int x = startX; x <= endX; x++)
                    {
                        float pixelWeight = 0;
                        var offset = x + y * width;
                        for (var p = 0; p < _channelsFromOutput1; p++, offset += width * height)
                            pixelWeight += ortSpan[offset] * maskWeights[p];

                        pixelData[y * width + x] = (byte)(YoloCore.Sigmoid(pixelWeight) * aByte);
                    }
                }
            }
        }

        /// <summary>
        /// Retrieves mask weights from the output tensor for a given bounding box.
        /// </summary>
        private MaskWeights32 GetMaskWeightsFromBoundingBoxArea(ObjectResult box, ReadOnlySpan<float> ortSpan0)
        {
            MaskWeights32 maskWeights = default;
            var maskOffset = box.BoundingBoxIndex + (_channelsFromOutput0 * _elements);

            for (var m = 0; m < _channelsFromOutput1; m++, maskOffset += _channelsFromOutput0)
                maskWeights[m] = ortSpan0[maskOffset];

            return maskWeights;
        }

        /// <summary>
        /// Downscales a bounding box to the segmentation mask's dimensions.
        /// </summary>
        private SKRectI DownscaleBoundingBoxToSegmentationOutput(SKRect box)
        {
            int left = (int)Math.Floor(box.Left * _scalingFactorW);
            int top = (int)Math.Floor(box.Top * _scalingFactorH);
            int right = (int)Math.Ceiling(box.Right * _scalingFactorW);
            int bottom = (int)Math.Ceiling(box.Bottom * _scalingFactorH);

            left = Math.Clamp(left, 0, _maskWidth - 1);
            top = Math.Clamp(top, 0, _maskHeight - 1);
            right = Math.Clamp(right, 0, _maskWidth - 1);
            bottom = Math.Clamp(bottom, 0, _maskHeight - 1);

            return new SKRectI(left, top, right, bottom);
        }

        /// <summary>
        /// Applies a grayscale pixel mask within the bounds of a given bounding box.
        /// </summary>
        private unsafe void ApplySegmentationPixelMask(SKBitmap bitmap, SKRect bbox, ReadOnlySpan<float> outputOrtSpan, MaskWeights32 maskWeights)
        {
            var scaledBoundingBox = DownscaleBoundingBoxToSegmentationOutput(bbox);
            byte* ptr = (byte*)bitmap.GetPixels().ToPointer();

            for (int y = scaledBoundingBox.Top; y <= scaledBoundingBox.Bottom; y++)
            {
                for (int x = scaledBoundingBox.Left; x <= scaledBoundingBox.Right; x++)
                {
                    float pixelWeight = 0;
                    int offset = x + y * _maskWidth;

                    for (int p = 0; p < _channelsFromOutput1; p++, offset += _maskWidth * _maskHeight)
                        pixelWeight += outputOrtSpan[offset] * maskWeights[p];

                    ptr[y * bitmap.RowBytes + x] = (byte)(YoloCore.Sigmoid(pixelWeight) * 255);
                }
            }
        }

        /// <summary>
        /// Packs a bitmap mask into a compact byte array for efficient storage.
        /// </summary>
        private unsafe byte[] PackUpscaledMaskToBitArray(SKBitmap resizedBitmap, double confidenceThreshold)
        {
            byte* resizedPixelData = (byte*)resizedBitmap.GetPixels().ToPointer();
            var totalPixels = resizedBitmap.Width * resizedBitmap.Height;
            var bytes = new byte[(totalPixels + 7) / 8];

            for (int i = 0; i < totalPixels; i++)
            {
                if (YoloCore.CalculatePixelConfidence(resizedPixelData[i]) > confidenceThreshold)
                {
                    bytes[i >> 3] |= (byte)(1 << (i & 7));
                }
            }
            return bytes;
        }

        public void Dispose()
        {
            _objectDetectionModule?.Dispose();
            _yoloCore?.Dispose();
            GC.SuppressFinalize(this);
        }

        //public Texture TextureFromSegmentations(
        //    GraphicsDevice device,
        //    int width,
        //    int height,
        //    IEnumerable<Segmentation> segmentations)
        //{
        //    if (segmentations is null)
        //        segmentations = Enumerable.Empty<Segmentation>();

        //    var pixelFormat = PixelFormat.R8_UNorm;
        //    var finalMaskData = new byte[width * height];

        //    foreach (var seg in segmentations)
        //    {
        //        // Must have a mask and a valid bbox
        //        if (seg?.BitPackedPixelMask is null || seg.BitPackedPixelMask.Length == 0)
        //            continue;

        //        var bbox = seg.BoundingBox; // Assumed SKRectI
        //        int left = Math.Clamp(bbox.Left, 0, width - 1);
        //        int top = Math.Clamp(bbox.Top, 0, height - 1);
        //        int boxWidth = Math.Max(0, Math.Min(bbox.Width, width - left));
        //        int boxHeight = Math.Max(0, Math.Min(bbox.Height, height - top));

        //        if (boxWidth == 0 || boxHeight == 0)
        //            continue;

        //        var mask = seg.BitPackedPixelMask;

        //        // Iterate local pixels in bbox, unpack bits, write to final buffer
        //        for (int y = 0; y < boxHeight; y++)
        //        {
        //            for (int x = 0; x < boxWidth; x++)
        //            {
        //                int i = y * bbox.Width + x; // indexing uses original bbox.Width for bit layout
        //                int byteIndex = i >> 3;
        //                int bitIndex = i & 7;

        //                if (byteIndex >= mask.Length)
        //                    break;

        //                if ((mask[byteIndex] & (1 << bitIndex)) == 0)
        //                    continue;

        //                int targetX = left + x;
        //                int targetY = top + y;
        //                int destIndex = targetY * width + targetX;

        //                // Max-over composition (grayscale)
        //                if (finalMaskData[destIndex] < 255)
        //                    finalMaskData[destIndex] = 255;
        //            }
        //        }
        //    }

        //    return Texture.New2D(
        //        device,
        //        width,
        //        height,
        //        pixelFormat,
        //        finalMaskData,
        //        TextureFlags.ShaderResource,
        //        GraphicsResourceUsage.Immutable);
        //}
    }
}