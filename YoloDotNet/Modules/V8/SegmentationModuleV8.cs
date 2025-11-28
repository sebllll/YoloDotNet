// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V8
{
    internal class SegmentationModuleV8 : ISegmentationModule
    {
        private readonly object _lock = new();
        private YoloCore _yoloCore = default!;
        private ObjectDetectionModuleV8 _objectDetectionModule = default!;
        private float _scalingFactorW;
        private float _scalingFactorH;
        private int _maskWidth;
        private int _maskHeight;
        private int _elements;
        private int _channelsFromOutput0;
        private int _channelsFromOutput1;

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
            Initialize(yoloCore);
        }

        private void Initialize(YoloCore yoloCore)
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
            lock (_lock)
            {
                var (ortValues, imageSize) = _yoloCore.Run(image);
                using (ortValues)
                {
                    return RunSegmentation(imageSize, ortValues, confidence, pixelConfidence, iou);
                }
            }
        }

        public List<Segmentation> ProcessImageData(byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou, int labelIndex, bool cropToBB, double scaleBB, Func<ObjectResult, bool>? bboxFilter, int maxBoundingBoxesToProcess = 250)
        {
            lock (_lock)
            {
                try
                {
                    if (imageData is null || imageData.Length == 0 || width <= 0 || height <= 0)
                    {
                        return [];
                    }

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

                    // Safeguard against an excessive number of bounding boxes to prevent memory overflow.
                    if (maxBoundingBoxesToProcess > 0 && boundingBoxes.Length > maxBoundingBoxesToProcess)
                    {
                        boundingBoxes = boundingBoxes[..maxBoundingBoxesToProcess];
                    }

                    foreach (var box in boundingBoxes)
                    {
                        try
                        {
                            // Guard against processing bounding boxes with zero or negative dimensions.
                            if (box.BoundingBox.Width <= 0 || box.BoundingBox.Height <= 0)
                            {
                                continue;
                            }

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

                            // Guard against processing bounding boxes with zero or negative dimensions.
                            if (finalBox.Width <= 0 || finalBox.Height <= 0)
                            {
                                continue;
                            }

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
                                cropped.ScalePixels(resizedCrop, ImageConfig.SegmentationFilterQuality);

                            // 5) Pack to compact bit array (threshold = pixelConfidence)
                            box.BitPackedPixelMask = PackUpscaledMaskToBitArray(resizedCrop, pixelConfidence);

                            // Ensure the returned Segmentation reflects the transformed bbox
                            box.BoundingBox = finalBox;
                        }
                        catch (Exception ex)
                        {
                            // Wrap the original exception with more context about the object that failed.
                            throw new YoloDotNetException($"Failed during mask processing for label '{box?.Label?.Name ?? "N/A"}' with confidence {box?.Confidence:P2}. See inner exception for details.", ex);
                        }
                    }

                    // ortValues is automatically disposed by the 'using var' statement above
                    return [.. boundingBoxes.Select(x => (Segmentation)x)];
                }
                catch (Exception ex)
                {
                    // Catch exceptions from the entire method, especially from Run() or ObjectDetection().
                    throw new YoloDotNetException($"Failed during {nameof(ProcessImageData)} for image size {width}x{height}. See inner exception for details.", ex);
                }
            }
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
                    cropped.ScalePixels(resizedCrop, ImageConfig.SegmentationFilterQuality);

                // 5) Pack the upscaled pixel mask into a compact bit array (1 bit per pixel)
                // for cleaner, memory-efficient storage of the mask in the detection box.
                box.BitPackedPixelMask = PackUpscaledMaskToBitArray(resizedCrop, pixelConfidence);
            }

            return [.. boundingBoxes.Select(x => (Segmentation)x)];
        }

        private MaskWeights32 GetMaskWeightsFromBoundingBoxArea(ObjectResult box, ReadOnlySpan<float> ortSpan0)
        {
            MaskWeights32 maskWeights = default;

            var maskOffset = box.BoundingBoxIndex + (_channelsFromOutput0 * _elements);

            // Calculate the required span length for safe access.
            var requiredSpanLength = maskOffset + (_channelsFromOutput1 - 1) * _channelsFromOutput0;
            if (requiredSpanLength >= ortSpan0.Length)
            {
                // The calculated offset is out of bounds, return default to prevent a crash.
                return default;
            }

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

        public void Reset()
        {
            lock (_lock)
            {
                var options = _yoloCore.YoloOptions;

                _objectDetectionModule?.Dispose();
                _yoloCore?.Dispose();

                var yoloCore = new YoloCore(options);
                yoloCore.InitializeYolo();

                Initialize(yoloCore);
            }
        }

        public void Dispose()
        {
            _objectDetectionModule?.Dispose();
            _yoloCore?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}