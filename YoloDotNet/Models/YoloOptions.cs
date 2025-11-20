// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using SkiaSharp;

namespace YoloDotNet.Models
{
    /// <summary>
    /// Represents options for configuring a Yolo object.
    /// </summary>
    public class YoloOptions
    {
        /// <summary>
        /// Gets or sets the file path to the ONNX model.
        /// </summary>
        public string OnnxModel { get; set; } = default!;

        /// <summary>
        /// Gets or sets the ONNX model as a byte array.
        /// </summary>
        public byte[]? OnnxModelBytes { get; set; }

        /// <summary>
        /// Gets or sets Execution Provider (CPU, CUDA or TensorRT).
        /// </summary>
        public IExecutionProvider ExecutionProvider { get; set; } = new CpuExecutionProvider();

        /// <summary>
        /// Gets or sets the type of image resizing the onnx model requires.
        /// </summary>
        public ImageResize ImageResize { get; set; }

        /// <summary>
        /// SkiaSharp sampling options optimized for efficient downscaling.
        /// </summary>
        /// <remarks>
        /// - **Modifiability:** This property can be changed at runtime to adjust filtering behavior.
        /// </remarks>
        public SKFilterQuality FilterQuality { get; set; } = SKFilterQuality.Low;
    }
}
