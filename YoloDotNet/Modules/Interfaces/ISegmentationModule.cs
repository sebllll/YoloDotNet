// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using Stride.Graphics;
using Stride.Core.Mathematics;
using System.Collections.Generic;
using YoloDotNet.Models;

namespace YoloDotNet.Modules.Interfaces
{
    public interface ISegmentationModule : IModule
    {
        (ObjectResult[], Texture) ProcessMaskAsTexture(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou, int labelIndex, bool cropToBB, Color4 tint, double scaleBB, bool doRGB, Func<ObjectResult, bool>? bboxFilter);

        List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou);
        List<Segmentation> ProcessImageData(byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou, int labelIndex, bool cropToBB, double scaleBB, Func<ObjectResult, bool>? bboxFilter);
    }
}