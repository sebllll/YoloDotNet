// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using SkiaSharp;
using Stride.Graphics;
using System.Collections.Generic;
using YoloDotNet.Models;

namespace YoloDotNet.Modules.Interfaces
{
    public interface ISegmentationModule : IModule
    {
        // List<Segmentation> ProcessImage(SKImage image, double confidence, double pixelConfidence, double iou);
        // List<Segmentation> ProcessImage(byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou);

        Texture ProcessPersonMaskAsTexture(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou);
        (List<SKRectI>, Texture) ProcessPersonMaskAsTextureAndBB(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou);
        (List<SKRectI>, Texture) ProcessPersonMaskAsTextureAndBBFull(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou);
        Dictionary<int, List<Segmentation>> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou);

        List<Segmentation> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou);
    }
}