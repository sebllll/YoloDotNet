using SkiaSharp;
using Stride.Graphics;
using System.Collections.Generic;
using YoloDotNet.Models;

namespace YoloDotNet.Modules.Interfaces
{
    public interface ISegmentationModule : IModule
    {
        List<Segmentation> ProcessImage(SKImage image, double confidence, double pixelConfidence, double iou);
        List<Segmentation> ProcessImage(byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou);
        Dictionary<int, List<Segmentation>> ProcessVideo(VideoOptions options, double confidence, double pixelConfidence, double iou);
    }
}