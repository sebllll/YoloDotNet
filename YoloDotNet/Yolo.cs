// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using Stride.Graphics;
using Stride.Core.Mathematics;

namespace YoloDotNet
{
    /// <summary>
    /// Initializes a new instance of the Yolo class, which provides access to YOLO model inference for various vision tasks.
    /// </summary>
    /// <param name="options">Configuration options for the YOLO model and execution providers.</param>
    public class Yolo(YoloOptions options) : IDisposable
    {
        private readonly IModule _module = ModuleFactory.CreateModule(options);
        private FFmpegService? _ffmpegService;

        /// <summary>
        /// Gets the metadata for the loaded ONNX model.
        /// </summary>
        public OnnxModel OnnxModel => _module.OnnxModel;

        /// <summary>
        /// An action that is invoked when a video frame has been processed.
        /// The callback receives the processed SKBitmap and the frame's timestamp.
        /// </summary>
        public Action<SKBitmap, long>? OnVideoFrameReceived { get; set; }

        /// <summary>
        /// An action that is invoked when video processing has completed.
        /// </summary>
        public Action? OnVideoEnd { get; set; }

        /// <summary>
        /// Runs image classification on a given image.
        /// </summary>
        /// <typeparam name="T">The type of the image (e.g., SKBitmap, SKImage).</typeparam>
        /// <param name="image">The image to classify.</param>
        /// <param name="classes">The number of top classes to return.</param>
        /// <returns>A list of classification results.</returns>
        public List<Classification> RunClassification<T>(T image, int classes = 1)
            => ((IClassificationModule)_module).ProcessImage(image, classes, 0, 0);

        /// <summary>
        /// Runs object detection on a given image.
        /// </summary>
        /// <typeparam name="T">The type of the image (e.g., SKBitmap, SKImage).</typeparam>
        /// <param name="image">The image to perform detection on.</param>
        /// <param name="confidence">The confidence threshold for filtering detected objects.</param>
        /// <param name="iou">The Intersection over Union (IoU) threshold for non-maximum suppression.</param>
        /// <returns>A list of detected objects.</returns>
        public List<ObjectDetection> RunObjectDetection<T>(T image, double confidence = 0.2, double iou = 0.7)
             => ((IObjectDetectionModule)_module).ProcessImage(image, confidence, 0, iou);

        /// <summary>
        /// Runs oriented bounding box (OBB) detection on a given image.
        /// </summary>
        /// <typeparam name="T">The type of the image (e.g., SKBitmap, SKImage).</typeparam>
        /// <param name="image">The image to perform OBB detection on.</param>
        /// <param name="confidence">The confidence threshold for filtering detected objects.</param>
        /// <param name="iou">The Intersection over Union (IoU) threshold for non-maximum suppression.</param>
        /// <returns>A list of detected objects with oriented bounding boxes.</returns>
        public List<OBBDetection> RunObbDetection<T>(T image, double confidence = 0.2, double iou = 0.7)
            => ((IOBBDetectionModule)_module).ProcessImage(image, confidence, 0, iou);

        /// <summary>
        /// Runs instance segmentation on a given image.
        /// </summary>
        /// <typeparam name="T">The type of the image (e.g., SKBitmap, SKImage).</typeparam>
        /// <param name="image">The image to segment.</param>
        /// <param name="confidence">The confidence threshold for filtering detected objects.</param>
        /// <param name="pixelConfidence">The confidence threshold for the pixels in the segmentation mask.</param>
        /// <param name="iou">The Intersection over Union (IoU) threshold for non-maximum suppression.</param>
        /// <returns>A list of segmentation results, including masks and bounding boxes.</returns>
        public List<Segmentation> RunSegmentation<T>(T image, double confidence = 0.2, double pixelConfidence = 0.65, double iou = 0.7)
            => ((ISegmentationModule)_module).ProcessImage(image, confidence, pixelConfidence, iou);

        /// <summary>
        /// Runs pose estimation on a given image.
        /// </summary>
        /// <typeparam name="T">The type of the image (e.g., SKBitmap, SKImage).</typeparam>
        /// <param name="image">The image to perform pose estimation on.</param>
        /// <param name="confidence">The confidence threshold for filtering detected objects.</param>
        /// <param name="iou">The Intersection over Union (IoU) threshold for non-maximum suppression.</param>
        /// <returns>A list of pose estimation results, including keypoints.</returns>
        public List<PoseEstimation> RunPoseEstimation<T>(T image, double confidence = 0.2, double iou = 0.7)
            => ((IPoseEstimationModule)_module).ProcessImage(image, confidence, 0, iou);

        /// <summary>
        /// Runs segmentation on image data and returns a texture mask.
        /// </summary>
        /// <param name="device">The graphics device to create the texture with.</param>
        /// <param name="imageData">The raw image data bytes.</param>
        /// <param name="width">The width of the image.</param>
        /// <param name="height">The height of the image.</param>
        /// <param name="confidence">The confidence threshold for detected objects.</param>
        /// <param name="pixelConfidence">The pixel confidence threshold for segmentation masks.</param>
        /// <param name="iou">The IoU threshold for removing overlapping bounding boxes.</param>
        /// <param name="labelIndex">The specific class label index to generate a mask for.</param>
        /// <param name="CropToBB">Whether to crop the mask to the bounding box of the detected object.</param>
        /// <param name="tint">The color to tint the segmentation mask.</param>
        /// <param name="scaleBB">A scaling factor for the bounding box when cropping.</param>
        /// <returns>A tuple containing a list of bounding boxes and a Stride Texture with the generated mask.</returns>
        public (List<SKRectI>, Texture) RunSegmentationAsTexture(GraphicsDevice device, byte[] imageData, int width, int height, double confidence = 0.23, double pixelConfidence = 0.65, double iou = 0.7, int labelIndex = 0, bool CropToBB = false, Color4 tint = default, double scaleBB = 1.0f)
            => ((ISegmentationModule)_module).ProcessMaskAsTexture(device, imageData, width, height, confidence, pixelConfidence, iou, labelIndex, CropToBB, tint, scaleBB);

        /// <summary>
        /// Initializes the video stream using the specified options.
        /// </summary>
        /// <param name="videoOptions">The configuration for video processing.</param>
        /// <exception cref="YoloDotNetVideoException">Thrown if the CPU execution provider is used for video, as GPU is required.</exception>
        public void InitializeVideo(VideoOptions videoOptions)
        {
            if (options.ExecutionProvider is CpuExecutionProvider)
                throw new YoloDotNetVideoException(
                    "Video inference requires GPU acceleration (CUDA or TensorRT) and FFmpeg. " +
                    "Please select a GPU-based execution provider and ensure FFmpeg is installed and accessible in your system's PATH.");

            _ffmpegService = new(videoOptions, options)
            {
                OnFrameReady = (frame, frameIndex) => OnVideoFrameReceived?.Invoke(frame, frameIndex),
                OnVideoEnd = () => OnVideoEnd?.Invoke()
            };
        }

        /// <summary>
        /// Retrieves a list of available video input devices on the system.
        /// </summary>
        /// <returns>A list of video device names.</returns>
        public static List<string> GetVideoDevices() => FFmpegService.GetVideoDevicesOnSystem();

        /// <summary>
        /// Retrieves metadata for the initialized video stream.
        /// </summary>
        /// <returns>The video metadata.</returns>
        /// <exception cref="YoloDotNetVideoException">Thrown if video has not been initialized.</exception>
        public VideoMetadata GetVideoMetaData()
            => _ffmpegService?.VideoMetadata ?? throw new YoloDotNetVideoException(
                "Video not initialized. Call InitializeVideo() before retrieving metadata.");

        /// <summary>
        /// Starts decoding and processing frames from the initialized video stream.
        /// </summary>
        public void StartVideoProcessing() => _ffmpegService?.Start();

        /// <summary>
        /// Stops video frame processing and releases associated resources.
        /// </summary>
        public void StopVideoProcessing() => _ffmpegService?.Stop();

        /// <summary>
        /// Gets a description of the currently loaded YOLO model.
        /// </summary>
        public string ModelInfo =>
            _module.OnnxModel == null
                ? "No model loaded"
                : $"{_module.OnnxModel.ModelType} (yolo {_module.OnnxModel.ModelVersion.ToString().ToLower()})";

        /// <summary>
        /// Releases all resources used by the Yolo instance.
        /// </summary>
        public void Dispose()
        {
            _module.Dispose();
            _ffmpegService?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}