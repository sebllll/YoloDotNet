using Stride.Graphics;

namespace YoloDotNet
{
    public class Yolo : IDisposable
    {
        #region Fields

        public event EventHandler VideoProgressEvent = delegate { };
        public event EventHandler VideoCompleteEvent = delegate { };
        public event EventHandler VideoStatusEvent = delegate { };

        private readonly IModule _detection;

        public OnnxModel OnnxModel => _detection.OnnxModel;

        /// <summary>
        /// Gets the execution provider currently used for inference (e.g., CPU, CUDA).
        /// </summary>
        public string ExecutionProvider => OnnxModel.CustomMetaData.TryGetValue("ExecutionProvider", out var provider) ? provider : "Unknown";

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the Yolo object detection model, which detects objects in an image or video based on a Yolo model in ONNX format.
        /// </summary>
        /// <param name="options">Options for initializing the YoloDotNet model.</param>
        public Yolo(YoloOptions options)
        {
            _detection = ModuleFactory.CreateModule(options);

            // Forward events from the detection module to the facade class
            _detection.VideoProgressEvent += (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            _detection.VideoCompleteEvent += (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            _detection.VideoStatusEvent += (sender, e) => VideoStatusEvent?.Invoke(sender, e);
        }

        #endregion

        #region Exposed methods for running inference on images
    
        /// <summary>
        /// Run image classification on an Image.
        /// </summary>
        /// <param name="img">The image to classify.</param>
        /// <param name="classes">The number of classes to return (default is 1).</param>
        /// <returns>A list of classification results.</returns>
        public List<Classification> RunClassification(SKImage img, int classes = 1)
            => ((IClassificationModule)_detection).ProcessImage(img, classes, 0, 0);

        /// <summary>
        /// Run object detection on an Image.
        /// </summary>
        /// <param name="img">The image to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of classification results.</returns>
        public List<ObjectDetection> RunObjectDetection(SKImage img, double confidence = 0.23, double iou = 0.7)
            => ((IObjectDetectionModule)_detection).ProcessImage(img, confidence, 0, iou);
        
        /// <summary>
        /// Run oriented bounding bBox detection on an image.
        /// </summary>
        /// <param name="img">The image to obb detect.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<OBBDetection> RunObbDetection(SKImage img, double confidence = 0.23, double iou = 0.7)
            => ((IOBBDetectionModule)_detection).ProcessImage(img, confidence, 0, iou);

        /// <summary>
        /// Run segmentation on an image.
        /// </summary>
        /// <param name="img">The image to segmentate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<Segmentation> RunSegmentation(SKImage img, double confidence = 0.23, double pixelConfedence = 0.65, double iou = 0.7)
            => ((ISegmentationModule)_detection).ProcessImage(img, confidence, pixelConfedence, iou);

        /// <summary>
        /// Run segmentation on an image.
        /// </summary>
        /// <param name="imageData">The image to segmentate.</param>
        /// <param name="width">The image width</param>
        /// <param name="height">The image height</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="pixelConfidence">The pixel confidence threshold for segmentation masks (default is 0.65).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<Segmentation> RunSegmentation(byte[] imageData, int width, int height, double confidence = 0.23, double pixelConfidence = 0.65, double iou = 0.7)
            => ((ISegmentationModule)_detection).ProcessImage(imageData, width, height, confidence, pixelConfidence, iou);

        /// <summary>
        /// Run segmentation on an image and return a texture mask for persons.
        /// </summary>
        /// <param name="device">The graphics device to create the texture with.</param>
        /// <param name="imageData">The image to segmentate.</param>
        /// <param name="width">The image width</param>
        /// <param name="height">The image height</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="pixelConfidence">The pixel confidence threshold for segmentation masks (default is 0.65).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A Stride Texture with the person mask.</returns>
        public Texture RunSegmentationAsTexture(GraphicsDevice device, byte[] imageData, int width, int height, double confidence = 0.23, double pixelConfidence = 0.65, double iou = 0.7)
            => ((ISegmentationModule)_detection).ProcessPersonMaskAsTexture(device, imageData, width, height, confidence, pixelConfidence, iou);

        /// <summary>
        /// Run segmentation on an image and return a texture mask for persons plus a list of BoundingBoxes.
        /// </summary>
        /// <param name="device">The graphics device to create the texture with.</param>
        /// <param name="imageData">The image to segmentate.</param>
        /// <param name="width">The image width</param>
        /// <param name="height">The image height</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="pixelConfidence">The pixel confidence threshold for segmentation masks (default is 0.65).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A Stride Texture with the person mask plus a list of BoundingBoxes.</returns>
        public (List<SKRectI>, Texture) RunSegmentationAndBB(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou)
            => ((ISegmentationModule)_detection).ProcessPersonMaskAsTextureAndBB(device, imageData, width, height, confidence, pixelConfidence, iou);

        /// <summary>
        /// Run segmentation on an image and return a texture mask for persons plus a list of BoundingBoxes.
        /// This version does not crop the texture to the bounding boxes, but returns the full texture.
        /// </summary>
        /// <param name="device">The graphics device to create the texture with.</param>
        /// <param name="imageData">The image to segmentate.</param>
        /// <param name="width">The image width</param>
        /// <param name="height">The image height</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="pixelConfidence">The pixel confidence threshold for segmentation masks (default is 0.65).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A Stride Texture with the person mask plus a list of BoundingBoxes.</returns>
        public (List<SKRectI>, Texture) RunSegmentationAndBBFull(GraphicsDevice device, byte[] imageData, int width, int height, double confidence, double pixelConfidence, double iou)
            => ((ISegmentationModule)_detection).ProcessPersonMaskAsTextureAndBBFull(device, imageData, width, height, confidence, pixelConfidence, iou);

        /// <summary>
        /// Run pose estimation on an image.
        /// </summary>
        /// <param name="img">The image to pose estimate.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        /// <returns>A list of Segmentation results.</returns>
        public List<PoseEstimation> RunPoseEstimation(SKImage img, double confidence = 0.23, double iou = 0.7)
            => ((IPoseEstimationModule)_detection).ProcessImage(img, confidence, 0, iou);

        #endregion

        #region Exposed methods for running inference on video

        /// <summary>
        /// Run image classification on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="classes">The number of classes to return for each frame (default is 1).</param>
        public Dictionary<int, List<Classification>> RunClassification(VideoOptions options, int classes = 1)
            => ((IClassificationModule)_detection).ProcessVideo(options, classes, 0, 0);

        /// <summary>
        /// Run object detection on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        public Dictionary<int, List<ObjectDetection>> RunObjectDetection(VideoOptions options, double confidence = 0.23, double iou = 0.7)
            => ((IObjectDetectionModule)_detection).ProcessVideo(options, confidence, 0, iou);  //((ObjectDetectionModule)_detection).ProcessVideo(options, confidence, iou);
        
        /// <summary>
        /// Run oriented bounding box detection on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        public Dictionary<int, List<OBBDetection>> RunObbDetection(VideoOptions options, double confidence = 0.23, double iou = 0.7)
            => ((IOBBDetectionModule)_detection).ProcessVideo(options, confidence, 0, iou);

        /// <summary>
        /// Run object detection on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        public Dictionary<int, List<Segmentation>> RunSegmentation(VideoOptions options, double confidence = 0.23, double pixelConfidence = 0.65, double iou = 0.7)
            => ((ISegmentationModule)_detection).ProcessVideo(options, confidence, pixelConfidence, iou);

        /// <summary>
        /// Run pose estimation on a video file.
        /// </summary>
        /// <param name="options">Options for video processing.</param>
        /// <param name="confidence">The confidence threshold for detected objects (default is 0.23).</param>
        /// <param name="iou">IoU (Intersection Over Union) overlap threshold value for removing overlapping bounding boxes (default: 0.7).</param>
        public Dictionary<int, List<PoseEstimation>> RunPoseEstimation(VideoOptions options, double confidence = 0.23, double iou = 0.7)
            => ((IPoseEstimationModule)_detection).ProcessVideo(options, confidence, 0, iou);

        public void Dispose()
        {
            VideoProgressEvent -= (sender, e) => VideoProgressEvent?.Invoke(sender, e);
            VideoCompleteEvent -= (sender, e) => VideoCompleteEvent?.Invoke(sender, e);
            VideoStatusEvent -= (sender, e) => VideoStatusEvent?.Invoke(sender, e);

            _detection.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}