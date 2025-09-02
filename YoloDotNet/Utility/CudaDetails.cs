using Microsoft.ML.OnnxRuntime;

namespace YoloDotNet.Utility
{
    /// <summary>
    /// Represents details about the available CUDA environment.
    /// </summary>
    /// <param name="IsAvailable">Indicates whether the CUDA execution provider is available.</param>
    /// <param name="CudaVersion">The version of the CUDA runtime.</param>
    /// <param name="CudnnVersion">The version of the cuDNN library.</param>
    public record CudaDetails(
        bool IsAvailable,
        string CudaVersion,
        string CudnnVersion);

    /// <summary>
    /// Provides functionality to retrieve information about the CUDA environment.
    /// </summary>
    public static class CudaInfo
    {
        /// <summary>
        /// Gets details about the CUDA environment recognized by the ONNX Runtime.
        /// </summary>
        /// <returns>A CudaDetails object with information about the CUDA installation.</returns>
        public static CudaDetails GetCudaDetails()
        {
            var isAvailable = false;
            var cudaVersion = "N/A";
            var cudnnVersion = "N/A";

            try
            {
                var availableProviders = OrtEnv.Instance().GetAvailableProviders();
                isAvailable = availableProviders.Contains("CUDA");

                if (isAvailable)
                {
                    // The C# API for ONNX Runtime does not directly expose methods
                    // to get the specific CUDA and cuDNN versions.
                    // However, these versions are typically printed to the debug
                    // output when an InferenceSession is created with the CUDA provider.
                    cudaVersion = "See Debug Output";
                    cudnnVersion = "See Debug Output";
                }
            }
            catch (OnnxRuntimeException ex)
            {
                // Handle cases where the ONNX Runtime library might not be fully initialized
                // or if there's an issue loading the CUDA provider dependencies.
                System.Diagnostics.Debug.WriteLine($"Error checking CUDA availability: {ex.Message}");
            }

            return new CudaDetails(isAvailable, cudaVersion, cudnnVersion);
        }
    }
}
