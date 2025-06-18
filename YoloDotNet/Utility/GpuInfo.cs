using System.Management;

namespace YoloDotNet.Utility
{
    /// <summary>
    /// Represents the details of a GPU device.
    /// </summary>
    /// <param name="Id">The sequential ID of the GPU, corresponding to the GpuId in YoloOptions.</param>
    /// <param name="Name">The name of the GPU.</param>
    /// <param name="VideoProcessor">The video processor or chipset.</param>
    /// <param name="DriverVersion">The currently installed driver version.</param>
    /// <param name="AdapterRamBytes">The adapter RAM in bytes.</param>
    public record GpuDetails(
        int Id,
        string Name,
        string VideoProcessor,
        string DriverVersion,
        long AdapterRamBytes);

    /// <summary>
    /// Provides functionality to retrieve information about available GPUs.
    /// </summary>
    public static class GpuInfo
    {
        /// <summary>
        /// Gets a list of details for all detected GPUs.
        /// This method uses WMI and is intended for Windows environments.
        /// </summary>
        /// <returns>A list of GpuDetails for each detected GPU.</returns>
        public static List<GpuDetails> GetGpuDetails()
        {
            var gpuDetailsList = new List<GpuDetails>();
            try
            {
                using var searcher = new ManagementObjectSearcher("select * from Win32_VideoController");
                var gpuId = 0;
                foreach (var obj in searcher.Get())
                {
                    var name = obj["Name"]?.ToString() ?? "N/A";
                    var videoProcessor = obj["VideoProcessor"]?.ToString() ?? "N/A";
                    var driverVersion = obj["DriverVersion"]?.ToString() ?? "N/A";
                    long.TryParse(obj["AdapterRAM"]?.ToString(), out var adapterRam);

                    gpuDetailsList.Add(new GpuDetails(
                        gpuId,
                        name,
                        videoProcessor,
                        driverVersion,
                        adapterRam
                    ));
                    gpuId++;
                }
            }
            catch (ManagementException)
            {
                // System.Management is not supported on this platform (e.g., Linux, macOS).
                // Return an empty list as a fallback.
                return gpuDetailsList;
            }
            return gpuDetailsList;
        }
    }
}