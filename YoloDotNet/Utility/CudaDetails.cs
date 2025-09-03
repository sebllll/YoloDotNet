using Microsoft.ML.OnnxRuntime;
using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

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
        // Keep a reference so the env isn't GC'd/disposed prematurely.
        private static readonly OrtEnv? _env;

        static CudaInfo()
        {
            try
            {
                // Ensure ORT environment is initialized.
                _env = OrtEnv.Instance();
            }
            catch (OnnxRuntimeException)
            {
                // Env already created elsewhere; safe to ignore.
            }
        }

        /// <summary>
        /// Gets details about the CUDA environment recognized by the ONNX Runtime.
        /// </summary>
        /// <param name="modelPath">Optional path to an ONNX model file to use for session initialization.</param>
        /// <returns>A CudaDetails object with information about the CUDA installation.</returns>
        public static CudaDetails GetCudaDetails(string modelPath = null)
        {
            var isAvailable = false;

            try
            {
                using var sessionOptions = new SessionOptions { LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO };
                sessionOptions.AppendExecutionProvider_CUDA();

                using var session = !string.IsNullOrEmpty(modelPath) && File.Exists(modelPath)
                    ? new InferenceSession(modelPath, sessionOptions)
                    : new InferenceSession(new byte[]
                    {
                        // Minimal valid ONNX model to trigger provider initialization.
                        0x08, 0x09, 0x12, 0x08, 0x6f, 0x6e, 0x6e, 0x78, 0x1a, 0x02, 0x10, 0x11, 0x22, 0x43, 0x0a, 0x41,
                        0x0a, 0x01, 0x78, 0x12, 0x0c, 0x0a, 0x0a, 0x0a, 0x01, 0x31, 0x12, 0x05, 0x0a, 0x03, 0x0a, 0x01,
                        0x31, 0x1a, 0x01, 0x79, 0x22, 0x0d, 0x0a, 0x01, 0x79, 0x12, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74,
                        0x69, 0x74, 0x79, 0x1a, 0x01, 0x78, 0x1a, 0x01, 0x79, 0x22, 0x0c, 0x0a, 0x0a, 0x0a, 0x01, 0x31,
                        0x12, 0x05, 0x0a, 0x03, 0x0a, 0x01, 0x31, 0x32, 0x00
                    }, sessionOptions);

                isAvailable = true;
            }
            catch (OnnxRuntimeException ex)
            {
                System.Diagnostics.Debug.WriteLine($"CUDA provider not available: {ex.Message}");
                isAvailable = false;
            }
            catch (DllNotFoundException ex)
            {
                System.Diagnostics.Debug.WriteLine($"ONNX Runtime or CUDA DLL not found: {ex.Message}");
                isAvailable = false;
            }

            // Query versions from native CUDA/cuDNN (independent from ORT logs)
            var cudaVersion = TryGetCudaRuntimeVersion() ?? TryGetCudaVersionFromDriver() ?? "N/A";
            var cudnnVersion = TryGetCuDnnVersion() ?? TryGetCuDnnVersionFromDllMetadata() ?? "N/A";

            return new CudaDetails(isAvailable, cudaVersion, cudnnVersion);
        }

        // ---- CUDA runtime version via cudaRuntimeGetVersion ----

        private static string? TryGetCudaRuntimeVersion()
        {
            if (TryLoadCudaRuntimeLibrary(out var handle, out _))
            {
                try
                {
                    if (NativeLibrary.TryGetExport(handle, "cudaRuntimeGetVersion", out var fn))
                    {
                        var del = Marshal.GetDelegateForFunctionPointer<cudaRuntimeGetVersionDelegate>(fn);
                        var rc = del(out var ver);
                        if (rc == 0 && ver > 0)
                        {
                            return FormatCudaVersion(ver);
                        }
                    }
                }
                catch { /* ignore */ }
                finally
                {
                    if (handle != IntPtr.Zero) NativeLibrary.Free(handle);
                }
            }

            return null;
        }

        private static bool TryLoadCudaRuntimeLibrary(out IntPtr handle, out string? resolvedPath)
        {
            handle = IntPtr.Zero;
            resolvedPath = null;

            string[] patterns;
            if (OperatingSystem.IsWindows())
            {
                patterns = new[] { "cudart64*.dll" };
            }
            else if (OperatingSystem.IsLinux())
            {
                patterns = new[] { "libcudart.so*", "libcudart.so" };
            }
            else
            {
                return false;
            }

            foreach (var file in EnumerateCandidateLibraries(patterns))
            {
                if (NativeLibrary.TryLoad(file, out handle))
                {
                    resolvedPath = file;
                    return true;
                }
            }

            // Try default names (rely on OS loader PATH)
            string[] defaultNames = OperatingSystem.IsWindows()
                ? new[] { "cudart64_12.dll", "cudart64_110.dll", "cudart64.dll" }
                : new[] { "libcudart.so.12", "libcudart.so.11", "libcudart.so" };

            foreach (var name in defaultNames)
            {
                if (NativeLibrary.TryLoad(name, out handle))
                {
                    resolvedPath = name;
                    return true;
                }
            }

            return false;
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int cudaRuntimeGetVersionDelegate(out int version);

        private static string FormatCudaVersion(int v)
        {
            // CUDA encodes as (1000*major + 10*minor [+ patch])
            var major = v / 1000;
            var minor = (v % 1000) / 10;
            var patch = v % 10;
            return patch == 0 ? $"{major}.{minor}" : $"{major}.{minor}.{patch}";
        }

        // ---- CUDA driver version via cuDriverGetVersion (fallback) ----

        private static string? TryGetCudaVersionFromDriver()
        {
            try
            {
                if (!OperatingSystem.IsWindows())
                    return null;

                var rc = cuDriverGetVersion(out var v);
                if (rc == 0 && v > 0)
                {
                    return FormatCudaVersion(v) + " (driver)";
                }
            }
            catch { /* ignore */ }

            return null;
        }

        [DllImport("nvcuda", CallingConvention = CallingConvention.Cdecl)]
        private static extern int cuDriverGetVersion(out int driverVersion);

        // ---- cuDNN version via cudnnGetVersion ----

        private static string? TryGetCuDnnVersion()
        {
            if (!TryLoadCuDnnLibrary(out var handle, out _))
            {
                return null;
            }

            try
            {
                if (NativeLibrary.TryGetExport(handle, "cudnnGetVersion", out var fn))
                {
                    var del = Marshal.GetDelegateForFunctionPointer<cudnnGetVersionDelegate>(fn);
                    var v = del();
                    if (v > 0)
                    {
                        return FormatCuDnnVersion(v);
                    }
                }
            }
            catch { /* ignore */ }
            finally
            {
                if (handle != IntPtr.Zero) NativeLibrary.Free(handle);
            }

            return null;
        }

        private static bool TryLoadCuDnnLibrary(out IntPtr handle, out string? resolvedPath)
        {
            handle = IntPtr.Zero;
            resolvedPath = null;

            string[] patterns;
            if (OperatingSystem.IsWindows())
            {
                // cuDNN 8: cudnn64_8.dll; cuDNN 9: cudnn64_9*.dll
                patterns = new[] { "cudnn64_*.dll", "cudnn*.dll" };
            }
            else if (OperatingSystem.IsLinux())
            {
                // cuDNN 8: libcudnn.so.8* ; cuDNN 9: libcudnn.so.9*
                patterns = new[] { "libcudnn.so*", "libcudnn*.so*" };
            }
            else
            {
                return false;
            }

            foreach (var file in EnumerateCandidateLibraries(patterns))
            {
                if (NativeLibrary.TryLoad(file, out handle))
                {
                    resolvedPath = file;
                    return true;
                }
            }

            // Try common names
            string[] defaultNames = OperatingSystem.IsWindows()
                ? new[] { "cudnn64_9.dll", "cudnn64_8.dll", "cudnn64.dll" }
                : new[] { "libcudnn.so.9", "libcudnn.so.8", "libcudnn.so" };

            foreach (var name in defaultNames)
            {
                if (NativeLibrary.TryLoad(name, out handle))
                {
                    resolvedPath = name;
                    return true;
                }
            }

            return false;
        }

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate ulong cudnnGetVersionDelegate();

        private static string FormatCuDnnVersion(ulong v)
        {
            var major = (int)(v / 1000);
            var minor = (int)((v % 1000) / 10);
            var patch = (int)(v % 10);
            return patch == 0 ? $"{major}.{minor}" : $"{major}.{minor}.{patch}";
        }

        // ---- Fallback: read DLL file version if API calls are not possible ----

        private static string? TryGetCuDnnVersionFromDllMetadata()
        {
            string[] patterns = OperatingSystem.IsWindows()
                ? new[] { "cudnn64_*.dll", "cudnn*.dll" }
                : new[] { "libcudnn.so*", "libcudnn*.so*" };

            foreach (var file in EnumerateCandidateLibraries(patterns))
            {
                try
                {
                    var fvi = FileVersionInfo.GetVersionInfo(file);
                    if (!string.IsNullOrWhiteSpace(fvi.FileVersion))
                    {
                        return fvi.FileVersion;
                    }
                    if (!string.IsNullOrWhiteSpace(fvi.ProductVersion))
                    {
                        return fvi.ProductVersion;
                    }
                }
                catch { /* ignore */ }
            }

            return null;
        }

        // ---- Helpers: enumerate candidate library files on PATH and common locations ----

        private static System.Collections.Generic.IEnumerable<string> EnumerateCandidateLibraries(string[] patterns)
        {
            var seen = new System.Collections.Generic.HashSet<string>(StringComparer.OrdinalIgnoreCase);

            // Enumerate local and PATH locations
            var dirs = new System.Collections.Generic.List<string?>
            {
                AppContext.BaseDirectory,
                Environment.CurrentDirectory
            };

            if (OperatingSystem.IsWindows())
            {
                dirs.Add(Environment.SystemDirectory);
                dirs.Add(Environment.GetFolderPath(Environment.SpecialFolder.System));
            }

            var pathEnv = Environment.GetEnvironmentVariable("PATH");
            if (!string.IsNullOrEmpty(pathEnv))
            {
                var split = pathEnv.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries);
                foreach (var d in split) dirs.Add(d);
            }

            foreach (var d in dirs)
            {
                foreach (var f in EnumerateFromDir(d, patterns, seen))
                    yield return f;
            }

            static System.Collections.Generic.IEnumerable<string> EnumerateFromDir(string? d, string[] pats, System.Collections.Generic.HashSet<string> seenSet)
            {
                if (string.IsNullOrWhiteSpace(d) || !Directory.Exists(d)) yield break;
                foreach (var p in pats)
                {
                    System.Collections.Generic.IEnumerable<string> files = Array.Empty<string>();
                    try { files = Directory.EnumerateFiles(d, p); } catch { /* ignore */ }
                    foreach (var f in files)
                    {
                        if (seenSet.Add(f)) yield return f;
                    }
                }
            }
        }
    }
}
