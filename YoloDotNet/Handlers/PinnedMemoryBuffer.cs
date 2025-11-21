// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Handlers
{
    public class PinnedMemoryBuffer : IDisposable
    {
        public readonly SKImageInfo ImageInfo;
        public readonly byte[] Buffer;
        public readonly IntPtr Pointer;
        public readonly SKBitmap TargetBitmap;
        public readonly SKCanvas Canvas;

        private readonly GCHandle _handle;
        private bool _disposed;

        public PinnedMemoryBuffer(SKImageInfo imageInfo)
        {
            ImageInfo = imageInfo;

            //var _imageInfo = new SKImageInfo(width, height, SKColorType.Rgb888x, SKAlphaType.Opaque);
            Buffer = new byte[imageInfo.BytesSize];

            _handle = GCHandle.Alloc(Buffer, GCHandleType.Pinned);
            Pointer = _handle.AddrOfPinnedObject();

            // Wrap the pinned buffer in a SKBitmap so we can draw into it
            TargetBitmap = new SKBitmap();

            if (!TargetBitmap.InstallPixels(imageInfo, Pointer, imageInfo.RowBytes))
                throw new YoloDotNetException("Failed to install pixels into SKBitmap");

            Canvas = new SKCanvas(TargetBitmap);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            if (disposing)
            {
                // Dispose managed state (managed objects).
                Canvas?.Dispose();
                TargetBitmap?.Dispose();
            }

            // Free unmanaged resources (unmanaged objects) and override finalizer
            if (_handle.IsAllocated)
                _handle.Free();

            _disposed = true;
        }

        ~PinnedMemoryBuffer()
        {
            Dispose(false);
        }
    }
}
