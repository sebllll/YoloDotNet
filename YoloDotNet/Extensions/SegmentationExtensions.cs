// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Extensions
{
    /// <summary>
    /// Provides extension methods for working with segmentation masks.
    /// </summary>
    public static class SegmentationExtensions
    {
        /// <summary>
        /// Unpacks a bit-packed pixel mask into a byte array where each byte is either 0 or 255.
        /// </summary>
        /// <param name="packedMask">The source bit-packed mask.</param>
        /// <param name="width">The width of the mask.</param>
        /// <param name="height">The height of the mask.</param>
        /// <returns>A new byte array representing the unpacked mask.</returns>
        public static byte[] UnpackPixelMaskToByteArray(this byte[] packedMask, int width, int height)
        {
            int totalPixels = width * height;
            var bytes = new byte[totalPixels];
            UnpackPixelMaskToSpan(packedMask, bytes, width, height);
            return bytes;
        }

        /// <summary>
        /// Unpacks a bit-packed pixel mask into a destination span where each byte is either 0 or 255.
        /// </summary>
        /// <param name="packedMask">The source bit-packed mask.</param>
        /// <param name="destination">The destination span to write the unpacked mask to.</param>
        /// <param name="width">The width of the mask.</param>
        /// <param name="height">The height of the mask.</param>
        public static void UnpackPixelMaskToSpan(this byte[] packedMask, Span<byte> destination, int width, int height)
        {
            int totalPixels = width * height;
            if (destination.Length < totalPixels)
            {
                throw new ArgumentException("Destination span is too small to hold the unpacked mask.", nameof(destination));
            }

            destination.Clear();

            for (int i = 0; i < totalPixels; i++)
            {
                int byteIndex = i >> 3; // i / 8
                int bitIndex = i & 0b0111; // i % 8

                if (byteIndex < packedMask.Length && (packedMask[byteIndex] & (1 << bitIndex)) != 0)
                {
                    destination[i] = 255;
                }
            }
        }
    }
}