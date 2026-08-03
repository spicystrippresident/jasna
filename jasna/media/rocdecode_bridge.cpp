/*
Copyright (c) 2026 Jasna contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <cstdint>
#include <exception>
#include <memory>
#include <string>

#include <hip/hip_runtime.h>
#include "roc_video_dec.h"

namespace {

struct Bridge {
    explicit Bridge(int device_id, rocDecVideoCodec codec)
        : decoder(std::make_unique<RocVideoDecoder>(
              device_id, OUT_SURFACE_MEM_DEV_INTERNAL, codec)) {}

    std::unique_ptr<RocVideoDecoder> decoder;
    std::string error;
};

thread_local std::string construction_error;

void set_error(Bridge *bridge, const char *message) {
    if (bridge) {
        bridge->error = message ? message : "unknown rocDecode bridge error";
    } else {
        construction_error = message ? message : "unknown rocDecode bridge error";
    }
}

template <typename Function>
int guarded(Bridge *bridge, Function &&function) {
    try {
        if (bridge) {
            bridge->error.clear();
        }
        function();
        return 0;
    } catch (const std::exception &error) {
        set_error(bridge, error.what());
    } catch (...) {
        set_error(bridge, "unknown C++ exception");
    }
    return -1;
}

}  // namespace

extern "C" {

void *jasna_rocdecode_create(int device_id, int codec) {
    construction_error.clear();
    try {
        return new Bridge(device_id, static_cast<rocDecVideoCodec>(codec));
    } catch (const std::exception &error) {
        set_error(nullptr, error.what());
    } catch (...) {
        set_error(nullptr, "unknown C++ exception while creating rocDecode");
    }
    return nullptr;
}

void jasna_rocdecode_destroy(void *opaque) {
    delete static_cast<Bridge *>(opaque);
}

const char *jasna_rocdecode_error(void *opaque) {
    auto *bridge = static_cast<Bridge *>(opaque);
    return bridge ? bridge->error.c_str() : construction_error.c_str();
}

int jasna_rocdecode_decode(
    void *opaque,
    const uint8_t *data,
    size_t size,
    int64_t pts,
    int end_of_stream,
    int *frames_available
) {
    auto *bridge = static_cast<Bridge *>(opaque);
    if (!bridge || !frames_available) {
        set_error(bridge, "invalid decode arguments");
        return -1;
    }
    return guarded(bridge, [&] {
        *frames_available = bridge->decoder->DecodeFrame(
            end_of_stream ? nullptr : data,
            end_of_stream ? 0 : size,
            0,
            pts
        );
    });
}

int jasna_rocdecode_copy_frame(
    void *opaque,
    uint8_t *destination,
    size_t destination_size,
    int64_t *pts,
    uint32_t *width,
    uint32_t *height,
    uint32_t *bit_depth
) {
    auto *bridge = static_cast<Bridge *>(opaque);
    if (!bridge || !destination || !pts || !width || !height || !bit_depth) {
        set_error(bridge, "invalid frame-copy arguments");
        return -1;
    }
    return guarded(bridge, [&] {
        int64_t frame_pts = 0;
        uint8_t *source = bridge->decoder->GetFrame(&frame_pts);
        if (!source) {
            throw std::runtime_error("rocDecode returned no frame surface");
        }

        OutputSurfaceInfo *info = nullptr;
        if (!bridge->decoder->GetOutputSurfaceInfo(&info) || !info) {
            bridge->decoder->ReleaseFrame(frame_pts);
            throw std::runtime_error("rocDecode returned no surface metadata");
        }
        if (info->surface_format != rocDecVideoSurfaceFormat_NV12 &&
            info->surface_format != rocDecVideoSurfaceFormat_P016) {
            bridge->decoder->ReleaseFrame(frame_pts);
            throw std::runtime_error("rocDecode output is not NV12/P016");
        }

        const size_t row_bytes =
            static_cast<size_t>(info->output_width) * info->bytes_per_pixel;
        const size_t packed_size = row_bytes *
            (static_cast<size_t>(info->output_height) + info->chroma_height);
        if (destination_size < packed_size) {
            bridge->decoder->ReleaseFrame(frame_pts);
            throw std::runtime_error("Torch destination is smaller than rocDecode output");
        }

        hipError_t status = hipMemcpy2D(
            destination,
            row_bytes,
            source,
            info->output_pitch,
            row_bytes,
            info->output_height,
            hipMemcpyDeviceToDevice
        );
        if (status == hipSuccess) {
            const uint8_t *source_uv = source +
                static_cast<size_t>(info->output_pitch) * info->output_vstride;
            uint8_t *destination_uv = destination + row_bytes * info->output_height;
            status = hipMemcpy2D(
                destination_uv,
                row_bytes,
                source_uv,
                info->output_pitch,
                row_bytes,
                info->chroma_height,
                hipMemcpyDeviceToDevice
            );
        }
        const bool released = bridge->decoder->ReleaseFrame(frame_pts);
        if (status != hipSuccess) {
            throw std::runtime_error(
                std::string("HIP surface copy failed: ") + hipGetErrorString(status));
        }
        if (!released) {
            throw std::runtime_error("rocDecode surface release failed");
        }

        *pts = frame_pts;
        *width = info->output_width;
        *height = info->output_height;
        *bit_depth = info->bit_depth;
    });
}

int jasna_rocdecode_drop_frame(
    void *opaque,
    int64_t *pts,
    uint32_t *width,
    uint32_t *height,
    uint32_t *bit_depth
) {
    auto *bridge = static_cast<Bridge *>(opaque);
    if (!bridge || !pts || !width || !height || !bit_depth) {
        set_error(bridge, "invalid frame-drop arguments");
        return -1;
    }
    return guarded(bridge, [&] {
        int64_t frame_pts = 0;
        uint8_t *source = bridge->decoder->GetFrame(&frame_pts);
        if (!source) {
            throw std::runtime_error("rocDecode returned no frame surface");
        }
        OutputSurfaceInfo *info = nullptr;
        if (!bridge->decoder->GetOutputSurfaceInfo(&info) || !info) {
            bridge->decoder->ReleaseFrame(frame_pts);
            throw std::runtime_error("rocDecode returned no surface metadata");
        }
        if (!bridge->decoder->ReleaseFrame(frame_pts)) {
            throw std::runtime_error("rocDecode surface release failed");
        }
        *pts = frame_pts;
        *width = info->output_width;
        *height = info->output_height;
        *bit_depth = info->bit_depth;
    });
}

}  // extern "C"
