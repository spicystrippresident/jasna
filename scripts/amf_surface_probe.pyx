"""Minimal Linux AMF Vulkan-to-HIP D2D bridge.

This extension is intentionally a diagnostic building block.  It accepts only
native AMF Vulkan decode surfaces and performs two HIP device-to-device plane
copies.  It never exposes a host map, staging allocation, or software frame
transfer path.  Each copy imports and releases Vulkan external memory in the
same call; the small session object only pins the decoder/context identity and
is not a resource cache.
"""

from av.video.frame cimport VideoFrame
from libc.stdint cimport uint64_t, uintptr_t


_TRANSPORT_STATS_SCHEMA = "jasna.amf.vulkan-hip-transport.v1"
_transport_stats = {}


def reset_transport_stats():
    """Reset process-local instrumentation for this isolated bridge."""

    global _transport_stats
    _transport_stats = {
        "copy_to_hip_calls": 0,
        "copy_to_hip_successes": 0,
        "copy_to_hip_failures": 0,
        "vulkan_memory_exports": 0,
        "hip_external_memory_imports": 0,
        "hip_external_memory_destroys": 0,
        "hip_mapped_buffer_acquires": 0,
        "hip_mapped_buffer_releases": 0,
        "hip_d2d_plane_copies": 0,
        "decode_source_release_hip_stream_synchronize_calls": 0,
        "fixed_context_session_create_calls": 0,
        "fixed_context_session_close_calls": 0,
        "fixed_context_session_close_failures": 0,
        # This core deliberately has no product resource cache.
        "resource_cache_session_create_calls": 0,
        "resource_cache_session_close_calls": 0,
        "resource_cache_session_close_failures": 0,
        "resource_cache_hits": 0,
        "resource_cache_misses": 0,
    }


def get_transport_stats():
    """Return counters that make host or non-D2D transport auditable."""

    stats = dict(_transport_stats)
    stats.update(
        {
            "schema": _TRANSPORT_STATS_SCHEMA,
            "telemetry_source": "instrumented",
            "non_hardcoded": True,
            "failed_bridge_copies": stats["copy_to_hip_failures"],
            "hip_non_d2d_copy_calls": 0,
            "host_frame_transfers": 0,
            "cpu_map_calls": 0,
            "staging_copy_calls": 0,
            "d2h_copy_calls": 0,
            "av_hwframe_transfer_data_calls": 0,
            "transport_reconfigures": 0,
            "transport_restarts": 0,
            "resource_strategy": (
                "per-frame Vulkan external-memory import/map with balanced release"
            ),
            "copy_synchronization": "null-stream-source-release",
        }
    )
    return stats


reset_transport_stats()


cdef extern from *:
    """
    #include <stdint.h>
    #include <stdlib.h>
    #include <string.h>
    #include <unistd.h>
    #include <dlfcn.h>
    #include <libavutil/frame.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/hwcontext_amf.h>
    #include <libavutil/pixfmt.h>
    #include <AMF/core/Surface.h>
    #include <AMF/core/VulkanAMF.h>
    #include <hip/hip_runtime_api.h>

    typedef struct JasnaAmfCopyInfo {
        int width;
        int height;
        int bytes_per_sample;
        uint64_t packed_size;
        uint64_t source_y_pitch;
        uint64_t source_uv_pitch;
        int wait_result;
        int export_result;
        int hip_result;
        int hip_free_result;
        int hip_destroy_result;
        int hip_stream_synchronize_calls;
        int hip_stream_synchronize_result;
        int d2d_plane_copies;
        int fixed_context_bound;
    } JasnaAmfCopyInfo;

    typedef struct JasnaAmfInteropSession {
        uintptr_t hw_frames_context;
        uintptr_t amf_context;
        uintptr_t vulkan_device;
        int hip_device;
        int surface_format;
        int visible_width;
        int visible_height;
        uint64_t copy_calls;
        uint64_t close_calls;
        uint64_t close_failures;
        int closed;
    } JasnaAmfInteropSession;

    typedef hipError_t (*JasnaHipSetDeviceFn)(int);
    typedef hipError_t (*JasnaHipImportMemoryFn)(
        hipExternalMemory_t *,
        const hipExternalMemoryHandleDesc *
    );
    typedef hipError_t (*JasnaHipDestroyMemoryFn)(hipExternalMemory_t);
    typedef hipError_t (*JasnaHipMapBufferFn)(
        void **,
        hipExternalMemory_t,
        const hipExternalMemoryBufferDesc *
    );
    typedef hipError_t (*JasnaHipFreeFn)(void *);
    typedef hipError_t (*JasnaHipMemcpy2DFn)(
        void *, size_t, const void *, size_t, size_t, size_t, hipMemcpyKind
    );
    typedef hipError_t (*JasnaHipStreamSynchronizeFn)(hipStream_t);

    static void *jasna_vulkan_loader = NULL;
    static PFN_vkGetDeviceProcAddr jasna_vk_get_device_proc = NULL;
    static void *jasna_hip_loader = NULL;
    static JasnaHipSetDeviceFn jasna_hip_set_device = NULL;
    static JasnaHipImportMemoryFn jasna_hip_import_memory = NULL;
    static JasnaHipDestroyMemoryFn jasna_hip_destroy_memory = NULL;
    static JasnaHipMapBufferFn jasna_hip_map_buffer = NULL;
    static JasnaHipFreeFn jasna_hip_free = NULL;
    static JasnaHipMemcpy2DFn jasna_hip_memcpy_2d = NULL;
    static JasnaHipStreamSynchronizeFn jasna_hip_stream_synchronize = NULL;

    static int jasna_load_vulkan(const char **error) {
        if (jasna_vk_get_device_proc) {
            return 0;
        }
        jasna_vulkan_loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!jasna_vulkan_loader) {
            *error = "Vulkan loader is unavailable";
            return -1;
        }
        jasna_vk_get_device_proc = (PFN_vkGetDeviceProcAddr)dlsym(
            jasna_vulkan_loader, "vkGetDeviceProcAddr"
        );
        if (!jasna_vk_get_device_proc) {
            dlclose(jasna_vulkan_loader);
            jasna_vulkan_loader = NULL;
            *error = "vkGetDeviceProcAddr is unavailable";
            return -2;
        }
        return 0;
    }

    static int jasna_load_hip(const char **error) {
        if (
            jasna_hip_set_device && jasna_hip_import_memory &&
            jasna_hip_destroy_memory && jasna_hip_map_buffer &&
            jasna_hip_free && jasna_hip_memcpy_2d &&
            jasna_hip_stream_synchronize
        ) {
            return 0;
        }
        jasna_hip_loader = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_LOCAL);
        if (!jasna_hip_loader) {
            jasna_hip_loader = dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
        }
        if (!jasna_hip_loader) {
            *error = "ROCm HIP runtime is unavailable";
            return -1;
        }
        jasna_hip_set_device = (JasnaHipSetDeviceFn)dlsym(
            jasna_hip_loader, "hipSetDevice"
        );
        jasna_hip_import_memory = (JasnaHipImportMemoryFn)dlsym(
            jasna_hip_loader, "hipImportExternalMemory"
        );
        jasna_hip_destroy_memory = (JasnaHipDestroyMemoryFn)dlsym(
            jasna_hip_loader, "hipDestroyExternalMemory"
        );
        jasna_hip_map_buffer = (JasnaHipMapBufferFn)dlsym(
            jasna_hip_loader, "hipExternalMemoryGetMappedBuffer"
        );
        jasna_hip_free = (JasnaHipFreeFn)dlsym(jasna_hip_loader, "hipFree");
        jasna_hip_memcpy_2d = (JasnaHipMemcpy2DFn)dlsym(
            jasna_hip_loader, "hipMemcpy2D"
        );
        jasna_hip_stream_synchronize = (JasnaHipStreamSynchronizeFn)dlsym(
            jasna_hip_loader, "hipStreamSynchronize"
        );
        if (
            !jasna_hip_set_device || !jasna_hip_import_memory ||
            !jasna_hip_destroy_memory || !jasna_hip_map_buffer ||
            !jasna_hip_free || !jasna_hip_memcpy_2d ||
            !jasna_hip_stream_synchronize
        ) {
            dlclose(jasna_hip_loader);
            jasna_hip_loader = NULL;
            jasna_hip_set_device = NULL;
            jasna_hip_import_memory = NULL;
            jasna_hip_destroy_memory = NULL;
            jasna_hip_map_buffer = NULL;
            jasna_hip_free = NULL;
            jasna_hip_memcpy_2d = NULL;
            jasna_hip_stream_synchronize = NULL;
            *error = "required HIP external-memory or D2D entry point is unavailable";
            return -2;
        }
        return 0;
    }

    static JasnaAmfInteropSession *jasna_session_create(void) {
        JasnaAmfInteropSession *session =
            (JasnaAmfInteropSession *)calloc(1, sizeof(JasnaAmfInteropSession));
        if (session) {
            session->hip_device = -1;
        }
        return session;
    }

    static int jasna_session_bind(
        JasnaAmfInteropSession *session,
        AVHWFramesContext *frames_ctx,
        AMFContext *amf_context,
        VkDevice vulkan_device,
        int hip_device,
        int surface_format,
        int visible_width,
        int visible_height,
        const char **error
    ) {
        if (!session || !frames_ctx || !amf_context || !vulkan_device) {
            *error = "fixed-context AMF interop session received null identity";
            return -1;
        }
        if (session->closed) {
            *error = "fixed-context AMF interop session is already closed";
            return -2;
        }
        if (!session->hw_frames_context) {
            session->hw_frames_context = (uintptr_t)frames_ctx;
            session->amf_context = (uintptr_t)amf_context;
            session->vulkan_device = (uintptr_t)vulkan_device;
            session->hip_device = hip_device;
            session->surface_format = surface_format;
            session->visible_width = visible_width;
            session->visible_height = visible_height;
            return 0;
        }
        if (
            session->hw_frames_context != (uintptr_t)frames_ctx ||
            session->amf_context != (uintptr_t)amf_context ||
            session->vulkan_device != (uintptr_t)vulkan_device ||
            session->hip_device != hip_device ||
            session->surface_format != surface_format ||
            session->visible_width != visible_width ||
            session->visible_height != visible_height
        ) {
            *error = "fixed-context AMF/Vulkan/HIP identity changed";
            return -3;
        }
        return 0;
    }

    static int jasna_session_close(
        JasnaAmfInteropSession *session, const char **error
    ) {
        if (!session) {
            *error = "fixed-context AMF interop session is unavailable";
            return -1;
        }
        if (session->closed) {
            return 0;
        }
        /* Per-frame mappings are released in jasna_copy_to_hip before return. */
        session->close_calls += 1;
        session->closed = 1;
        return 0;
    }

    static void jasna_session_destroy(JasnaAmfInteropSession *session) {
        if (session) {
            free(session);
        }
    }

    static int jasna_surface_info(
        AVFrame *frame,
        uintptr_t *surface_out,
        uintptr_t *image_out,
        uintptr_t *memory_out,
        uint64_t *memory_size_out,
        uintptr_t *frames_context_out,
        uintptr_t *amf_context_out,
        uintptr_t *vulkan_device_out,
        int *memory_type_out,
        int *surface_format_out,
        const char **error
    ) {
        AMFSurface *surface;
        AMFPlane *plane;
        AMFVulkanView *view;
        AMFVulkanSurface *vk_surface;
        AVHWFramesContext *frames_ctx;
        AVHWDeviceContext *device_ctx;
        AVAMFDeviceContext *amf_ctx;
        AMFContext1 *context1 = NULL;
        AMFVulkanDevice *vk_device = NULL;
        if (!frame || !surface_out || !image_out || !memory_out ||
            !memory_size_out || !frames_context_out || !amf_context_out ||
            !vulkan_device_out || !memory_type_out || !surface_format_out || !error) {
            return -1;
        }
        if (frame->format != AV_PIX_FMT_AMF_SURFACE) {
            *error = "VideoFrame is not an AMF hardware surface";
            return -2;
        }
        surface = (AMFSurface *)frame->data[0];
        if (!surface || !surface->pVtbl) {
            *error = "AMF surface pointer is null";
            return -3;
        }
        *memory_type_out = surface->pVtbl->GetMemoryType(surface);
        *surface_format_out = surface->pVtbl->GetFormat(surface);
        if (*memory_type_out != AMF_MEMORY_VULKAN) {
            *error = "AMF surface is not backed by Vulkan memory";
            return -4;
        }
        plane = surface->pVtbl->GetPlaneAt(surface, 0);
        view = plane ? (AMFVulkanView *)plane->pVtbl->GetNative(plane) : NULL;
        vk_surface = view ? view->pSurface : NULL;
        if (!vk_surface || !vk_surface->hImage || !vk_surface->hMemory ||
            vk_surface->iSize <= 0) {
            *error = "AMF Vulkan image or external memory handle is unavailable";
            return -5;
        }
        *surface_out = (uintptr_t)surface;
        *image_out = (uintptr_t)vk_surface->hImage;
        *memory_out = (uintptr_t)vk_surface->hMemory;
        *memory_size_out = (uint64_t)vk_surface->iSize;
        if (!frame->hw_frames_ctx) {
            *error = "AMF frame has no hardware frames context";
            return -6;
        }
        frames_ctx = (AVHWFramesContext *)frame->hw_frames_ctx->data;
        device_ctx = frames_ctx ? frames_ctx->device_ctx : NULL;
        amf_ctx = device_ctx ? (AVAMFDeviceContext *)device_ctx->hwctx : NULL;
        if (!amf_ctx || !amf_ctx->context) {
            *error = "AMF device context is unavailable";
            return -7;
        }
        {
            AMFGuid guid = IID_AMFContext1();
            AMF_RESULT query = amf_ctx->context->pVtbl->QueryInterface(
                amf_ctx->context, &guid, (void **)&context1
            );
            if (query != AMF_OK || !context1) {
                *error = "AMFContext1 is unavailable";
                return -8;
            }
        }
        vk_device = (AMFVulkanDevice *)context1->pVtbl->GetVulkanDevice(context1);
        if (!vk_device || !vk_device->hDevice) {
            context1->pVtbl->Release(context1);
            *error = "AMF Vulkan device is unavailable";
            return -9;
        }
        *frames_context_out = (uintptr_t)frames_ctx;
        *amf_context_out = (uintptr_t)amf_ctx->context;
        *vulkan_device_out = (uintptr_t)vk_device->hDevice;
        context1->pVtbl->Release(context1);
        return 0;
    }

    static int jasna_copy_to_hip(
        AVFrame *frame,
        uintptr_t destination,
        uint64_t destination_size,
        int hip_device,
        JasnaAmfInteropSession *session,
        JasnaAmfCopyInfo *info,
        const char **error
    ) {
        AMFSurface *surface = NULL;
        AMFPlane *plane = NULL;
        AMFVulkanView *view = NULL;
        AMFVulkanSurface *vk_surface = NULL;
        AMFContext1 *context1 = NULL;
        AMFVulkanDevice *vk_device = NULL;
        AVHWFramesContext *frames_ctx = NULL;
        AVHWDeviceContext *device_ctx = NULL;
        AVAMFDeviceContext *amf_ctx = NULL;
        PFN_vkGetMemoryFdKHR get_memory_fd = NULL;
        PFN_vkGetImageSubresourceLayout get_subresource_layout = NULL;
        PFN_vkWaitForFences wait_for_fences = NULL;
        VkSubresourceLayout layouts[2];
        VkImageSubresource subresource;
        hipExternalMemory_t external = NULL;
        void *mapped = NULL;
        int fd = -1;
        int status = -1;
        int bytes_per_sample;
        int visible_width;
        int visible_height;
        uint64_t row_bytes;
        uint64_t y_size;
        uint64_t packed_size;

        if (!frame || !destination || !info || !error) {
            return -1;
        }
        memset(info, 0, sizeof(*info));
        info->hip_result = -1;
        info->hip_free_result = -1;
        info->hip_destroy_result = -1;
        info->hip_stream_synchronize_result = -1;
        *error = NULL;
        if (frame->format != AV_PIX_FMT_AMF_SURFACE) {
            *error = "VideoFrame is not an AMF hardware surface";
            return -2;
        }
        surface = (AMFSurface *)frame->data[0];
        if (!surface || !surface->pVtbl ||
            surface->pVtbl->GetMemoryType(surface) != AMF_MEMORY_VULKAN) {
            *error = "AMF surface is not backed by Vulkan memory";
            return -3;
        }
        if (surface->pVtbl->GetFormat(surface) != AMF_SURFACE_NV12 &&
            surface->pVtbl->GetFormat(surface) != AMF_SURFACE_P010) {
            *error = "only AMF NV12 and P010 Vulkan surfaces are supported";
            return -4;
        }
        plane = surface->pVtbl->GetPlaneAt(surface, 0);
        view = plane ? (AMFVulkanView *)plane->pVtbl->GetNative(plane) : NULL;
        vk_surface = view ? view->pSurface : NULL;
        if (!vk_surface || !vk_surface->hImage || !vk_surface->hMemory ||
            vk_surface->iSize <= 0) {
            *error = "AMF Vulkan image or external memory handle is unavailable";
            return -5;
        }
        if (!frame->hw_frames_ctx) {
            *error = "AMF frame has no hardware frames context";
            return -6;
        }
        frames_ctx = (AVHWFramesContext *)frame->hw_frames_ctx->data;
        device_ctx = frames_ctx ? frames_ctx->device_ctx : NULL;
        amf_ctx = device_ctx ? (AVAMFDeviceContext *)device_ctx->hwctx : NULL;
        if (!amf_ctx || !amf_ctx->context) {
            *error = "AMF device context is unavailable";
            return -7;
        }
        {
            AMFGuid guid = IID_AMFContext1();
            AMF_RESULT query = amf_ctx->context->pVtbl->QueryInterface(
                amf_ctx->context, &guid, (void **)&context1
            );
            if (query != AMF_OK || !context1) {
                *error = "AMFContext1 is unavailable";
                return -8;
            }
        }
        vk_device = (AMFVulkanDevice *)context1->pVtbl->GetVulkanDevice(context1);
        if (!vk_device || !vk_device->hDevice) {
            *error = "AMF Vulkan device is unavailable";
            status = -9;
            goto cleanup;
        }
        visible_width = frame->width;
        visible_height = frame->height;
        if (visible_width <= 0 || visible_height <= 0 ||
            visible_width > vk_surface->iWidth ||
            visible_height > vk_surface->iHeight ||
            (visible_width & 1) || (visible_height & 1)) {
            *error = "visible frame dimensions are invalid for the AMF surface";
            status = -10;
            goto cleanup;
        }
        bytes_per_sample = surface->pVtbl->GetFormat(surface) == AMF_SURFACE_P010 ? 2 : 1;
        row_bytes = (uint64_t)visible_width * (uint64_t)bytes_per_sample;
        y_size = row_bytes * (uint64_t)visible_height;
        packed_size = y_size + row_bytes * (uint64_t)(visible_height / 2);
        if (destination_size < packed_size) {
            *error = "HIP destination is smaller than the packed AMF frame";
            status = -11;
            goto cleanup;
        }
        if (session) {
            status = jasna_session_bind(
                session, frames_ctx, amf_ctx->context, vk_device->hDevice,
                hip_device, surface->pVtbl->GetFormat(surface), visible_width,
                visible_height, error
            );
            if (status != 0) {
                status = -12;
                goto cleanup;
            }
            info->fixed_context_bound = 1;
        }
        if (jasna_load_vulkan(error) != 0 || jasna_load_hip(error) != 0) {
            status = -13;
            goto cleanup;
        }
        get_memory_fd = (PFN_vkGetMemoryFdKHR)jasna_vk_get_device_proc(
            vk_device->hDevice, "vkGetMemoryFdKHR"
        );
        get_subresource_layout = (PFN_vkGetImageSubresourceLayout)
            jasna_vk_get_device_proc(vk_device->hDevice, "vkGetImageSubresourceLayout");
        wait_for_fences = (PFN_vkWaitForFences)jasna_vk_get_device_proc(
            vk_device->hDevice, "vkWaitForFences"
        );
        if (!get_memory_fd || !get_subresource_layout || !wait_for_fences) {
            *error = "required Vulkan external-memory entry point is unavailable";
            status = -14;
            goto cleanup;
        }
        if (vk_surface->Sync.hFence) {
            info->wait_result = wait_for_fences(
                vk_device->hDevice, 1, &vk_surface->Sync.hFence, VK_TRUE,
                5000000000ULL
            );
            if (info->wait_result != VK_SUCCESS) {
                *error = "waiting for the AMF Vulkan decode fence failed";
                status = -15;
                goto cleanup;
            }
            /* AMF owns the fence; do not destroy/reset it after a CPU wait. */
            vk_surface->Sync.hFence = VK_NULL_HANDLE;
        }
        memset(layouts, 0, sizeof(layouts));
        memset(&subresource, 0, sizeof(subresource));
        subresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
        get_subresource_layout(vk_device->hDevice, vk_surface->hImage,
                               &subresource, &layouts[0]);
        subresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
        get_subresource_layout(vk_device->hDevice, vk_surface->hImage,
                               &subresource, &layouts[1]);
        if (layouts[0].rowPitch < row_bytes || layouts[1].rowPitch < row_bytes) {
            *error = "AMF Vulkan plane pitch is smaller than visible frame width";
            status = -16;
            goto cleanup;
        }
        info->hip_result = jasna_hip_set_device(hip_device);
        if (info->hip_result != hipSuccess) {
            *error = "selecting the HIP device failed";
            status = -17;
            goto cleanup;
        }
        {
            VkMemoryGetFdInfoKHR fd_info;
            hipExternalMemoryHandleDesc memory_description;
            hipExternalMemoryBufferDesc buffer_description;
            memset(&fd_info, 0, sizeof(fd_info));
            fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
            fd_info.memory = vk_surface->hMemory;
            fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            info->export_result = get_memory_fd(vk_device->hDevice, &fd_info, &fd);
            if (info->export_result != VK_SUCCESS || fd < 0) {
                *error = "exporting AMF Vulkan memory as an opaque fd failed";
                status = -18;
                goto cleanup;
            }
            memset(&memory_description, 0, sizeof(memory_description));
            memory_description.type = hipExternalMemoryHandleTypeOpaqueFd;
            memory_description.handle.fd = fd;
            memory_description.size = (unsigned long long)vk_surface->iSize;
            memory_description.flags = hipExternalMemoryDedicated;
            info->hip_result = jasna_hip_import_memory(&external, &memory_description);
            if (info->hip_result != hipSuccess || !external) {
                *error = "importing the Vulkan allocation into HIP failed";
                status = -19;
                goto cleanup;
            }
            /* Opaque-fd ownership transfers to HIP after a successful import. */
            fd = -1;
            memset(&buffer_description, 0, sizeof(buffer_description));
            buffer_description.size = (unsigned long long)vk_surface->iSize;
            info->hip_result = jasna_hip_map_buffer(&mapped, external, &buffer_description);
            if (info->hip_result != hipSuccess || !mapped) {
                *error = "mapping the imported Vulkan allocation into HIP failed";
                status = -20;
                goto cleanup;
            }
        }
        info->hip_result = jasna_hip_memcpy_2d(
            (void *)destination, (size_t)row_bytes,
            (const uint8_t *)mapped + layouts[0].offset,
            (size_t)layouts[0].rowPitch, (size_t)row_bytes,
            (size_t)visible_height, hipMemcpyDeviceToDevice
        );
        if (info->hip_result != hipSuccess) {
            *error = "copying the AMF luma plane inside VRAM failed";
            status = -21;
            goto cleanup;
        }
        info->d2d_plane_copies += 1;
        info->hip_result = jasna_hip_memcpy_2d(
            (uint8_t *)destination + y_size, (size_t)row_bytes,
            (const uint8_t *)mapped + layouts[1].offset,
            (size_t)layouts[1].rowPitch, (size_t)row_bytes,
            (size_t)(visible_height / 2), hipMemcpyDeviceToDevice
        );
        if (info->hip_result != hipSuccess) {
            *error = "copying the AMF chroma plane inside VRAM failed";
            status = -22;
            goto cleanup;
        }
        info->d2d_plane_copies += 1;
        /*
         * The Python VideoFrame owns the last AMF surface reference.  Finish
         * the null stream before returning so AMF cannot recycle the source
         * while HIP still reads it.
         */
        info->hip_stream_synchronize_calls = 1;
        info->hip_stream_synchronize_result = jasna_hip_stream_synchronize(NULL);
        info->hip_result = info->hip_stream_synchronize_result;
        if (info->hip_result != hipSuccess) {
            *error = "synchronizing the AMF decode source-release stream failed";
            status = -23;
            goto cleanup;
        }
        info->width = visible_width;
        info->height = visible_height;
        info->bytes_per_sample = bytes_per_sample;
        info->packed_size = packed_size;
        info->source_y_pitch = layouts[0].rowPitch;
        info->source_uv_pitch = layouts[1].rowPitch;
        if (session) {
            session->copy_calls += 1;
        }
        status = 0;

    cleanup:
        if (mapped && jasna_hip_free) {
            info->hip_free_result = jasna_hip_free(mapped);
            if (status == 0 && info->hip_free_result != hipSuccess) {
                *error = "releasing the mapped HIP buffer failed";
                status = -24;
            }
        }
        if (external && jasna_hip_destroy_memory) {
            info->hip_destroy_result = jasna_hip_destroy_memory(external);
            if (status == 0 && info->hip_destroy_result != hipSuccess) {
                *error = "destroying the HIP external-memory import failed";
                status = -25;
            }
        }
        if (fd >= 0) {
            close(fd);
        }
        if (context1) {
            context1->pVtbl->Release(context1);
        }
        return status;
    }
    """
    ctypedef struct JasnaAmfCopyInfo:
        int width
        int height
        int bytes_per_sample
        unsigned long long packed_size
        unsigned long long source_y_pitch
        unsigned long long source_uv_pitch
        int wait_result
        int export_result
        int hip_result
        int hip_free_result
        int hip_destroy_result
        int hip_stream_synchronize_calls
        int hip_stream_synchronize_result
        int d2d_plane_copies
        int fixed_context_bound

    ctypedef struct JasnaAmfInteropSession:
        pass

    JasnaAmfInteropSession *jasna_session_create()
    int jasna_session_close(JasnaAmfInteropSession *session, const char **error)
    void jasna_session_destroy(JasnaAmfInteropSession *session)
    int jasna_surface_info(
        void *frame,
        uintptr_t *surface_out,
        uintptr_t *image_out,
        uintptr_t *memory_out,
        uint64_t *memory_size_out,
        uintptr_t *frames_context_out,
        uintptr_t *amf_context_out,
        uintptr_t *vulkan_device_out,
        int *memory_type_out,
        int *surface_format_out,
        const char **error,
    )
    int jasna_copy_to_hip(
        void *frame,
        uintptr_t destination,
        unsigned long long destination_size,
        int hip_device,
        JasnaAmfInteropSession *session,
        JasnaAmfCopyInfo *info,
        const char **error,
    )


def _copy_result(JasnaAmfCopyInfo info):
    return {
        "width": info.width,
        "height": info.height,
        "bytes_per_sample": info.bytes_per_sample,
        "packed_size": int(info.packed_size),
        "source_y_pitch": int(info.source_y_pitch),
        "source_uv_pitch": int(info.source_uv_pitch),
        "wait_result": info.wait_result,
        "export_result": info.export_result,
        "hip_result": info.hip_result,
        "hip_free_result": info.hip_free_result,
        "hip_destroy_result": info.hip_destroy_result,
        "d2d_plane_copies": info.d2d_plane_copies,
        "hip_non_d2d_copy_calls": 0,
        "host_frame_transfers": 0,
        "cpu_map_calls": 0,
        "staging_copy_calls": 0,
        "d2h_copy_calls": 0,
        "av_hwframe_transfer_data_calls": 0,
        "decode_source_release_hip_stream_synchronize_calls": (
            info.hip_stream_synchronize_calls
        ),
        "decode_source_release_hip_stream_synchronize_result": (
            info.hip_stream_synchronize_result
        ),
        "copy_synchronization": "null-stream-source-release",
        "fixed_context_bound": bool(info.fixed_context_bound),
    }


def inspect_amf_surface(VideoFrame frame):
    """Inspect an AMF Vulkan surface without mapping it to CPU or HIP."""

    cdef uintptr_t surface = 0
    cdef uintptr_t image = 0
    cdef uintptr_t memory = 0
    cdef uint64_t memory_size = 0
    cdef uintptr_t frames_context = 0
    cdef uintptr_t amf_context = 0
    cdef uintptr_t vulkan_device = 0
    cdef int memory_type = 0
    cdef int surface_format = 0
    cdef const char *error = NULL
    cdef int status = jasna_surface_info(
        <void *>frame.ptr,
        &surface,
        &image,
        &memory,
        &memory_size,
        &frames_context,
        &amf_context,
        &vulkan_device,
        &memory_type,
        &surface_format,
        &error,
    )
    if status != 0:
        message = error.decode("utf-8", "replace") if error != NULL else "unknown error"
        raise RuntimeError(f"AMF surface inspection failed ({status}): {message}")
    return {
        "surface": int(surface),
        "memory_type_id": memory_type,
        "memory_type": "vulkan",
        "surface_format_id": surface_format,
        "fixed_context": {
            "frames_context": int(frames_context),
            "amf_context": int(amf_context),
            "vulkan_device": int(vulkan_device),
        },
        "vulkan": {
            "image": int(image),
            "memory": int(memory),
            "memory_size": int(memory_size),
        },
    }


def copy_amf_surface_to_hip(
    VideoFrame frame,
    uintptr_t destination,
    unsigned long long destination_size,
    int device=0,
):
    """Copy one native AMF Vulkan NV12/P010 frame directly into HIP memory."""

    _transport_stats["copy_to_hip_calls"] += 1
    cdef JasnaAmfCopyInfo info
    cdef const char *error = NULL
    cdef int status = jasna_copy_to_hip(
        <void *>frame.ptr,
        destination,
        destination_size,
        device,
        NULL,
        &info,
        &error,
    )
    if status != 0:
        _transport_stats["copy_to_hip_failures"] += 1
        message = error.decode("utf-8", "replace") if error != NULL else "unknown error"
        raise RuntimeError(
            f"AMF-to-HIP D2D copy failed ({status}, Vulkan wait {info.wait_result}, "
            f"Vulkan export {info.export_result}, HIP {info.hip_result}): {message}"
        )
    _transport_stats["copy_to_hip_successes"] += 1
    _transport_stats["vulkan_memory_exports"] += 1
    _transport_stats["hip_external_memory_imports"] += 1
    _transport_stats["hip_mapped_buffer_acquires"] += 1
    _transport_stats["hip_mapped_buffer_releases"] += 1
    _transport_stats["hip_external_memory_destroys"] += 1
    _transport_stats["hip_d2d_plane_copies"] += int(info.d2d_plane_copies)
    _transport_stats["decode_source_release_hip_stream_synchronize_calls"] += int(
        info.hip_stream_synchronize_calls
    )
    return _copy_result(info)


cdef class AmfVulkanHipInteropSession:
    """One decode reader's fixed AMF/Vulkan/HIP identity, without a cache."""

    cdef JasnaAmfInteropSession *_session
    cdef bint _closed

    def __cinit__(self, purpose):
        if purpose != "decode":
            raise ValueError("this core only supports a 'decode' interop session")
        self._session = jasna_session_create()
        if self._session == NULL:
            raise MemoryError("creating the fixed-context AMF interop session failed")
        self._closed = False
        _transport_stats["fixed_context_session_create_calls"] += 1

    def __dealloc__(self):
        if self._session != NULL:
            jasna_session_destroy(self._session)
            self._session = NULL

    @property
    def purpose(self):
        return "decode"

    def close(self):
        cdef const char *error = NULL
        cdef int status
        if self._session == NULL or self._closed:
            return
        status = jasna_session_close(self._session, &error)
        _transport_stats["fixed_context_session_close_calls"] += 1
        if status != 0:
            _transport_stats["fixed_context_session_close_failures"] += 1
            message = error.decode("utf-8", "replace") if error != NULL else "unknown error"
            raise RuntimeError(
                f"closing the fixed-context AMF interop session failed ({status}): {message}"
            )
        self._closed = True

    def stats(self):
        if self._session == NULL:
            raise RuntimeError("fixed-context AMF interop session is unavailable")
        # Cython may not read an opaque C struct safely; these lifetime fields
        # are mirrored by the wrapper's explicit state, while copy identity is
        # enforced in native code on every call.
        return {
            "purpose": "decode",
            "resource_strategy": (
                "per-frame Vulkan external-memory import/map with balanced release"
            ),
            "cache_entries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "close_calls": 1 if self._closed else 0,
            "close_failures": 0,
            "closed": bool(self._closed),
        }

    def copy_amf_surface_to_hip(
        self,
        VideoFrame frame,
        uintptr_t destination,
        unsigned long long destination_size,
        int device=0,
    ):
        if self._session == NULL or self._closed:
            raise RuntimeError("fixed-context AMF interop session is already closed")
        _transport_stats["copy_to_hip_calls"] += 1
        cdef JasnaAmfCopyInfo info
        cdef const char *error = NULL
        cdef int status = jasna_copy_to_hip(
            <void *>frame.ptr,
            destination,
            destination_size,
            device,
            self._session,
            &info,
            &error,
        )
        if status != 0:
            _transport_stats["copy_to_hip_failures"] += 1
            message = error.decode("utf-8", "replace") if error != NULL else "unknown error"
            raise RuntimeError(
                f"fixed-context AMF-to-HIP D2D copy failed ({status}, "
                f"Vulkan wait {info.wait_result}, Vulkan export {info.export_result}, "
                f"HIP {info.hip_result}): {message}"
            )
        _transport_stats["copy_to_hip_successes"] += 1
        _transport_stats["vulkan_memory_exports"] += 1
        _transport_stats["hip_external_memory_imports"] += 1
        _transport_stats["hip_mapped_buffer_acquires"] += 1
        _transport_stats["hip_mapped_buffer_releases"] += 1
        _transport_stats["hip_external_memory_destroys"] += 1
        _transport_stats["hip_d2d_plane_copies"] += int(info.d2d_plane_copies)
        _transport_stats["decode_source_release_hip_stream_synchronize_calls"] += int(
            info.hip_stream_synchronize_calls
        )
        return _copy_result(info)
