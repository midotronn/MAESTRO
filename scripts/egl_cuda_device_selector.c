#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef void *EGLDisplay;
typedef void *EGLDeviceEXT;
typedef unsigned int EGLBoolean;
typedef unsigned int EGLenum;
typedef int EGLint;
typedef intptr_t EGLAttrib;

#define EGL_FALSE 0
#define EGL_TRUE 1
#define EGL_PLATFORM_DEVICE_EXT 0x313F
#define EGL_CUDA_DEVICE_NV 0x323A
#define MAX_EGL_DEVICES 64

#ifndef AGENTLODGE_SELECTOR_BUILD_ID
#define AGENTLODGE_SELECTOR_BUILD_ID "development-unset"
#endif

const unsigned int agentlodge_egl_selector_version = 2;
const char agentlodge_egl_selector_build_id[] =
    AGENTLODGE_SELECTOR_BUILD_ID;

typedef void *(*DlsymFn)(void *, const char *);
typedef void *(*GetProcAddressFn)(const char *);
typedef EGLBoolean (*QueryDevicesFn)(EGLint, EGLDeviceEXT *, EGLint *);
typedef EGLBoolean (*QueryDeviceAttribFn)(EGLDeviceEXT, EGLint, EGLAttrib *);
typedef EGLDisplay (*GetPlatformDisplayExtFn)(EGLenum, void *, const EGLint *);

static DlsymFn real_dlsym;
static GetProcAddressFn real_get_proc_address;
static QueryDevicesFn real_query_devices;
static QueryDeviceAttribFn real_query_device_attrib;
static GetPlatformDisplayExtFn real_get_platform_display_ext;
static void *egl_library;
static int configured_cuda_index = INT_MIN;
static int selection_attempted;
static EGLDeviceEXT selected_device;
static int selected_egl_index = -1;

EGLDisplay eglGetDisplay(void *native_display);
EGLDisplay eglGetPlatformDisplayEXT(
    EGLenum platform,
    void *native_display,
    const EGLint *attributes
);
EGLDisplay eglGetPlatformDisplay(
    EGLenum platform,
    void *native_display,
    const EGLAttrib *attributes
);
EGLBoolean eglQueryDevicesEXT(
    EGLint max_devices,
    EGLDeviceEXT *devices,
    EGLint *num_devices
);
void *eglGetProcAddress(const char *name);

static void log_error(const char *message)
{
    fprintf(
        stderr,
        "[agentlodge-egl-selector pid=%ld] %s\n",
        (long)getpid(),
        message
    );
}

static int write_attestation(int requested, int selected, int egl_index)
{
    const char *path = getenv("AGENTLODGE_EGL_ATTESTATION_PATH");
    char temporary[PATH_MAX];
    int descriptor;
    int written;

    if (path == NULL || *path == '\0') {
        log_error(
            "AGENTLODGE_EGL_ATTESTATION_PATH is required when the selector is loaded"
        );
        return 0;
    }
    written = snprintf(
        temporary,
        sizeof(temporary),
        "%s.%ld.tmp",
        path,
        (long)getpid()
    );
    if (written < 0 || (size_t)written >= sizeof(temporary)) {
        log_error("EGL selector attestation path is too long");
        return 0;
    }
    descriptor = open(
        temporary,
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
        0600
    );
    if (descriptor < 0) {
        fprintf(
            stderr,
            "[agentlodge-egl-selector pid=%ld] could not create attestation %s: %s\n",
            (long)getpid(),
            temporary,
            strerror(errno)
        );
        return 0;
    }
    written = dprintf(
        descriptor,
        "{\"schema_version\":1,\"selector_version\":%u,"
        "\"selector_build_id\":\"%s\",\"pid\":%ld,"
        "\"requested_cuda_index\":%d,\"selected_cuda_index\":%d,"
        "\"egl_device_index\":%d}\n",
        agentlodge_egl_selector_version,
        agentlodge_egl_selector_build_id,
        (long)getpid(),
        requested,
        selected,
        egl_index
    );
    if (written < 0 || fsync(descriptor) != 0) {
        log_error("could not persist EGL selector attestation");
        close(descriptor);
        unlink(temporary);
        return 0;
    }
    if (close(descriptor) != 0) {
        log_error("could not close EGL selector attestation");
        unlink(temporary);
        return 0;
    }
    if (rename(temporary, path) != 0) {
        fprintf(
            stderr,
            "[agentlodge-egl-selector pid=%ld] could not publish attestation %s: %s\n",
            (long)getpid(),
            path,
            strerror(errno)
        );
        unlink(temporary);
        return 0;
    }
    return 1;
}

static DlsymFn get_real_dlsym(void)
{
    if (real_dlsym == NULL) {
        real_dlsym = (DlsymFn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
        if (real_dlsym == NULL) {
            real_dlsym = (DlsymFn)dlvsym(
                RTLD_NEXT,
                "dlsym",
                "GLIBC_2.2.5"
            );
        }
    }
    return real_dlsym;
}

static void *lookup_direct(const char *name)
{
    DlsymFn resolver = get_real_dlsym();
    void *symbol;

    if (resolver == NULL) {
        log_error("could not resolve the real dlsym");
        return NULL;
    }
    symbol = resolver(RTLD_NEXT, name);
    if (symbol == NULL && strncmp(name, "egl", 3) == 0) {
        if (egl_library == NULL) {
            egl_library = dlopen("libEGL.so.1", RTLD_LAZY | RTLD_LOCAL);
        }
        if (egl_library != NULL) {
            symbol = resolver(egl_library, name);
        }
    }
    return symbol;
}

static GetProcAddressFn get_real_get_proc_address(void)
{
    if (real_get_proc_address == NULL) {
        real_get_proc_address =
            (GetProcAddressFn)lookup_direct("eglGetProcAddress");
    }
    return real_get_proc_address;
}

static void *lookup_egl(const char *name)
{
    void *symbol = lookup_direct(name);
    GetProcAddressFn getter;

    if (symbol != NULL) {
        return symbol;
    }
    getter = get_real_get_proc_address();
    return getter == NULL ? NULL : getter(name);
}

static int selected_cuda_index(void)
{
    const char *value;
    char *end = NULL;
    long index;

    if (configured_cuda_index != INT_MIN) {
        return configured_cuda_index;
    }

    value = getenv("AGENTLODGE_GPU_INDEX");
    if (value == NULL || *value == '\0') {
        log_error(
            "AGENTLODGE_GPU_INDEX is required when the selector is loaded"
        );
        configured_cuda_index = -1;
        return configured_cuda_index;
    }
    index = strtol(value, &end, 10);
    if (
        end == value
        || *end != '\0'
        || index < 0
        || index > INT_MAX
    ) {
        fprintf(
            stderr,
            "[agentlodge-egl-selector pid=%ld] invalid "
            "AGENTLODGE_GPU_INDEX=%s\n",
            (long)getpid(),
            value
        );
        configured_cuda_index = -1;
        return configured_cuda_index;
    }
    configured_cuda_index = (int)index;
    return configured_cuda_index;
}

static QueryDevicesFn get_real_query_devices(void)
{
    if (real_query_devices == NULL) {
        real_query_devices = (QueryDevicesFn)lookup_egl(
            "eglQueryDevicesEXT"
        );
    }
    return real_query_devices;
}

static QueryDeviceAttribFn get_real_query_device_attrib(void)
{
    if (real_query_device_attrib == NULL) {
        real_query_device_attrib = (QueryDeviceAttribFn)lookup_egl(
            "eglQueryDeviceAttribEXT"
        );
    }
    return real_query_device_attrib;
}

static EGLDeviceEXT choose_device(void)
{
    EGLDeviceEXT devices[MAX_EGL_DEVICES];
    QueryDevicesFn query_devices;
    QueryDeviceAttribFn query_attrib;
    EGLint count = 0;
    int wanted;
    int matches = 0;

    if (selection_attempted) {
        return selected_device;
    }
    selection_attempted = 1;
    wanted = selected_cuda_index();
    if (wanted < 0) {
        return NULL;
    }
    query_devices = get_real_query_devices();
    query_attrib = get_real_query_device_attrib();
    if (query_devices == NULL || query_attrib == NULL) {
        log_error(
            "EGL_EXT_device_enumeration/EGL_NV_device_cuda unavailable"
        );
        return NULL;
    }
    if (!query_devices(MAX_EGL_DEVICES, devices, &count)) {
        log_error("eglQueryDevicesEXT failed");
        return NULL;
    }
    for (EGLint index = 0; index < count; ++index) {
        EGLAttrib cuda_index = -1;
        EGLBoolean queried = query_attrib(
            devices[index],
            EGL_CUDA_DEVICE_NV,
            &cuda_index
        );
        fprintf(
            stderr,
            "[agentlodge-egl-selector pid=%ld] egl_device=%d cuda_index=%ld%s\n",
            (long)getpid(),
            (int)index,
            (long)cuda_index,
            queried ? "" : " unavailable"
        );
        if (queried && cuda_index == (EGLAttrib)wanted) {
            selected_device = devices[index];
            selected_egl_index = (int)index;
            ++matches;
        }
    }
    if (matches != 1) {
        fprintf(
            stderr,
            "[agentlodge-egl-selector pid=%ld] CUDA index %d matched "
            "%d EGL devices; refusing default-device fallback\n",
            (long)getpid(),
            wanted,
            matches
        );
        selected_device = NULL;
        selected_egl_index = -1;
        return NULL;
    }
    if (!write_attestation(wanted, wanted, selected_egl_index)) {
        selected_device = NULL;
        selected_egl_index = -1;
        log_error("refusing EGL selection without a durable attestation");
        return NULL;
    }
    fprintf(
        stderr,
        "[agentlodge-egl-selector pid=%ld] selected CUDA index %d via "
        "EGL_CUDA_DEVICE_NV\n",
        (long)getpid(),
        wanted
    );
    return selected_device;
}

static EGLDisplay selected_platform_display(void)
{
    EGLDeviceEXT device = choose_device();

    if (device == NULL) {
        return NULL;
    }
    if (real_get_platform_display_ext == NULL) {
        real_get_platform_display_ext = (GetPlatformDisplayExtFn)lookup_egl(
            "eglGetPlatformDisplayEXT"
        );
    }
    if (real_get_platform_display_ext == NULL) {
        log_error("eglGetPlatformDisplayEXT unavailable");
        return NULL;
    }
    return real_get_platform_display_ext(
        EGL_PLATFORM_DEVICE_EXT,
        device,
        NULL
    );
}

EGLBoolean eglQueryDevicesEXT(
    EGLint max_devices,
    EGLDeviceEXT *devices,
    EGLint *num_devices
)
{
    EGLDeviceEXT device;

    if (num_devices == NULL) {
        return EGL_FALSE;
    }
    device = choose_device();
    if (device == NULL) {
        return EGL_FALSE;
    }
    *num_devices = 1;
    if (devices != NULL && max_devices > 0) {
        devices[0] = device;
    }
    return EGL_TRUE;
}

EGLDisplay eglGetDisplay(void *native_display)
{
    (void)native_display;
    return selected_platform_display();
}

EGLDisplay eglGetPlatformDisplayEXT(
    EGLenum platform,
    void *native_display,
    const EGLint *attributes
)
{
    (void)platform;
    (void)native_display;
    (void)attributes;
    return selected_platform_display();
}

EGLDisplay eglGetPlatformDisplay(
    EGLenum platform,
    void *native_display,
    const EGLAttrib *attributes
)
{
    (void)platform;
    (void)native_display;
    (void)attributes;
    return selected_platform_display();
}

void *eglGetProcAddress(const char *name)
{
    GetProcAddressFn getter;

    if (name == NULL) {
        return NULL;
    }
    if (strcmp(name, "eglGetDisplay") == 0) {
        return (void *)&eglGetDisplay;
    }
    if (strcmp(name, "eglQueryDevicesEXT") == 0) {
        return (void *)&eglQueryDevicesEXT;
    }
    if (strcmp(name, "eglGetPlatformDisplayEXT") == 0) {
        return (void *)&eglGetPlatformDisplayEXT;
    }
    if (strcmp(name, "eglGetPlatformDisplay") == 0) {
        return (void *)&eglGetPlatformDisplay;
    }
    getter = get_real_get_proc_address();
    return getter == NULL ? NULL : getter(name);
}

void *dlsym(void *handle, const char *name)
{
    DlsymFn resolver;

    if (strcmp(name, "eglGetDisplay") == 0) {
        return (void *)&eglGetDisplay;
    }
    if (strcmp(name, "eglQueryDevicesEXT") == 0) {
        return (void *)&eglQueryDevicesEXT;
    }
    if (strcmp(name, "eglGetPlatformDisplayEXT") == 0) {
        return (void *)&eglGetPlatformDisplayEXT;
    }
    if (strcmp(name, "eglGetPlatformDisplay") == 0) {
        return (void *)&eglGetPlatformDisplay;
    }
    if (strcmp(name, "eglGetProcAddress") == 0) {
        return (void *)&eglGetProcAddress;
    }
    resolver = get_real_dlsym();
    return resolver == NULL ? NULL : resolver(handle, name);
}
