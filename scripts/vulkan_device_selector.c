#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct VkInstance_T* VkInstance;
typedef struct VkPhysicalDevice_T* VkPhysicalDevice;
typedef int32_t VkResult;
typedef void (*PFN_vkVoidFunction)(void);
typedef PFN_vkVoidFunction (*PFN_vkGetInstanceProcAddr)(
        VkInstance instance, const char* name);
typedef VkResult (*PFN_vkEnumeratePhysicalDevices)(
        VkInstance instance, uint32_t* count, VkPhysicalDevice* devices);

enum {
    VK_SUCCESS = 0,
    VK_INCOMPLETE = 5,
    VK_ERROR_INITIALIZATION_FAILED = -3,
};

static PFN_vkGetInstanceProcAddr real_get_instance_proc_addr;
static void* real_vulkan_handle;

static void resolve_vulkan(void) {
    if (real_get_instance_proc_addr == NULL) {
        const char* real_library = getenv("MAESTRO_VK_REAL_LIBRARY");
        if (real_library != NULL && *real_library != '\0') {
            real_vulkan_handle = dlopen(
                    real_library, RTLD_NOW | RTLD_LOCAL);
            if (real_vulkan_handle != NULL) {
                real_get_instance_proc_addr = (PFN_vkGetInstanceProcAddr)
                        dlsym(real_vulkan_handle, "vkGetInstanceProcAddr");
            }
        } else {
            real_get_instance_proc_addr = (PFN_vkGetInstanceProcAddr)
                    dlsym(RTLD_NEXT, "vkGetInstanceProcAddr");
        }
        if (real_get_instance_proc_addr == NULL) {
            fprintf(stderr,
                    "MAESTRO_VK_SELECTOR unable to resolve vkGetInstanceProcAddr: %s\n",
                    dlerror());
        }
    }
}

static int requested_device_index(void) {
    const char* value = getenv("MAESTRO_VK_DEVICE_INDEX");
    if (value == NULL || *value == '\0') {
        return -1;
    }
    char* end = NULL;
    const long index = strtol(value, &end, 10);
    if (end == value || *end != '\0' || index < 0 || index > INT32_MAX) {
        fprintf(stderr, "MAESTRO_VK_SELECTOR invalid device index: %s\n", value);
        return -2;
    }
    return (int) index;
}

static VkResult selected_enumerate_physical_devices(
        VkInstance instance, uint32_t* count, VkPhysicalDevice* devices) {
    resolve_vulkan();
    if (real_get_instance_proc_addr == NULL || count == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    PFN_vkEnumeratePhysicalDevices real_enumerate =
            (PFN_vkEnumeratePhysicalDevices) real_get_instance_proc_addr(
                    instance, "vkEnumeratePhysicalDevices");
    if (real_enumerate == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    const int requested = requested_device_index();
    if (requested == -1) {
        return real_enumerate(instance, count, devices);
    }
    if (requested < 0) {
        *count = 0;
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    uint32_t physical_device_count = 0;
    VkResult result = real_enumerate(instance, &physical_device_count, NULL);
    if (result != VK_SUCCESS || (uint32_t) requested >= physical_device_count) {
        fprintf(stderr,
                "MAESTRO_VK_SELECTOR device %d unavailable; enumerated %u devices\n",
                requested, physical_device_count);
        *count = 0;
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (devices == NULL) {
        *count = 1;
        return VK_SUCCESS;
    }
    if (*count == 0) {
        return VK_INCOMPLETE;
    }

    VkPhysicalDevice* all_devices = (VkPhysicalDevice*) calloc(
            physical_device_count, sizeof(VkPhysicalDevice));
    if (all_devices == NULL) {
        *count = 0;
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    uint32_t loaded_count = physical_device_count;
    result = real_enumerate(instance, &loaded_count, all_devices);
    if (result == VK_SUCCESS || result == VK_INCOMPLETE) {
        devices[0] = all_devices[requested];
        *count = 1;
        fprintf(stderr,
                "MAESTRO_VK_SELECTOR selected Vulkan physical device index %d of %u\n",
                requested, physical_device_count);
        result = VK_SUCCESS;
    }
    free(all_devices);
    return result;
}

VkResult vkEnumeratePhysicalDevices(
        VkInstance instance, uint32_t* count, VkPhysicalDevice* devices) {
    return selected_enumerate_physical_devices(instance, count, devices);
}

PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance instance, const char* name) {
    resolve_vulkan();
    if (name != NULL && strcmp(name, "vkEnumeratePhysicalDevices") == 0) {
        return (PFN_vkVoidFunction) selected_enumerate_physical_devices;
    }
    return real_get_instance_proc_addr == NULL
            ? NULL
            : real_get_instance_proc_addr(instance, name);
}
