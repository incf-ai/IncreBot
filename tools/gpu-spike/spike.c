// Milestone 1 spike (gpu-plan.md).
//   spike measure <pci>          sysfs read cost, device overhead, per-size slack,
//                                visibility latency, fdinfo vs. sysfs
//   spike hold <pci> <MiB> <s>   allocate MiB device-local and hold for s seconds
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d (line %d)\n", #x, r_, __LINE__); exit(1); } } while (0)
#define MIB (1024.0 * 1024.0)

static char vram_path[128], gtt_path[128];

static double now_us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static long long read_ll(const char *path) {
    char b[32]; int fd = open(path, O_RDONLY);
    ssize_t n = read(fd, b, sizeof b - 1); close(fd);
    b[n > 0 ? n : 0] = 0; return atoll(b);
}
static long long vram(void) { return read_ll(vram_path); }
static long long gtt(void) { return read_ll(gtt_path); }

// Sum drm-memory-vram / drm-total-vram (KiB) over this process's DRM fds for pci,
// counting each drm-client-id once.
static void own_fdinfo(const char *pci, long long *mem_kib, long long *total_kib, int *clients) {
    *mem_kib = *total_kib = 0; *clients = 0;
    long seen[64]; int nseen = 0;
    DIR *d = opendir("/proc/self/fdinfo"); struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        char p[64], line[256]; snprintf(p, sizeof p, "/proc/self/fdinfo/%s", e->d_name);
        FILE *f = fopen(p, "r"); if (!f) continue;
        long id = -1; int match = 0; long long m = 0, t = 0;
        while (fgets(line, sizeof line, f)) {
            if (!strncmp(line, "drm-pdev:", 9) && strstr(line, pci)) match = 1;
            sscanf(line, "drm-client-id: %ld", &id);
            sscanf(line, "drm-memory-vram: %lld", &m);
            sscanf(line, "drm-total-vram: %lld", &t);
        }
        fclose(f);
        if (!match || id < 0) continue;
        int dup = 0; for (int i = 0; i < nseen; i++) dup |= seen[i] == id;
        if (dup) continue;
        seen[nseen++] = id; *mem_kib += m; *total_kib += t; (*clients)++;
    }
    closedir(d);
}

// Wait until vram() != from, busy-polling; returns µs or -1 on timeout.
static double wait_change(long long from, double t_start, double timeout_us, long long *out) {
    for (;;) {
        long long v = vram(); double t = now_us();
        if (v != from) { *out = v; return t - t_start; }
        if (t - t_start > timeout_us) { *out = v; return -1; }
    }
}

static long long settle(void) { usleep(200000); return vram(); }

static VkDevice open_dev(const char *want, VkPhysicalDevice *out_pd) {
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai };
    VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 16; VkPhysicalDevice pds[16]; vkEnumeratePhysicalDevices(inst, &n, pds);
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDevicePCIBusInfoPropertiesEXT pci = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT };
        VkPhysicalDeviceProperties2 pr = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &pci };
        vkGetPhysicalDeviceProperties2(pds[i], &pr);
        char a[32]; snprintf(a, sizeof a, "%04x:%02x:%02x.%x", pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
        if (strcmp(a, want)) continue;
        float prio = 1;
        VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio };
        VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
        VkDevice dev; CK(vkCreateDevice(pds[i], &dci, NULL, &dev));
        *out_pd = pds[i]; return dev;
    }
    fprintf(stderr, "no Vulkan device at %s\n", want); exit(1);
}

static uint32_t mem_type(VkPhysicalDevice pd, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & want) == want && !(f & avoid)) return i;
    }
    fprintf(stderr, "no memory type\n"); exit(1);
}

static VkDeviceMemory alloc(VkDevice dev, uint32_t type, VkDeviceSize size) {
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = size, .memoryTypeIndex = type };
    VkDeviceMemory m; CK(vkAllocateMemory(dev, &mai, NULL, &m)); return m;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: spike measure|hold <pci> [MiB secs]\n"); return 1; }
    const char *pci = argv[2];
    snprintf(vram_path, sizeof vram_path, "/sys/bus/pci/devices/%s/mem_info_vram_used", pci);
    snprintf(gtt_path, sizeof gtt_path, "/sys/bus/pci/devices/%s/mem_info_gtt_used", pci);

    if (!strcmp(argv[1], "leak")) {
        // Leak 2 GiB, destroy the device; watch sysfs. The caller checks after exit.
        long long b = settle();
        VkPhysicalDevice pd; VkDevice dev = open_dev(pci, &pd);
        alloc(dev, mem_type(pd, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT), 2048ull << 20);
        printf("leak: allocated, +%.0f MiB\n", (settle() - b) / MIB);
        vkDestroyDevice(dev, NULL); sleep(2);
        printf("leak: 2 s after vkDestroyDevice, +%.0f MiB\n", (vram() - b) / MIB);
        fflush(stdout);
        return 0;
    }
    if (!strcmp(argv[1], "hold")) {
        VkPhysicalDevice pd; VkDevice dev = open_dev(pci, &pd);
        uint32_t dl = mem_type(pd, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        alloc(dev, dl, (VkDeviceSize)(atof(argv[3]) * MIB));
        printf("HOLDING %s MiB on %s, pid %d\n", argv[3], pci, getpid()); fflush(stdout);
        sleep(atoi(argv[4]));
        return 0;
    }

    // 1. Cost of one sysfs read.
    double t0 = now_us(); for (int i = 0; i < 20000; i++) vram();
    printf("sysfs read: %.2f us each\n", (now_us() - t0) / 20000);

    // 2. Overhead of the instance + VkDevice alone.
    long long base0 = settle();
    VkPhysicalDevice pd; VkDevice dev = open_dev(pci, &pd);
    long long base = settle();
    long long fm, ft; int fc; own_fdinfo(pci, &fm, &ft, &fc);
    printf("idle vram_used: %.2f MiB; after vkCreateDevice: %.2f MiB (+%.2f MiB device overhead); "
           "own fdinfo: drm-memory-vram %.2f MiB, drm-total-vram %.2f MiB, %d client(s)\n",
           base0 / MIB, base / MIB, (base - base0) / MIB, fm / 1024.0, ft / 1024.0, fc);

    // 3. Per-size: allocation visibility latency, sysfs delta (slack), fdinfo, release latency.
    uint32_t dl = mem_type(pd, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    double sizes[32] = { 0.004, 1, 4, 16, 64, 100, 256, 500, 1024, 4096 };
    unsigned nsizes = 10;
    if (argc > 3) { nsizes = 0; for (int i = 3; i < argc && nsizes < 32; i++) sizes[nsizes++] = atof(argv[i]); }
    printf("\n%9s %10s %12s %12s %10s %14s %12s\n", "alloc MiB", "alloc ms", "visible us", "delta MiB", "slack KiB", "fdinfo MiB", "free->gone us");
    for (unsigned k = 0; k < nsizes; k++) {
        VkDeviceSize sz = (VkDeviceSize)(sizes[k] * MIB);
        long long before = settle(), seen;
        double a0 = now_us();
        VkDeviceMemory m = alloc(dev, dl, sz);
        double a1 = now_us();
        double vis = wait_change(before, a1, 2e6, &seen);
        long long after = settle();
        own_fdinfo(pci, &fm, &ft, &fc);
        double f0 = now_us();
        vkFreeMemory(dev, m, NULL);
        double gone = wait_change(after, f0, 2e6, &seen);
        while (vram() > before + 65536 && now_us() - f0 < 5e6) {}
        double all = now_us() - f0;
        printf("%9.3f %10.2f %12.0f %12.3f %10.0f %6.1f / %5.1f %12.0f %10.1f ms\n", sizes[k], (a1 - a0) / 1e3, vis,
               (after - before) / MIB, (after - before - (double)sz) / 1024, fm / 1024.0, ft / 1024.0, gone, all / 1e3);
    }

    // 3b. Release of a realistic footprint: 40 × 200 MiB (≈ one weight tensor set
    // per layer, ~7.8 GiB), freed back to back, then the time until sysfs is back
    // at the pre-allocation level.
    {
        enum { NB = 40 }; VkDeviceMemory ms[NB];
        long long before = settle();
        for (int i = 0; i < NB; i++) ms[i] = alloc(dev, dl, 200 * 1024 * 1024);
        long long full = settle();
        double f0 = now_us();
        for (int i = 0; i < NB; i++) vkFreeMemory(dev, ms[i], NULL);
        double f1 = now_us(), first = -1;
        long long v;
        while ((v = vram()) > before + 1024 * 1024) { if (first < 0 && v < full) first = now_us() - f0; if (now_us() - f0 > 5e6) break; }
        printf("\n40 x 200 MiB (%.2f GiB): vkFreeMemory calls %.2f ms; first drop %.0f us; back to baseline %.1f ms\n",
               (full - before) / MIB / 1024, (f1 - f0) / 1e3, first, (now_us() - f0) / 1e3);
    }
    {
        long long before = settle();
        VkDeviceMemory m = alloc(dev, dl, 7800ull * 1024 * 1024);
        settle();
        double f0 = now_us(); vkFreeMemory(dev, m, NULL);
        while (vram() > before + 1024 * 1024 && now_us() - f0 < 5e6) {}
        printf("1 x 7800 MiB: back to baseline %.1f ms\n", (now_us() - f0) / 1e3);
    }
    {
        long long before = settle();
        VkDeviceMemory m = alloc(dev, dl, 7800ull * 1024 * 1024);
        settle();
        double f0 = now_us(); vkDestroyDevice(dev, NULL);
        while (vram() > before + 1024 * 1024 && now_us() - f0 < 5e6) {}
        printf("1 x 7800 MiB, freed by vkDestroyDevice: back to baseline %.1f ms\n", (now_us() - f0) / 1e3);
        (void)m;
        dev = open_dev(pci, &pd);
    }

    // 4. Host-visible (non-device-local) memory: VRAM or GTT?
    uint32_t hv = mem_type(pd, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    long long v0 = settle(), g0 = gtt();
    VkDeviceMemory h = alloc(dev, hv, 256 * 1024 * 1024);
    long long v1 = settle(), g1 = gtt();
    printf("\n256 MiB HOST_VISIBLE (not DEVICE_LOCAL): vram +%.2f MiB, gtt +%.2f MiB\n", (v1 - v0) / MIB, (g1 - g0) / MIB);
    vkFreeMemory(dev, h, NULL);
    uint32_t bar = mem_type(pd, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);
    v0 = settle(); g0 = gtt();
    h = alloc(dev, bar, 256 * 1024 * 1024);
    v1 = settle(); g1 = gtt();
    printf("256 MiB DEVICE_LOCAL|HOST_VISIBLE (BAR): vram +%.2f MiB, gtt +%.2f MiB\n", (v1 - v0) / MIB, (g1 - g0) / MIB);
    vkFreeMemory(dev, h, NULL);

    // 5. Device teardown returns to the idle baseline?
    vkDestroyDevice(dev, NULL);
    printf("after vkDestroyDevice: %.2f MiB (idle was %.2f)\n", settle() / MIB, base0 / MIB);
    return 0;
}
