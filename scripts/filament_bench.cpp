#include <backend/PixelBufferDescriptor.h>
#include <filament/Box.h>
#include <filament/Camera.h>
#include <filament/ColorGrading.h>
#include <filament/Engine.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/Options.h>
#include <filament/RenderTarget.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/Texture.h>
#include <filament/ToneMapper.h>
#include <filament/TransformManager.h>
#include <filament/View.h>
#include <filament/Viewport.h>
#include <gltfio/Animator.h>
#include <gltfio/AssetLoader.h>
#include <gltfio/FilamentAsset.h>
#include <gltfio/MaterialProvider.h>
#include <gltfio/ResourceLoader.h>
#include <gltfio/materials/uberarchive.h>
#include <image/Ktx1Bundle.h>
#include <ktxreader/Ktx1Reader.h>
#include <utils/EntityManager.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr uint32_t kWidth = 1080;
constexpr uint32_t kHeight = 1080;
constexpr size_t kPixelChannels = 4;
constexpr uint32_t kWarmupFrames = 60;
constexpr uint32_t kStaticFrames = 600;
constexpr uint32_t kAnimatedFrames = 300;
constexpr std::array<const char*, 22> kJointNames = {
    "m_avg_Pelvis", "m_avg_L_Hip", "m_avg_R_Hip", "m_avg_Spine1",
    "m_avg_L_Knee", "m_avg_R_Knee", "m_avg_Spine2", "m_avg_L_Ankle",
    "m_avg_R_Ankle", "m_avg_Spine3", "m_avg_L_Foot", "m_avg_R_Foot",
    "m_avg_Neck", "m_avg_L_Collar", "m_avg_R_Collar", "m_avg_Head",
    "m_avg_L_Shoulder", "m_avg_R_Shoulder", "m_avg_L_Elbow",
    "m_avg_R_Elbow", "m_avg_L_Wrist", "m_avg_R_Wrist",
};

struct PhaseMetrics {
    double animationUpdateSeconds = 0.0;
    double submissionSeconds = 0.0;
    double completionSeconds = 0.0;
    double totalSeconds = 0.0;
};

struct ReadbackState {
    std::atomic<bool> complete = false;
};

struct PixelStats {
    uint64_t hash = 0;
    uint64_t foregroundPixels = 0;
    float foregroundFraction = 0.0f;
    std::array<uint8_t, 3> background{};
    int minX = static_cast<int>(kWidth);
    int minY = static_cast<int>(kHeight);
    int maxX = -1;
    int maxY = -1;
};

struct CameraValues {
    double eyeX = 0.0;
    double eyeY = 0.0;
    double eyeZ = 0.0;
    double targetX = 0.0;
    double targetY = 0.0;
    double targetZ = 0.0;
    double distance = 0.0;
};

struct AssetInfo {
    size_t entityCount = 0;
    size_t renderableCount = 0;
    filament::Aabb worldBounds;
    CameraValues camera;
    bool usesAssetCamera = false;
};

void onReadback(void*, size_t, void* user) {
    static_cast<ReadbackState*>(user)->complete.store(
            true, std::memory_order_release);
}

double secondsBetween(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

float envFloat(const char* name, float fallback) {
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : std::stof(value);
}

uint32_t envUint(const char* name, uint32_t fallback) {
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : static_cast<uint32_t>(std::stoul(value));
}

std::string shellQuote(const std::string& value) {
    std::string result = "'";
    for (const char character : value) {
        result += character == '\'' ? "'\\''" : std::string(1, character);
    }
    return result + "'";
}

std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("unable to read " + path);
    }
    const auto byteCount = static_cast<size_t>(input.tellg());
    std::vector<uint8_t> bytes(byteCount);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return bytes;
}

bool renderAndReadback(filament::Renderer* renderer, filament::RenderTarget* renderTarget,
        filament::View* view, filament::Engine* engine, std::vector<uint8_t>& pixels,
        PhaseMetrics* metrics) {
    ReadbackState state;
    const auto submissionStart = Clock::now();
    renderer->renderStandaloneView(view);
    renderer->readPixels(renderTarget, 0, 0, kWidth, kHeight,
            filament::backend::PixelBufferDescriptor(
                    pixels.data(), pixels.size(),
                    filament::backend::PixelDataFormat::RGBA,
                    filament::backend::PixelDataType::UBYTE,
                    onReadback, &state));
    const auto submissionEnd = Clock::now();
    engine->flushAndWait();
    const auto completionEnd = Clock::now();
    if (!state.complete.load(std::memory_order_acquire)) {
        std::cerr << "readPixels callback did not complete after flushAndWait\n";
        return false;
    }
    metrics->submissionSeconds += secondsBetween(submissionStart, submissionEnd);
    metrics->completionSeconds += secondsBetween(submissionEnd, completionEnd);
    return true;
}

PixelStats inspectPixels(const std::vector<uint8_t>& pixels) {
    PixelStats stats;
    const std::array<size_t, 4> corners = {
        0,
        static_cast<size_t>(kWidth - 1),
        static_cast<size_t>(kHeight - 1) * kWidth,
        static_cast<size_t>(kHeight) * kWidth - 1,
    };
    for (size_t corner : corners) {
        for (size_t channel = 0; channel < 3; ++channel) {
            stats.background[channel] +=
                    pixels[corner * kPixelChannels + channel] / 4;
        }
    }

    constexpr uint64_t kFnvOffset = 1469598103934665603ull;
    constexpr uint64_t kFnvPrime = 1099511628211ull;
    stats.hash = kFnvOffset;
    for (uint32_t y = 0; y < kHeight; ++y) {
        for (uint32_t x = 0; x < kWidth; ++x) {
            const size_t offset =
                    (static_cast<size_t>(y) * kWidth + x) * kPixelChannels;
            uint32_t difference = 0;
            for (size_t channel = 0; channel < 3; ++channel) {
                const uint8_t value = pixels[offset + channel];
                stats.hash = (stats.hash ^ value) * kFnvPrime;
                difference += static_cast<uint32_t>(
                        std::abs(static_cast<int>(value) - static_cast<int>(stats.background[channel])));
            }
            if (difference > 36) {
                ++stats.foregroundPixels;
                stats.minX = std::min(stats.minX, static_cast<int>(x));
                stats.minY = std::min(stats.minY, static_cast<int>(y));
                stats.maxX = std::max(stats.maxX, static_cast<int>(x));
                stats.maxY = std::max(stats.maxY, static_cast<int>(y));
            }
        }
    }
    stats.foregroundFraction = static_cast<float>(stats.foregroundPixels) /
            static_cast<float>(kWidth * kHeight);
    return stats;
}

bool visiblyRendered(const PixelStats& stats, const char* label) {
    constexpr float kMinimumForegroundFraction = 0.002f;
    std::cout << label << "_foreground_pixels=" << stats.foregroundPixels
              << " foreground_fraction=" << std::fixed << std::setprecision(6)
              << stats.foregroundFraction
              << " foreground_bbox=[" << stats.minX << "," << stats.minY << ","
              << stats.maxX << "," << stats.maxY << "] hash=" << stats.hash << "\n";
    if (stats.foregroundFraction < kMinimumForegroundFraction ||
            stats.minX < 0 || stats.minY < 0 || stats.maxX < stats.minX || stats.maxY < stats.minY) {
        std::cerr << label << " failed visible-foreground validation\n";
        return false;
    }
    return true;
}

void writePpm(const std::string& path, const std::vector<uint8_t>& pixels) {
    std::ofstream output(path, std::ios::binary);
    output << "P6\n" << kWidth << " " << kHeight << "\n255\n";
    for (uint32_t y = 0; y < kHeight; ++y) {
        const uint8_t* row = pixels.data() +
                static_cast<size_t>(y) * kWidth * kPixelChannels;
        for (uint32_t x = 0; x < kWidth; ++x) {
            output.write(
                    reinterpret_cast<const char*>(row + x * kPixelChannels),
                    3);
        }
    }
}

void addPoint(filament::Aabb* bounds, filament::math::float3 point) {
    bounds->min.x = std::min(bounds->min.x, point.x);
    bounds->min.y = std::min(bounds->min.y, point.y);
    bounds->min.z = std::min(bounds->min.z, point.z);
    bounds->max.x = std::max(bounds->max.x, point.x);
    bounds->max.y = std::max(bounds->max.y, point.y);
    bounds->max.z = std::max(bounds->max.z, point.z);
}

AssetInfo configureAsset(filament::Engine* engine, filament::Scene* scene,
        filament::View* view, filament::Camera* camera, filament::gltfio::FilamentAsset* asset) {
    scene->addEntities(asset->getEntities(), asset->getEntityCount());
    auto& renderables = engine->getRenderableManager();
    auto& lights = engine->getLightManager();
    auto& transforms = engine->getTransformManager();
    filament::Aabb worldBounds;
    for (size_t index = 0; index < asset->getRenderableEntityCount(); ++index) {
        const auto entity = asset->getRenderableEntities()[index];
        const auto renderable = renderables.getInstance(entity);
        renderables.setCulling(renderable, false);
        const auto localBox = renderables.getAxisAlignedBoundingBox(renderable);
        const auto extent = localBox.halfExtent;
        const char* entityName = asset->getName(entity);
        const bool isStudio = (entityName != nullptr && std::string(entityName) == "Cyclorama") ||
                std::max({extent.x, extent.y, extent.z}) > 5.0f;
        renderables.setCastShadows(renderable, !isStudio);
        renderables.setReceiveShadows(renderable, true);
        const auto transform = transforms.getWorldTransform(transforms.getInstance(entity));
        const auto transformed = filament::Aabb{localBox.getMin(), localBox.getMax()}.transform(transform);
        addPoint(&worldBounds, transformed.min);
        addPoint(&worldBounds, transformed.max);
    }
    for (size_t index = 0; index < asset->getLightEntityCount(); ++index) {
        const auto entity = asset->getLightEntities()[index];
        const auto light = lights.getInstance(entity);
        lights.setIntensityCandela(
                light,
                lights.getIntensity(light) *
                        envFloat("MAESTRO_FILAMENT_ASSET_LIGHT_SCALE", 1.0f));
        lights.setFalloff(
                light, envFloat("MAESTRO_FILAMENT_ASSET_LIGHT_FALLOFF", 20.0f));
        lights.setShadowCaster(light, true);
        filament::LightManager::ShadowOptions options;
        options.mapSize = 2048;
        options.shadowBulbRadius = 1.2f;
        options.screenSpaceContactShadows = true;
        lights.setShadowOptions(light, options);
        const auto lightPosition = lights.getPosition(light);
        const auto lightDirection = lights.getDirection(light);
        std::cout << "asset_light[" << index << "]_name="
                  << (asset->getName(entity) == nullptr ? "" : asset->getName(entity))
                  << " type=" << static_cast<int>(lights.getType(light))
                  << " intensity=" << lights.getIntensity(light)
                  << " falloff=" << lights.getFalloff(light)
                  << " position=[" << lightPosition.x << "," << lightPosition.y << ","
                  << lightPosition.z << "] direction=[" << lightDirection.x << ","
                  << lightDirection.y << "," << lightDirection.z << "]"
                  << " inner_cone=" << lights.getSpotLightInnerCone(light)
                  << " outer_cone=" << lights.getSpotLightOuterCone(light) << "\n";
    }
    if (worldBounds.isEmpty()) {
        throw std::runtime_error("asset has no non-empty world-space renderable bounds");
    }

    CameraValues values;
    bool usesAssetCamera = false;
    if (asset->getCameraEntityCount() > 0) {
        auto* assetCamera = engine->getCameraComponent(asset->getCameraEntities()[0]);
        if (assetCamera != nullptr) {
            const auto position = assetCamera->getPosition();
            const auto forward = assetCamera->getForwardVector();
            values.eyeX = position.x;
            values.eyeY = position.y;
            values.eyeZ = position.z;
            values.targetX = position.x + forward.x;
            values.targetY = position.y + forward.y;
            values.targetZ = position.z + forward.z;
            values.distance = 1.0;
            view->setCamera(assetCamera);
            usesAssetCamera = true;
        }
    }
    if (!usesAssetCamera) {
        const auto center = worldBounds.center();
        const auto extent = worldBounds.extent();
        const double halfVerticalFovRadians = 21.0 * 3.14159265358979323846 / 180.0;
        const double horizontalHalfExtent =
                std::max(static_cast<double>(extent.x), static_cast<double>(extent.z));
        const double verticalDistance =
                static_cast<double>(extent.y) / std::tan(halfVerticalFovRadians);
        const double horizontalDistance =
                horizontalHalfExtent / std::tan(halfVerticalFovRadians);
        const double distance =
                std::max({verticalDistance, horizontalDistance, 0.5}) * 1.35 + 0.35;
        values.targetX = center.x;
        values.targetY = center.y;
        values.targetZ = center.z;
        values.distance = distance;
        values.eyeX = center.x + distance * 0.35;
        values.eyeY = center.y + distance * 0.28;
        values.eyeZ = center.z + distance * 1.05;
        camera->setProjection(42.0, 1.0, 0.05, 100.0, filament::Camera::Fov::VERTICAL);
        camera->lookAt({values.eyeX, values.eyeY, values.eyeZ},
                {values.targetX, values.targetY, values.targetZ});
        view->setCamera(camera);
    }

    std::cout << "asset_entities=" << asset->getEntityCount()
              << " renderables=" << asset->getRenderableEntityCount()
              << " world_aabb_min=[" << worldBounds.min.x << "," << worldBounds.min.y << ","
              << worldBounds.min.z << "] world_aabb_max=[" << worldBounds.max.x << ","
              << worldBounds.max.y << "," << worldBounds.max.z << "]\n"
              << "camera_eye=[" << values.eyeX << "," << values.eyeY << "," << values.eyeZ
              << "] target=[" << values.targetX << "," << values.targetY << "," << values.targetZ
              << "] distance=" << values.distance
              << " source=" << (usesAssetCamera ? "asset" : "bounds") << "\n";
    return {
        asset->getEntityCount(),
        asset->getRenderableEntityCount(),
        worldBounds,
        values,
        usesAssetCamera,
    };
}

filament::gltfio::FilamentAsset* loadAsset(filament::Engine* engine,
        filament::gltfio::AssetLoader* loader, const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("unable to read " + path);
    }
    const auto byteCount = static_cast<size_t>(input.tellg());
    std::vector<uint8_t> bytes(byteCount);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    auto* asset = loader->createAsset(bytes.data(), static_cast<uint32_t>(bytes.size()));
    if (!asset) {
        throw std::runtime_error("unable to load " + path);
    }
    filament::gltfio::ResourceLoader resourceLoader({engine, path.c_str(), true});
    if (!resourceLoader.loadResources(asset)) {
        loader->destroyAsset(asset);
        throw std::runtime_error("unable to upload resources for " + path);
    }
    asset->releaseSourceData();
    return asset;
}

double fps(uint32_t frames, double seconds) {
    return seconds > 0.0 ? static_cast<double>(frames) / seconds : 0.0;
}

void writeBounds(std::ofstream& output, const filament::Aabb& bounds) {
    output << "{\"min\":[" << bounds.min.x << "," << bounds.min.y << "," << bounds.min.z
           << "],\"max\":[" << bounds.max.x << "," << bounds.max.y << "," << bounds.max.z << "]}";
}

void writeCamera(std::ofstream& output, const CameraValues& camera) {
    output << "{\"eye\":[" << camera.eyeX << "," << camera.eyeY << "," << camera.eyeZ
           << "],\"target\":[" << camera.targetX << "," << camera.targetY << ","
           << camera.targetZ << "],\"vertical_fov_degrees\":42,\"distance\":" << camera.distance << "}";
}

void writePixelStats(std::ofstream& output, const PixelStats& stats) {
    output << "{\"hash_fnv1a64\":\"" << stats.hash << "\",\"background_rgb\":["
           << static_cast<int>(stats.background[0]) << "," << static_cast<int>(stats.background[1])
           << "," << static_cast<int>(stats.background[2]) << "],\"foreground_pixels\":"
           << stats.foregroundPixels << ",\"foreground_fraction\":" << stats.foregroundFraction
           << ",\"foreground_bbox\":[" << stats.minX << "," << stats.minY << ","
           << stats.maxX << "," << stats.maxY << "]}";
}

void writeMetrics(const std::string& path, const AssetInfo& staticInfo,
        const AssetInfo& animatedInfo, const PixelStats& staticPixels,
        const PixelStats& animatedPixels, const PhaseMetrics& staticMetrics,
        const PhaseMetrics& animatedMetrics, const std::vector<uint64_t>& animatedHashes,
        size_t animationCount, float animationDuration) {
    std::ofstream output(path);
    output << std::fixed << std::setprecision(6);
    output << "{\n"
           << "  \"filament_version\":\"v1.75.0\",\n"
           << "  \"backend\":\"Vulkan\",\n"
           << "  \"dimensions\":{\"width\":" << kWidth << ",\"height\":" << kHeight << "},\n"
           << "  \"visibility_validation\":{\"minimum_foreground_fraction\":0.002,"
           << "\"static\":";
    writePixelStats(output, staticPixels);
    output << ",\"animated\":";
    writePixelStats(output, animatedPixels);
    output << "},\n"
           << "  \"static_asset\":{\"entities\":" << staticInfo.entityCount
           << ",\"renderables\":" << staticInfo.renderableCount << ",\"world_aabb\":";
    writeBounds(output, staticInfo.worldBounds);
    output << ",\"camera\":";
    writeCamera(output, staticInfo.camera);
    output << ",\"camera_source\":\""
           << (staticInfo.usesAssetCamera ? "asset" : "bounds") << "\"},\n"
           << "  \"animated_asset\":{\"entities\":" << animatedInfo.entityCount
           << ",\"renderables\":" << animatedInfo.renderableCount << ",\"world_aabb\":";
    writeBounds(output, animatedInfo.worldBounds);
    output << ",\"camera\":";
    writeCamera(output, animatedInfo.camera);
    output << ",\"camera_source\":\""
           << (animatedInfo.usesAssetCamera ? "asset" : "bounds")
           << "\",\"animation_count\":" << animationCount << ",\"animation_duration_seconds\":"
           << animationDuration << "},\n"
           << "  \"static_warm_readback\":{\"warmup_frames\":" << kWarmupFrames
           << ",\"benchmark_frames\":" << kStaticFrames
           << ",\"render_submission_seconds\":" << staticMetrics.submissionSeconds
           << ",\"gpu_completion_readback_seconds\":" << staticMetrics.completionSeconds
           << ",\"total_seconds\":" << staticMetrics.totalSeconds
           << ",\"fps\":" << fps(kStaticFrames, staticMetrics.totalSeconds) << "},\n"
           << "  \"animated_warm_readback\":{\"warmup_frames\":" << kWarmupFrames
           << ",\"benchmark_frames\":" << kAnimatedFrames
           << ",\"animation_update_seconds\":" << animatedMetrics.animationUpdateSeconds
           << ",\"render_submission_seconds\":" << animatedMetrics.submissionSeconds
           << ",\"gpu_completion_readback_seconds\":" << animatedMetrics.completionSeconds
           << ",\"total_seconds\":" << animatedMetrics.totalSeconds
           << ",\"fps\":" << fps(kAnimatedFrames, animatedMetrics.totalSeconds)
           << ",\"sampled_pixel_hashes\":[";
    for (size_t index = 0; index < animatedHashes.size(); ++index) {
        if (index) {
            output << ",";
        }
        output << "\"" << animatedHashes[index] << "\"";
    }
    output << "]},\n"
           << "  \"gate\":{\"required_fps\":75.0,\"animated_fps\":"
           << fps(kAnimatedFrames, animatedMetrics.totalSeconds)
           << ",\"passed\":" << (fps(kAnimatedFrames, animatedMetrics.totalSeconds) >= 75.0 ? "true" : "false")
           << "}\n}\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: filament_bench <visible-static.glb> <maestro-animated.glb> <output-directory>\n";
        return 2;
    }
    try {
        const std::string staticPath = argv[1];
        const std::string animatedPath = argv[2];
        const std::string outputDirectory = argv[3];
        auto* engine = filament::Engine::create(filament::Engine::Backend::VULKAN);
        if (!engine) {
            throw std::runtime_error("unable to create a Vulkan Filament engine");
        }
        auto* materialProvider = filament::gltfio::createUbershaderProvider(
                engine, UBERARCHIVE_DEFAULT_DATA, UBERARCHIVE_DEFAULT_SIZE);
        auto* assetLoader = filament::gltfio::AssetLoader::create({engine, materialProvider});
        auto* renderer = engine->createRenderer();
        auto* scene = engine->createScene();
        auto* view = engine->createView();
        auto& transformManager = engine->getTransformManager();
        const auto cameraEntity = utils::EntityManager::get().create();
        auto* camera = engine->createCamera(cameraEntity);
        auto* colorTexture = filament::Texture::Builder()
                .width(kWidth).height(kHeight).levels(1)
                .sampler(filament::Texture::Sampler::SAMPLER_2D)
                .format(filament::Texture::InternalFormat::RGBA8)
                .usage(filament::Texture::Usage::COLOR_ATTACHMENT |
                        filament::Texture::Usage::BLIT_SRC)
                .build(*engine);
        auto* depthTexture = filament::Texture::Builder()
                .width(kWidth).height(kHeight).levels(1)
                .sampler(filament::Texture::Sampler::SAMPLER_2D)
                .format(filament::Texture::InternalFormat::DEPTH24)
                .usage(filament::Texture::Usage::DEPTH_ATTACHMENT)
                .build(*engine);
        auto* renderTarget = filament::RenderTarget::Builder()
                .texture(filament::RenderTarget::AttachmentPoint::COLOR, colorTexture)
                .texture(filament::RenderTarget::AttachmentPoint::DEPTH, depthTexture)
                .build(*engine);
        const float worldLinear =
                envFloat("MAESTRO_FILAMENT_WORLD_LINEAR", 0.10f);
        filament::Renderer::ClearOptions clearOptions;
        clearOptions.clearColor = {
            worldLinear, worldLinear, worldLinear * 1.1f, 1.0f,
        };
        clearOptions.clear = true;
        renderer->setClearOptions(clearOptions);
        view->setScene(scene);
        view->setViewport({0, 0, kWidth, kHeight});
        view->setRenderTarget(renderTarget);
        view->setPostProcessingEnabled(true);
        view->setFrustumCullingEnabled(false);
        view->setAntiAliasing(filament::AntiAliasing::FXAA);
        view->setDithering(filament::Dithering::NONE);
        filament::MultiSampleAntiAliasingOptions msaa;
        msaa.enabled = true;
        msaa.sampleCount = 4;
        msaa.customResolve = true;
        view->setMultiSampleAntiAliasingOptions(msaa);
        filament::AmbientOcclusionOptions ao;
        ao.aoType = filament::AmbientOcclusionOptions::AmbientOcclusionType::GTAO;
        ao.radius = 0.3f;
        ao.power = 1.0f;
        ao.resolution = 1.0f;
        ao.intensity = 1.0f;
        ao.quality = filament::QualityLevel::HIGH;
        ao.lowPassFilter = filament::QualityLevel::HIGH;
        ao.enabled = envUint("MAESTRO_FILAMENT_AO_ENABLED", 1) != 0;
        view->setAmbientOcclusionOptions(ao);
        const char* shadowTypeValue =
                std::getenv("MAESTRO_FILAMENT_SHADOW_TYPE");
        const std::string shadowType =
                shadowTypeValue == nullptr ? "pcss" : shadowTypeValue;
        if (shadowType == "pcf") {
            view->setShadowType(filament::ShadowType::PCF);
        } else if (shadowType == "vsm") {
            view->setShadowType(filament::ShadowType::VSM);
        } else if (shadowType == "pcss") {
            view->setShadowType(filament::ShadowType::PCSS);
        } else {
            throw std::runtime_error(
                    "unsupported shadow type " + shadowType);
        }

        filament::AgxToneMapper toneMapper(
                filament::AgxToneMapper::AgxLook::PUNCHY);
        auto* colorGrading = filament::ColorGrading::Builder()
                .toneMapper(&toneMapper)
                .luminanceScaling(true)
                .gamutMapping(true)
                .build(*engine);
        view->setColorGrading(colorGrading);

        const char* iblEnvironment = std::getenv("MAESTRO_FILAMENT_IBL");
        const std::string iblPath = iblEnvironment == nullptr
                ? "/workspace/maestro-filament-poc/filament/bin/assets/ibl/"
                  "lightroom_14b/lightroom_14b_ibl.ktx"
                : iblEnvironment;
        auto iblBytes = readFile(iblPath);
        image::Ktx1Bundle iblBundle(
                iblBytes.data(), static_cast<uint32_t>(iblBytes.size()));
        std::array<filament::math::float3, 9> sphericalHarmonics;
        if (!iblBundle.getSphericalHarmonics(sphericalHarmonics.data())) {
            throw std::runtime_error("IBL has no spherical harmonics metadata: " + iblPath);
        }
        auto* iblTexture = ktxreader::Ktx1Reader::createTexture(
                engine, iblBundle, false, nullptr, nullptr);
        if (iblTexture == nullptr) {
            throw std::runtime_error("unable to create IBL texture from " + iblPath);
        }
        auto* indirectLight = filament::IndirectLight::Builder()
                .reflections(iblTexture)
                .irradiance(3, sphericalHarmonics.data())
                .intensity(envFloat("MAESTRO_FILAMENT_INDIRECT_LUX", 10000.0f))
                .build(*engine);
        scene->setIndirectLight(indirectLight);

        const auto keyLight = utils::EntityManager::get().create();
        filament::LightManager::Builder(filament::LightManager::Type::DIRECTIONAL)
                .color({1.0f, 0.96f, 0.90f})
                .intensity(envFloat("MAESTRO_FILAMENT_KEY_LUX", 35000.0f))
                .direction({0.25f, -0.8f, -0.55f}).castShadows(false)
                .build(*engine, keyLight);
        const auto fillLight = utils::EntityManager::get().create();
        filament::LightManager::Builder(filament::LightManager::Type::DIRECTIONAL)
                .color({1.0f, 1.0f, 1.0f})
                .intensity(envFloat("MAESTRO_FILAMENT_FILL_LUX", 15000.0f))
                .direction({0.485f, -0.485f, -0.728f}).castShadows(false)
                .build(*engine, fillLight);
        scene->addEntity(keyLight);
        scene->addEntity(fillLight);

        std::vector<uint8_t> pixels(
                static_cast<size_t>(kWidth) * kHeight * kPixelChannels);
        const bool jobOnly =
                envUint("MAESTRO_FILAMENT_JOB_ONLY", 0) != 0;
        auto* staticAsset = loadAsset(engine, assetLoader, staticPath);
        const auto staticInfo = configureAsset(engine, scene, view, camera, staticAsset);
        for (uint32_t frame = 0; frame < kWarmupFrames; ++frame) {
            renderer->renderStandaloneView(view);
        }
        engine->flushAndWait();
        PhaseMetrics preflight;
        if (!renderAndReadback(renderer, renderTarget, view, engine, pixels, &preflight)) {
            throw std::runtime_error("static preflight readback failed");
        }
        const auto staticPixels = inspectPixels(pixels);
        if (!visiblyRendered(staticPixels, "static")) {
            throw std::runtime_error("static capture was blank or only background");
        }
        writePpm(outputDirectory + "/filament_static_visible.ppm", pixels);

        PhaseMetrics staticMetrics;
        if (!jobOnly) {
            const auto staticStart = Clock::now();
            for (uint32_t frame = 0; frame < kStaticFrames; ++frame) {
                if (!renderAndReadback(
                            renderer, renderTarget, view, engine, pixels,
                            &staticMetrics)) {
                    throw std::runtime_error(
                            "static benchmark readback failed");
                }
            }
            staticMetrics.totalSeconds =
                    secondsBetween(staticStart, Clock::now());
        }
        scene->removeEntities(staticAsset->getEntities(), staticAsset->getEntityCount());
        assetLoader->destroyAsset(staticAsset);

        auto* animatedAsset = loadAsset(engine, assetLoader, animatedPath);
        const auto animatedInfo = configureAsset(engine, scene, view, camera, animatedAsset);
        auto* animator = animatedAsset->getInstance()->getAnimator();
        const size_t animationCount = animator ? animator->getAnimationCount() : 0;
        if (animationCount == 0) {
            throw std::runtime_error("animated GLB has no animation clip");
        }
        float animationDuration = 0.0f;
        for (size_t index = 0; index < animationCount; ++index) {
            const float duration = animator->getAnimationDuration(index);
            animationDuration = std::max(animationDuration, duration);
            std::cout << "animation[" << index << "]_name="
                      << animator->getAnimationName(index)
                      << " duration_seconds=" << duration << "\n";
        }
        if (animationDuration <= 0.0f) {
            throw std::runtime_error("animated GLB has a zero-duration animation clip");
        }
        const uint32_t frameOffset =
                envUint("MAESTRO_FILAMENT_FRAME_OFFSET", 0);
        auto applyAnimation = [&](uint32_t frame, PhaseMetrics* metrics) {
            const auto start = Clock::now();
            const float sourceTime = static_cast<float>(frame) / 30.0f;
            for (size_t index = 0; index < animationCount; ++index) {
                const float duration = animator->getAnimationDuration(index);
                const float loopDuration = duration + (1.0f / 30.0f);
                const float sampleTime = duration > 0.0001f
                        ? std::min(std::fmod(sourceTime, loopDuration), duration)
                        : 0.0f;
                animator->applyAnimation(index, sampleTime);
            }
            animator->updateBoneMatrices();
            metrics->animationUpdateSeconds += secondsBetween(start, Clock::now());
        };
        for (uint32_t frame = 0; frame < kWarmupFrames; ++frame) {
            PhaseMetrics ignored;
            applyAnimation(frame, &ignored);
            renderer->renderStandaloneView(view);
        }
        engine->flushAndWait();
        PhaseMetrics animationPreflight;
        // Capture MAESTRO source frame 0, corresponding to Blender's frame 1 reference.
        applyAnimation(0, &animationPreflight);
        if (!renderAndReadback(renderer, renderTarget, view, engine, pixels, &animationPreflight)) {
            throw std::runtime_error("animated preflight readback failed");
        }
        const auto animatedPixels = inspectPixels(pixels);
        if (!visiblyRendered(animatedPixels, "animated")) {
            throw std::runtime_error("animated capture was blank or only background");
        }
        writePpm(outputDirectory + "/filament_animated_visible.ppm", pixels);

        PhaseMetrics animatedMetrics;
        std::vector<uint64_t> animatedHashes;
        if (!jobOnly) {
            const auto animationStart = Clock::now();
            for (uint32_t frame = 0; frame < kAnimatedFrames; ++frame) {
                const uint32_t animationFrame = kWarmupFrames + frame;
                applyAnimation(animationFrame, &animatedMetrics);
                if (!renderAndReadback(
                            renderer, renderTarget, view, engine, pixels,
                            &animatedMetrics)) {
                    throw std::runtime_error(
                            "animated benchmark readback failed");
                }
                if (frame == 0 || frame == 75 || frame == 150 ||
                        frame == 225 || frame == kAnimatedFrames - 1) {
                    animatedHashes.push_back(inspectPixels(pixels).hash);
                }
            }
            animatedMetrics.totalSeconds =
                    secondsBetween(animationStart, Clock::now());
        }
        const std::set<uint64_t> uniqueHashes(animatedHashes.begin(), animatedHashes.end());
        if (!jobOnly && uniqueHashes.size() < 3) {
            throw std::runtime_error("animated frames did not produce at least three distinct pixel hashes");
        }
        const uint32_t longRunFrames =
                envUint("MAESTRO_FILAMENT_LONG_FRAMES", 0);
        if (longRunFrames > 0) {
            const char* outputValue = std::getenv("MAESTRO_FILAMENT_VIDEO_PATH");
            const std::string videoPath =
                    outputValue == nullptr ? "" : outputValue;
            const char* encoderValue = std::getenv("MAESTRO_FILAMENT_ENCODER");
            const std::string encoder =
                    encoderValue == nullptr ? "libx264" : encoderValue;
            FILE* encoderPipe = nullptr;
            if (!videoPath.empty()) {
                std::string encoderArguments;
                if (encoder == "libx264") {
                    encoderArguments =
                            "-vf format=rgb24 -c:v libx264 -preset veryfast "
                            "-pix_fmt yuv420p -movflags +faststart";
                } else if (encoder == "h264_nvenc") {
                    encoderArguments =
                            "-vf format=rgb24 -c:v h264_nvenc -preset p4 "
                            "-cq 23 -b:v 0 -pix_fmt yuv420p -movflags +faststart";
                } else {
                    throw std::runtime_error("unsupported encoder " + encoder);
                }
                const std::string command =
                        "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgba "
                        "-video_size 1080x1080 -framerate 30 -i pipe:0 " +
                        encoderArguments + " " + shellQuote(videoPath);
                encoderPipe = ::popen(command.c_str(), "w");
                if (encoderPipe == nullptr) {
                    throw std::runtime_error("unable to start ffmpeg encoder");
                }
            }
            PhaseMetrics longRunMetrics;
            double pipeWriteSeconds = 0.0;
            const auto longRunStart = Clock::now();
            for (uint32_t frame = 0; frame < longRunFrames; ++frame) {
                applyAnimation(frameOffset + frame, &longRunMetrics);
                if (!renderAndReadback(
                            renderer, renderTarget, view, engine, pixels,
                            &longRunMetrics)) {
                    throw std::runtime_error("long-run frame readback failed");
                }
                if (encoderPipe != nullptr) {
                    const auto writeStart = Clock::now();
                    const size_t written =
                            std::fwrite(pixels.data(), 1, pixels.size(), encoderPipe);
                    pipeWriteSeconds += secondsBetween(writeStart, Clock::now());
                    if (written != pixels.size()) {
                        ::pclose(encoderPipe);
                        throw std::runtime_error("ffmpeg raw-video pipe write failed");
                    }
                }
            }
            int encoderStatus = 0;
            if (encoderPipe != nullptr) {
                encoderStatus = ::pclose(encoderPipe);
                if (encoderStatus != 0) {
                    throw std::runtime_error(
                            "ffmpeg encoder exited with status " +
                            std::to_string(encoderStatus));
                }
            }
            longRunMetrics.totalSeconds =
                    secondsBetween(longRunStart, Clock::now());
            const uintmax_t videoBytes = videoPath.empty()
                    ? 0
                    : std::filesystem::file_size(videoPath);
            std::ofstream longRunReport(
                    outputDirectory + "/filament_long_run.json");
            longRunReport << std::fixed << std::setprecision(6)
                          << "{\"frames\":" << longRunFrames
                          << ",\"animation_update_seconds\":"
                          << longRunMetrics.animationUpdateSeconds
                          << ",\"render_submission_seconds\":"
                          << longRunMetrics.submissionSeconds
                          << ",\"gpu_completion_readback_seconds\":"
                          << longRunMetrics.completionSeconds
                          << ",\"encoder_pipe_write_seconds\":" << pipeWriteSeconds
                          << ",\"total_seconds\":" << longRunMetrics.totalSeconds
                          << ",\"fps\":"
                          << fps(longRunFrames, longRunMetrics.totalSeconds)
                          << ",\"encoder\":\"" << encoder << "\""
                          << ",\"video_path\":\"" << videoPath << "\""
                          << ",\"video_bytes\":" << videoBytes
                          << ",\"encoder_status\":" << encoderStatus << "}\n";
            std::cout << "long_run_frames=" << longRunFrames
                      << " total_seconds=" << longRunMetrics.totalSeconds
                      << " fps=" << fps(longRunFrames, longRunMetrics.totalSeconds)
                      << " encoder=" << encoder
                      << " video_bytes=" << videoBytes << "\n";
        }
        const uint32_t asyncRunFrames =
                envUint("MAESTRO_FILAMENT_ASYNC_FRAMES", 0);
        if (asyncRunFrames > 0) {
            struct AsyncReadbackSlot {
                ReadbackState state;
                uint32_t frame = 0;
                filament::Texture* color = nullptr;
                filament::Texture* depth = nullptr;
                filament::RenderTarget* renderTarget = nullptr;
                std::vector<uint8_t> pixels;
            };
            const uint32_t ringSize = std::max(
                    2u, envUint("MAESTRO_FILAMENT_ASYNC_RING", 8));
            const char* asyncOutputValue =
                    std::getenv("MAESTRO_FILAMENT_ASYNC_VIDEO_PATH");
            const std::string asyncVideoPath =
                    asyncOutputValue == nullptr ? "" : asyncOutputValue;
            const char* asyncEncoderValue =
                    std::getenv("MAESTRO_FILAMENT_ASYNC_ENCODER");
            const std::string asyncEncoder =
                    asyncEncoderValue == nullptr ? "h264_nvenc" : asyncEncoderValue;
            FILE* asyncEncoderPipe = nullptr;
            if (!asyncVideoPath.empty()) {
                std::string encoderArguments;
                if (asyncEncoder == "libx264") {
                    encoderArguments =
                            "-vf format=rgb24 -c:v libx264 -preset veryfast "
                            "-pix_fmt yuv420p -movflags +faststart";
                } else if (asyncEncoder == "h264_nvenc") {
                    encoderArguments =
                            "-vf format=rgb24 -c:v h264_nvenc -preset p4 "
                            "-cq 23 -b:v 0 -pix_fmt yuv420p -movflags +faststart";
                } else {
                    throw std::runtime_error(
                            "unsupported async encoder " + asyncEncoder);
                }
                const std::string command =
                        "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgba "
                        "-video_size 1080x1080 -framerate 30 -i pipe:0 " +
                        encoderArguments + " " + shellQuote(asyncVideoPath);
                asyncEncoderPipe = ::popen(command.c_str(), "w");
                if (asyncEncoderPipe == nullptr) {
                    throw std::runtime_error(
                            "unable to start asynchronous ffmpeg encoder");
                }
            }
            std::vector<std::unique_ptr<AsyncReadbackSlot>> slots;
            slots.reserve(ringSize);
            for (uint32_t index = 0; index < ringSize; ++index) {
                auto slot = std::make_unique<AsyncReadbackSlot>();
                slot->color = filament::Texture::Builder()
                        .width(kWidth).height(kHeight).levels(1)
                        .sampler(filament::Texture::Sampler::SAMPLER_2D)
                        .format(filament::Texture::InternalFormat::RGBA8)
                        .usage(filament::Texture::Usage::COLOR_ATTACHMENT |
                                filament::Texture::Usage::BLIT_SRC)
                        .build(*engine);
                slot->depth = filament::Texture::Builder()
                        .width(kWidth).height(kHeight).levels(1)
                        .sampler(filament::Texture::Sampler::SAMPLER_2D)
                        .format(filament::Texture::InternalFormat::DEPTH24)
                        .usage(filament::Texture::Usage::DEPTH_ATTACHMENT)
                        .build(*engine);
                slot->renderTarget = filament::RenderTarget::Builder()
                        .texture(
                                filament::RenderTarget::AttachmentPoint::COLOR,
                                slot->color)
                        .texture(
                                filament::RenderTarget::AttachmentPoint::DEPTH,
                                slot->depth)
                        .build(*engine);
                slot->pixels.resize(
                        static_cast<size_t>(kWidth) * kHeight * kPixelChannels);
                slots.push_back(std::move(slot));
            }
            const std::set<uint32_t> sampleFrames = {
                0,
                asyncRunFrames / 4,
                asyncRunFrames / 2,
                (3 * asyncRunFrames) / 4,
                asyncRunFrames - 1,
            };
            const bool writeAsyncSamples =
                    envUint("MAESTRO_FILAMENT_WRITE_ASYNC_SAMPLES", 0) != 0;
            std::vector<uint64_t> asyncHashes;
            double asyncPipeWriteSeconds = 0.0;
            uint32_t submitted = 0;
            uint32_t completed = 0;
            PhaseMetrics asyncMetrics;
            auto drainCompleted = [&]() {
                engine->pumpMessageQueues();
                bool madeProgress = false;
                while (completed < submitted) {
                    auto& slot = *slots[completed % ringSize];
                    if (!slot.state.complete.load(std::memory_order_acquire)) {
                        break;
                    }
                    if (sampleFrames.count(slot.frame) != 0) {
                        asyncHashes.push_back(inspectPixels(slot.pixels).hash);
                        if (writeAsyncSamples) {
                            writePpm(
                                    outputDirectory + "/async_sample_" +
                                            std::to_string(slot.frame) + ".ppm",
                                    slot.pixels);
                        }
                    }
                    if (asyncEncoderPipe != nullptr) {
                        const auto writeStart = Clock::now();
                        const size_t written = std::fwrite(
                                slot.pixels.data(), 1, slot.pixels.size(),
                                asyncEncoderPipe);
                        asyncPipeWriteSeconds +=
                                secondsBetween(writeStart, Clock::now());
                        if (written != slot.pixels.size()) {
                            throw std::runtime_error(
                                    "asynchronous ffmpeg pipe write failed");
                        }
                    }
                    ++completed;
                    madeProgress = true;
                }
                return madeProgress;
            };

            const auto asyncStart = Clock::now();
            while (submitted < asyncRunFrames) {
                while (submitted - completed >= ringSize) {
                    if (!drainCompleted()) {
                        engine->flush();
                        std::this_thread::yield();
                    }
                }
                auto& slot = *slots[submitted % ringSize];
                slot.frame = submitted;
                slot.state.complete.store(false, std::memory_order_release);
                applyAnimation(frameOffset + submitted, &asyncMetrics);
                const auto submissionStart = Clock::now();
                view->setRenderTarget(slot.renderTarget);
                renderer->renderStandaloneView(view);
                renderer->readPixels(
                        slot.renderTarget, 0, 0, kWidth, kHeight,
                        filament::backend::PixelBufferDescriptor(
                                slot.pixels.data(), slot.pixels.size(),
                                filament::backend::PixelDataFormat::RGBA,
                                filament::backend::PixelDataType::UBYTE,
                                onReadback, &slot.state));
                asyncMetrics.submissionSeconds +=
                        secondsBetween(submissionStart, Clock::now());
                ++submitted;
                engine->flush();
                drainCompleted();
            }
            while (completed < submitted) {
                if (!drainCompleted()) {
                    engine->flush();
                    std::this_thread::yield();
                }
            }
            engine->flushAndWait();
            engine->pumpMessageQueues();
            view->setRenderTarget(renderTarget);
            int asyncEncoderStatus = 0;
            if (asyncEncoderPipe != nullptr) {
                asyncEncoderStatus = ::pclose(asyncEncoderPipe);
                if (asyncEncoderStatus != 0) {
                    throw std::runtime_error(
                            "asynchronous ffmpeg encoder exited with status " +
                            std::to_string(asyncEncoderStatus));
                }
            }
            asyncMetrics.totalSeconds = secondsBetween(asyncStart, Clock::now());
            const uintmax_t asyncVideoBytes = asyncVideoPath.empty()
                    ? 0
                    : std::filesystem::file_size(asyncVideoPath);
            if (asyncHashes.size() != sampleFrames.size()) {
                throw std::runtime_error(
                        "async readback did not retain all sampled frames");
            }
            std::ofstream asyncReport(
                    outputDirectory + "/filament_async_run.json");
            asyncReport << std::fixed << std::setprecision(6)
                        << "{\"frames\":" << asyncRunFrames
                        << ",\"ring_size\":" << ringSize
                        << ",\"animation_update_seconds\":"
                        << asyncMetrics.animationUpdateSeconds
                        << ",\"submission_seconds\":"
                        << asyncMetrics.submissionSeconds
                        << ",\"encoder_pipe_write_seconds\":"
                        << asyncPipeWriteSeconds
                        << ",\"total_seconds\":" << asyncMetrics.totalSeconds
                        << ",\"fps\":"
                        << fps(asyncRunFrames, asyncMetrics.totalSeconds)
                        << ",\"encoder\":\"" << asyncEncoder << "\""
                        << ",\"video_path\":\"" << asyncVideoPath << "\""
                        << ",\"video_bytes\":" << asyncVideoBytes
                        << ",\"encoder_status\":" << asyncEncoderStatus
                        << ",\"sampled_pixel_hashes\":[";
            for (size_t index = 0; index < asyncHashes.size(); ++index) {
                if (index) {
                    asyncReport << ",";
                }
                asyncReport << "\"" << asyncHashes[index] << "\"";
            }
            asyncReport << "]}\n";
            std::cout << "async_run_frames=" << asyncRunFrames
                      << " ring_size=" << ringSize
                      << " total_seconds=" << asyncMetrics.totalSeconds
                      << " fps=" << fps(asyncRunFrames, asyncMetrics.totalSeconds)
                      << " encoder=" << asyncEncoder
                      << " video_bytes=" << asyncVideoBytes
                      << "\n";
            for (auto& slot : slots) {
                engine->destroy(slot->renderTarget);
                engine->destroy(slot->depth);
                engine->destroy(slot->color);
            }
        }
        auto* animatedCamera = engine->getCameraComponent(animatedAsset->getCameraEntities()[0]);
        if (animatedCamera == nullptr) {
            throw std::runtime_error("animated asset camera is unavailable");
        }
        const std::array<uint32_t, 5> metricFrames = {0, 75, 150, 225, 299};
        std::ofstream rigMetrics(outputDirectory + "/filament_rig_metrics.json");
        rigMetrics << std::fixed << std::setprecision(9)
                   << "{\"coordinate_system\":\"gltf_y_up\","
                   << "\"projection_origin\":\"bottom_left\",\"frames\":[";
        for (size_t frameIndex = 0; frameIndex < metricFrames.size(); ++frameIndex) {
            PhaseMetrics ignored;
            const uint32_t frame = metricFrames[frameIndex];
            applyAnimation(frame, &ignored);
            const auto viewMatrix = animatedCamera->getViewMatrix();
            const auto projectionMatrix = animatedCamera->getProjectionMatrix();
            if (frameIndex) {
                rigMetrics << ",";
            }
            rigMetrics << "{\"frame\":" << frame << ",\"joints\":[";
            for (size_t jointIndex = 0; jointIndex < kJointNames.size(); ++jointIndex) {
                const auto jointEntity =
                        animatedAsset->getFirstEntityByName(kJointNames[jointIndex]);
                const auto jointTransform = transformManager.getInstance(jointEntity);
                if (!jointEntity || !jointTransform) {
                    throw std::runtime_error(
                            std::string("animated asset is missing joint ") +
                            kJointNames[jointIndex]);
                }
                const auto worldMatrix =
                        transformManager.getWorldTransformAccurate(jointTransform);
                const auto world = worldMatrix[3].xyz;
                const auto clip = projectionMatrix * viewMatrix *
                        filament::math::float4{world, 1.0f};
                if (std::abs(clip.w) < 1.0e-8) {
                    throw std::runtime_error(
                            std::string("joint projected to zero w: ") +
                            kJointNames[jointIndex]);
                }
                const auto ndc = clip.xyz / clip.w;
                if (jointIndex) {
                    rigMetrics << ",";
                }
                rigMetrics << "{\"name\":\"" << kJointNames[jointIndex] << "\","
                           << "\"world\":[" << world.x << "," << world.y << ","
                           << world.z << "],\"projected\":["
                           << 0.5f * (ndc.x + 1.0f) << ","
                           << 0.5f * (ndc.y + 1.0f) << "," << ndc.z << "]}";
            }
            rigMetrics << "]}";
        }
        rigMetrics << "]}\n";
        writeMetrics(outputDirectory + "/filament_metrics.json", staticInfo, animatedInfo,
                staticPixels, animatedPixels, staticMetrics, animatedMetrics, animatedHashes,
                animationCount, animationDuration);
        if (!jobOnly) {
            std::cout << std::fixed << std::setprecision(3)
                      << "static_readback_fps="
                      << fps(kStaticFrames, staticMetrics.totalSeconds) << "\n"
                      << "animated_readback_fps="
                      << fps(kAnimatedFrames, animatedMetrics.totalSeconds) << "\n"
                      << "animated_hashes_unique=" << uniqueHashes.size() << "\n";
        }

        scene->removeEntities(animatedAsset->getEntities(), animatedAsset->getEntityCount());
        assetLoader->destroyAsset(animatedAsset);
        scene->removeEntities(&keyLight, 1);
        scene->removeEntities(&fillLight, 1);
        engine->destroy(keyLight);
        engine->destroy(fillLight);
        scene->setIndirectLight(nullptr);
        engine->destroy(indirectLight);
        engine->destroy(iblTexture);
        engine->destroy(colorGrading);
        engine->destroy(view);
        engine->destroy(scene);
        engine->destroy(renderer);
        engine->destroy(renderTarget);
        engine->destroy(depthTexture);
        engine->destroy(colorTexture);
        engine->destroyCameraComponent(cameraEntity);
        utils::EntityManager::get().destroy(cameraEntity);
        filament::gltfio::AssetLoader::destroy(&assetLoader);
        materialProvider->destroyMaterials();
        delete materialProvider;
        filament::Engine::destroy(&engine);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
