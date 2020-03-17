/*
  Copyright 2019-2020 David Robillard <http://drobilla.net>

  Permission to use, copy, modify, and/or distribute this software for any
  purpose with or without fee is hereby granted, provided that the above
  copyright notice and this permission notice appear in all copies.

  THIS SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/

/* Define default dispatcher to nonsense so that a missing dispatch parameter
   will be a hard error and we won't link to the static one. */
#ifdef PUGL_VK_DYNAMIC_DISPATCH
#	define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#	define VULKAN_HPP_DEFAULT_DISPATCHER (*(vk::DispatchLoaderDynamic*)nullptr)
#else
#	define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 0
#	define VULKAN_HPP_DEFAULT_DISPATCHER (*(vk::DispatchLoaderStatic*)nullptr)
#endif

#include "demo_utils.h"
#include "rects.h"
#include "test/test_utils.h"

#include "pugl/pugl.hpp"
#include "pugl/pugl_vulkan.hpp"

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

#ifdef PUGL_VK_DYNAMIC_DISPATCH
using Dispatch = vk::DispatchLoaderDynamic;
#else
using Dispatch = vk::DispatchLoaderStatic;
#endif

constexpr const uint32_t defaultWidth  = 512;
constexpr const uint32_t defaultHeight = 512;

struct UniformBufferObject {
	mat4 projection;
};

static vk::VertexInputBindingDescription
getModelBindingDescription()
{
	return vk::VertexInputBindingDescription{0,
	                                         sizeof(vec2),
	                                         vk::VertexInputRate::eVertex};
}

static std::array<vk::VertexInputAttributeDescription, 1>
getModelAttributeDescriptions()
{
	return {vk::VertexInputAttributeDescription{
	    0u, 0u, vk::Format::eR32G32Sfloat, 0}};
}

static vk::VertexInputBindingDescription
getRectBindingDescription()
{
	return vk::VertexInputBindingDescription{1,
	                                         sizeof(Rect),
	                                         vk::VertexInputRate::eInstance};
}

static std::array<vk::VertexInputAttributeDescription, 3>
getRectAttributeDescriptions()
{
	return {
	    {{1, 1, vk::Format::eR32G32Sfloat, offsetof(Rect, pos)},
	     {2, 1, vk::Format::eR32G32Sfloat, offsetof(Rect, size)},
	     {3, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Rect, fillColor)}}};
}

static std::vector<Rect>
makeRects(const size_t numRects)
{
	std::vector<Rect> rects(numRects);
	for (size_t i = 0; i < numRects; ++i) {
		rects[i] = makeRect(i, static_cast<float>(defaultWidth));
	}

	return rects;
}

std::vector<Rect> rects = makeRects(1000);

//////////////////////////

/// Helper macro for counted array arguments to make clang-format behave
#define COUNTED(count, ...) count, __VA_ARGS__

/// Vulkan physical device selection
struct PhysicalDeviceSelection {
	vk::PhysicalDevice physicalDevice;
	uint32_t           graphicsFamilyIndex;
};

/// Vulkan device, all the graphical state that depends on a physical device
struct VulkanDevice {
	VulkanDevice(Dispatch&               d,
	             vk::UniqueSurfaceKHR    s,
	             PhysicalDeviceSelection p);

	Dispatch&             dispatch;
	vk::UniqueSurfaceKHR  surface;
	vk::PhysicalDevice    physicalDevice;
	uint32_t              graphicsIndex;
	vk::SurfaceFormatKHR  surfaceFormat;
	vk::PresentModeKHR    presentMode;
	vk::UniqueDevice      device;
	vk::Queue             graphicsQueue;
	vk::UniqueCommandPool commandPool;
	vk::UniqueSemaphore   presentComplete;
	vk::UniqueSemaphore   renderFinished;
};

struct Buffer {
	Buffer(VulkanDevice&                 gpu,
	       const vk::DeviceSize          size,
	       const vk::BufferUsageFlags    usage,
	       const vk::MemoryPropertyFlags properties);

	vk::UniqueBuffer       buffer;
	vk::UniqueDeviceMemory deviceMemory;
};

struct Swapchain {
	Swapchain(VulkanDevice& gpu, uint32_t width, uint32_t height);

	vk::SurfaceCapabilitiesKHR           capabilities;
	vk::Extent2D                         extent;
	vk::UniqueSwapchainKHR               rawSwapchain;
	std::vector<vk::Image>               images;
	std::vector<vk::UniqueImageView>     imageViews;
	std::vector<vk::UniqueCommandBuffer> commandBuffers;
	std::vector<vk::UniqueFence>         fences;
	size_t                               currentFrame{};
};

struct RenderPass {
	RenderPass(VulkanDevice&                 gpu,
	           Swapchain&                    swapchain,
	           vk::UniqueDescriptorSetLayout descriptorSetLayout);

	vk::UniqueRenderPass                 renderPass;
	vk::UniqueDescriptorSetLayout        layout;
	vk::UniqueDescriptorPool             descriptorPool;
	std::vector<vk::UniqueFramebuffer>   frameBuffers;
	std::vector<vk::UniqueDescriptorSet> descriptorSets;
	std::vector<Buffer>                  uniformBuffers;
};

struct Pipeline {
	Pipeline(VulkanDevice&                  gpu,
	         const vk::DescriptorSetLayout& descriptorSetLayout,
	         const vk::RenderPass&          renderPass,
	         const vk::Extent2D             extent);

	vk::UniquePipelineLayout layout;
	vk::UniquePipeline       pipeline;
};

struct Renderer {
	Renderer(VulkanDevice& gpu, uint32_t width, uint32_t height);

	static vk::UniqueDescriptorSetLayout
	createDescriptorSetLayout(VulkanDevice& gpu);

	Swapchain  swapchain;
	RenderPass renderPass;
	Pipeline   pipeline;
	Buffer     modelBuffer;
	Buffer     vertexBuffer;
	Buffer     stagingBuffer;
};

static void
copyBuffer(vk::Device&        device,
           vk::CommandPool&   commandPool,
           vk::Queue&         graphicsQueue,
           vk::Buffer         srcBuffer,
           vk::Buffer         dstBuffer,
           const VkDeviceSize size,
           const Dispatch&    dispatch)
{
	std::vector<vk::UniqueCommandBuffer> commandBuffers =
	    device.allocateCommandBuffersUnique(
	        {commandPool, vk::CommandBufferLevel::ePrimary, 1u}, dispatch);

	commandBuffers[0]->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit},
	                         dispatch);
	commandBuffers[0]->copyBuffer(srcBuffer,
	                              dstBuffer,
	                              vk::BufferCopy{0, 0, size},
	                              dispatch);
	commandBuffers[0]->end(dispatch);

	const vk::SubmitInfo submitInfo{COUNTED(0, nullptr),
	                                nullptr,
	                                COUNTED(1, &*commandBuffers[0]),
	                                COUNTED(0, nullptr)};

	graphicsQueue.submit(submitInfo, nullptr, dispatch);
	graphicsQueue.waitIdle(dispatch);
}

static uint32_t
findMemoryType(vk::PhysicalDevice&           physicalDevice,
               const uint32_t                typeFilter,
               const vk::MemoryPropertyFlags properties,
               Dispatch&                     dispatch)
{
	vk::PhysicalDeviceMemoryProperties memProperties;
	physicalDevice.getMemoryProperties(&memProperties, dispatch);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) &&
		    (memProperties.memoryTypes[i].propertyFlags & properties) ==
		        properties) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

Buffer::Buffer(VulkanDevice&                 gpu,
               const vk::DeviceSize          size,
               const vk::BufferUsageFlags    usage,
               const vk::MemoryPropertyFlags properties)
{
	const vk::BufferCreateInfo bufferInfo{{},
	                                      size,
	                                      usage,
	                                      vk::SharingMode::eExclusive};

	auto& device = gpu.device;

	buffer = device->createBufferUnique(bufferInfo, nullptr, gpu.dispatch);

	vk::MemoryRequirements memRequirements;
	device->getBufferMemoryRequirements(*buffer,
	                                    &memRequirements,
	                                    gpu.dispatch);

	const vk::MemoryAllocateInfo allocInfo{
	    memRequirements.size,
	    findMemoryType(gpu.physicalDevice,
	                   memRequirements.memoryTypeBits,
	                   properties,
	                   gpu.dispatch)};

	deviceMemory = device->allocateMemoryUnique(allocInfo,
	                                            nullptr,
	                                            gpu.dispatch);

	device->bindBufferMemory(*buffer, *deviceMemory, 0, gpu.dispatch);
}

Renderer::Renderer(VulkanDevice& gpu, uint32_t width, uint32_t height)
    : swapchain{gpu, width, height}
    , renderPass{gpu, swapchain, createDescriptorSetLayout(gpu)}
    , pipeline{gpu,
               *renderPass.layout,
               *renderPass.renderPass,
               swapchain.extent}
    , modelBuffer{gpu,
                  sizeof(rectVertices),
                  (vk::BufferUsageFlagBits::eVertexBuffer |
                   vk::BufferUsageFlagBits::eTransferSrc |
                   vk::BufferUsageFlagBits::eTransferDst),
                  (vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent)}
    , vertexBuffer{gpu,
                   sizeof(rects[0]) * rects.size(),
                   (vk::BufferUsageFlagBits::eVertexBuffer |
                    vk::BufferUsageFlagBits::eTransferDst),
                   vk::MemoryPropertyFlagBits::eDeviceLocal}
    , stagingBuffer{gpu,
                    sizeof(rects[0]) * rects.size(),
                    vk::BufferUsageFlagBits::eTransferSrc,
                    (vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent)}
{
	// Copy model vertices (directly, we do this only once)

	void* modelData;
	gpu.device->mapMemory(*modelBuffer.deviceMemory,
	                      0,
	                      static_cast<vk::DeviceSize>(sizeof(rectVertices)),
	                      {},
	                      &modelData,
	                      gpu.dispatch);

	memcpy(modelData, rectVertices, sizeof(rectVertices));
	gpu.device->unmapMemory(*modelBuffer.deviceMemory, gpu.dispatch);

	// Copy attribute vertices (via staging buffer)
	const auto verticesSize = sizeof(rects[0]) * rects.size();

	void* vertexData;
	gpu.device->mapMemory(*stagingBuffer.deviceMemory,
	                      0,
	                      static_cast<vk::DeviceSize>(verticesSize),
	                      {},
	                      &vertexData,
	                      gpu.dispatch);

	memcpy(vertexData, rects.data(), verticesSize);
	gpu.device->unmapMemory(*stagingBuffer.deviceMemory, gpu.dispatch);

	copyBuffer(*gpu.device,
	           *gpu.commandPool,
	           gpu.graphicsQueue,
	           *stagingBuffer.buffer,
	           *vertexBuffer.buffer,
	           verticesSize,
	           gpu.dispatch);
}

vk::UniqueDescriptorSetLayout
Renderer::createDescriptorSetLayout(VulkanDevice& gpu)
{
	const vk::DescriptorSetLayoutBinding uboLayoutBinding{
	    0,
	    vk::DescriptorType::eUniformBuffer,
	    1,
	    vk::ShaderStageFlagBits::eVertex};

	return gpu.device->createDescriptorSetLayoutUnique(
	    vk::DescriptorSetLayoutCreateInfo{{}, 1, &uboLayoutBinding},
	    nullptr,
	    gpu.dispatch);
}

/// Complete Vulkan context, purely Vulkan functions can use only this
struct VulkanContext {
	Dispatch           dispatch;
	vk::UniqueInstance instance;
#ifdef PUGL_VK_DYNAMIC_DISPATCH
	vk::UniqueDebugReportCallbackEXT debugCallback;
#endif
	std::unique_ptr<VulkanDevice> gpu;
	std::unique_ptr<Renderer>     renderer;
};

/// Complete application
struct PuglTestApp {
	explicit PuglTestApp(const PuglTestOptions o)
	    : opts{o}
	    , world{pugl::WorldType::program, (PuglWorldFlags)pugl::WorldFlag::threads}
	    , loader{world}
	    , view{world, this}
	{}

	PuglTestOptions          opts;
	pugl::World              world;
	pugl::VulkanLoader       loader;
	pugl::View<PuglTestApp*> view;
	VulkanContext            vk;
	uint32_t                 framesDrawn{0};
	uint32_t                 width{defaultWidth};
	uint32_t                 height{defaultHeight};
	bool                     quit{false};
};

using View = pugl::View<PuglTestApp*>;

#ifdef PUGL_VK_DYNAMIC_DISPATCH
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugReportFlagsEXT flags,
              VkDebugReportObjectTypeEXT,
              uint64_t,
              size_t,
              int32_t,
              const char* layerPrefix,
              const char* msg,
              void*)
{
	std::cerr << vk::to_string(vk::DebugReportFlagsEXT{flags}) << " "
	          << layerPrefix << ": " << msg << std::endl;

	return VK_FALSE;
}
#endif

static bool
hasExtension(const char*                                 name,
             const std::vector<vk::ExtensionProperties>& properties)
{
	for (const auto& p : properties) {
		if (!strcmp(p.extensionName, name)) {
			return true;
		}
	}

	return false;
}

static bool
hasLayer(const char* name, const std::vector<vk::LayerProperties>& properties)
{
	for (const auto& p : properties) {
		if (!strcmp(p.layerName, name)) {
			return true;
		}
	}

	return false;
}

template<class Value>
void
logInfo(const char* heading, const Value& value)
{
	std::cout << std::setw(26) << std::left << (std::string(heading) + ":")
	          << value << std::endl;
}

static void
createInstance(PuglTestApp& app)
{
	Dispatch& dispatch = app.vk.dispatch;
#ifdef PUGL_VK_DYNAMIC_DISPATCH
	dispatch.init(app.loader.getInstanceProcAddrFunc());
#endif

	const auto layerProps = vk::enumerateInstanceLayerProperties(dispatch);
	const auto extProps   = vk::enumerateInstanceExtensionProperties(nullptr,
                                                                   dispatch);
	auto       extensions = pugl::getInstanceExtensions();

	// Add extra extensions we want to use if they are supported
	if (hasExtension("VK_EXT_debug_report", extProps)) {
		extensions.push_back("VK_EXT_debug_report");
	}

	// Add validation layers if error checking is enabled
	std::vector<const char*> layers;
	if (app.opts.errorChecking) {
		for (const char* l : {"VK_LAYER_KHRONOS_validation",
		                      "VK_LAYER_LUNARG_standard_validation"}) {
			if (hasLayer(l, layerProps)) {
				layers.push_back(l);
			}
		}
	}

	for (const auto& e : extensions) {
		logInfo("Using instance extension", e);
	}

	for (const auto& l : layers) {
		logInfo("Using instance layer", l);
	}

	const vk::ApplicationInfo appInfo{
	    "Pugl Vulkan Test",
	    0,
	    nullptr,
	    0,
	    VK_MAKE_VERSION(1, 0, 0),
	};

	const vk::InstanceCreateInfo createInfo{vk::InstanceCreateFlags{},
	                                        &appInfo,
	                                        COUNTED(uint32_t(layers.size()),
	                                                layers.data()),
	                                        COUNTED(uint32_t(extensions.size()),
	                                                extensions.data())};

	app.vk.instance = vk::createInstanceUnique(createInfo, nullptr, dispatch);

#ifdef PUGL_VK_DYNAMIC_DISPATCH
	dispatch.init(*app.vk.instance);
#endif
}

static void
enableDebugging(VulkanContext& vk, const bool verbose)
{
#ifdef PUGL_VK_DYNAMIC_DISPATCH
	if (vk.dispatch.vkCreateDebugReportCallbackEXT) {
		vk::DebugReportFlagsEXT flags =
		    (vk::DebugReportFlagBitsEXT::eWarning |
		     vk::DebugReportFlagBitsEXT::ePerformanceWarning |
		     vk::DebugReportFlagBitsEXT::eError);

		if (verbose) {
			flags |= vk::DebugReportFlagBitsEXT::eInformation;
			flags |= vk::DebugReportFlagBitsEXT::eDebug;
		}

		vk.debugCallback = vk.instance->createDebugReportCallbackEXTUnique(
		    {flags, debugCallback}, nullptr, vk.dispatch);
	}
#else
	(void)vk;
	(void)verbose;
#endif
}

/**
   Checks if a particular physical device is suitable for this application.

   Rendering in Vulkan is off-screen by default.  To get rendered results on
   screen they must be "presented" to a surface.  Vulkan allows devices to
   support presentation on a queue family other than GRAPHICS, or to have no
   present support at all.  However, every graphics card today with display
   output has at least one queue family capable of both GRAPHICS and present.

   This simple application uses one queue for all operations.  More
   specifically, it will make use of GRAPHICS and TRANSFER operations on a
   single queue retrieved from a GRAPHICS queue family that supports present.
*/
static bool
isDeviceSuitable(const vk::SurfaceKHR&    surface,
                 const vk::PhysicalDevice device,
                 uint32_t* const          graphicsIndex,
                 Dispatch&                dispatch)
{
	const auto queueProps = device.getQueueFamilyProperties(dispatch);
	const auto extProps   = device.enumerateDeviceExtensionProperties(nullptr,
                                                                    dispatch);

	uint32_t queueIndex = 0;
	for (; queueIndex < queueProps.size(); ++queueIndex) {
		if (queueProps[queueIndex].queueFlags & vk::QueueFlagBits::eGraphics) {
			if (device.getSurfaceSupportKHR(queueIndex, surface, dispatch)) {
				break;
			}
		}
	}

	if (queueIndex >= queueProps.size()) {
		logError("No graphics+support queue families found on this device\n");
		return false;
	}

	const auto canSwapchain =
	    std::any_of(extProps.begin(),
	                extProps.end(),
	                [&](const vk::ExtensionProperties& e) {
		                return !strcmp(e.extensionName, "VK_KHR_swapchain");
	                });

	if (!canSwapchain) {
		logError("Cannot use a swapchain on this device\n");
		return false;
	}

	*graphicsIndex = queueIndex;

	return true;
}

/**
   Selects a physical graphics device.

   This doesn't try to be clever, and just selects the first suitable device.
*/
static PhysicalDeviceSelection
selectPhysicalDevice(const vk::UniqueInstance& instance,
                     const vk::SurfaceKHR&     surface,
                     Dispatch&                 dispatch)
{
	const auto devices = instance->enumeratePhysicalDevices(dispatch);
	for (const auto& device : devices) {
		const auto deviceProps = device.getProperties(dispatch);

		uint32_t graphicsIndex;
		if (isDeviceSuitable(surface, device, &graphicsIndex, dispatch)) {
			logInfo("Using device", deviceProps.deviceName);
			return {device, graphicsIndex};
		}

		logInfo("Unsuitable device", deviceProps.deviceName);
	}

	return {{}, 0};
}

static vk::SurfaceFormatKHR
selectSurfaceFormat(const vk::PhysicalDevice& physicalDevice,
                    const vk::SurfaceKHR&     surface,
                    Dispatch&                 dispatch)
{
	const auto formats = physicalDevice.getSurfaceFormatsKHR(surface, dispatch);
	if (formats.empty()) {
		throw std::runtime_error("Device has no surface formats");
	}

	for (const auto& format : formats) {
		if (format.format == vk::Format::eB8G8R8A8Unorm &&
		    format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
			return format;
		}
	}

	return formats.at(0);
}

static vk::PresentModeKHR
selectPresentMode(const vk::PhysicalDevice& physicalDevice,
                  const vk::SurfaceKHR&     surface,
                  Dispatch&                 dispatch)
{
	const auto modes = physicalDevice.getSurfacePresentModesKHR(surface,
	                                                            dispatch);

	if (modes.empty()) {
		throw std::runtime_error("Device has no present modes");
	}

	constexpr vk::PresentModeKHR tryModes[] = {//vk::PresentModeKHR::eMailbox,
	                                           vk::PresentModeKHR::eFifo,
	                                           vk::PresentModeKHR::eImmediate};

	for (const auto m : tryModes) {
		if (std::find(modes.begin(), modes.end(), m) != modes.end()) {
			logInfo("Using present mode", vk::to_string(m));
			return m;
		}
	}

	return modes.front();
}

static vk::UniqueDevice
openDevice(const vk::PhysicalDevice& physicalDevice,
           const uint32_t            graphicsFamilyIndex,
           Dispatch&                 dispatch)
{
	const float       graphicsQueuePriority = 1.0f;
	const char* const swapchainName         = "VK_KHR_swapchain";

	const vk::DeviceQueueCreateInfo queueCreateInfo{
	    {},
	    graphicsFamilyIndex,
	    COUNTED(1, &graphicsQueuePriority),
	};

	const vk::DeviceCreateInfo createInfo{
	    {},
	    COUNTED(1, &queueCreateInfo),
	    COUNTED(0, nullptr), // TODO: Pass layers for old implementations
	    COUNTED(1, &swapchainName),
	    nullptr};

	return physicalDevice.createDeviceUnique(createInfo, nullptr, dispatch);
}

VulkanDevice::VulkanDevice(Dispatch&               d,
                           vk::UniqueSurfaceKHR    s,
                           PhysicalDeviceSelection p)
    : dispatch(d)
    , surface(std::move(s))
    , physicalDevice{p.physicalDevice}
    , graphicsIndex{p.graphicsFamilyIndex}
    , surfaceFormat{selectSurfaceFormat(physicalDevice, *surface, dispatch)}
    , presentMode{selectPresentMode(physicalDevice, *surface, dispatch)}
    , device{openDevice(physicalDevice, graphicsIndex, dispatch)}
    , graphicsQueue{device->getQueue(graphicsIndex, 0, dispatch)}
    , commandPool{device->createCommandPoolUnique({{}, graphicsIndex},
                                                  nullptr,
                                                  dispatch)}
    , presentComplete{device->createSemaphoreUnique({}, nullptr, dispatch)}
    , renderFinished{device->createSemaphoreUnique({}, nullptr, dispatch)}
{}

static vk::Extent2D
getSwapchainExtent(const vk::SurfaceCapabilitiesKHR& capabilities,
                   const uint32_t                    width,
                   const uint32_t                    height)
{
	/* The current window size isn't always equivalent to the Vulkan surface
	   drawable size, which can be observed when resizing the window on the fly
	   with some devices and drivers. I suspect this is the result of a race
	   condition between the window size and the drawable size, with the
	   reported window size lagging behind the drawable size, or that the
	   surface is proactively resized internally by the Vulkan driver before
	   the window system finalizes this new size.

	   This has been observed on multiple platforms and seems inherent to the
	   window system.

	   Nonetheless, clamping the returned window size to the minimum and
	   maximum surface capabilities seems to fix the problem completely.

	   Furthermore, it may be possible that we don't even need to query the
	   window system for its size at all, and could possibly simply use the
	   `currentExtent` of the `VkSurfaceCapabilitiesKHR`. */

	return vk::Extent2D{
	    std::min(capabilities.maxImageExtent.width,
	             std::max(capabilities.minImageExtent.width, width)),
	    std::min(capabilities.maxImageExtent.height,
	             std::max(capabilities.minImageExtent.height, height))};
}

static vk::UniqueSwapchainKHR
createRawSwapchain(VulkanDevice&                     gpu,
                   const vk::SurfaceCapabilitiesKHR& capabilities,
                   const vk::Extent2D                extent)
{
	const vk::SwapchainCreateInfoKHR createInfo{
	    {},
	    *gpu.surface,
	    capabilities.minImageCount,
	    gpu.surfaceFormat.format,
	    vk::ColorSpaceKHR::eSrgbNonlinear,
	    extent,
	    1,
	    (vk::ImageUsageFlagBits::eColorAttachment |
	     vk::ImageUsageFlagBits::eTransferDst),
	    vk::SharingMode::eExclusive,
	    COUNTED(0, nullptr),
	    capabilities.currentTransform,
	    vk::CompositeAlphaFlagBitsKHR::eInherit,
	    gpu.presentMode,
	    VK_TRUE,
	    nullptr,
	};

	return gpu.device->createSwapchainKHRUnique(createInfo,
	                                            nullptr,
	                                            gpu.dispatch);
}

static void
recordCommandBuffers(VulkanDevice&     gpu,
                     const Swapchain&  swapchain,
                     const RenderPass& renderPass,
                     const Pipeline&   pipeline,
                     const Renderer&   renderer)
{
	for (size_t i = 0; i < swapchain.images.size(); ++i) {
		const vk::ClearValue clearValue{
		    vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}}};

		swapchain.commandBuffers[i]->begin(
		    {vk::CommandBufferUsageFlagBits::eSimultaneousUse}, gpu.dispatch);

		swapchain.commandBuffers[i]->beginRenderPass(
		    {*renderPass.renderPass,
		     *renderPass.frameBuffers[i],
		     vk::Rect2D{{0, 0}, swapchain.extent},
		     COUNTED(1, &clearValue)},
		    vk::SubpassContents::eInline,
		    gpu.dispatch);

		swapchain.commandBuffers[i]->bindPipeline(
		    vk::PipelineBindPoint::eGraphics, *pipeline.pipeline, gpu.dispatch);

		const std::array<vk::DeviceSize, 1> offsets{0};
		swapchain.commandBuffers[i]->bindVertexBuffers(
		    0, renderer.modelBuffer.buffer.get(), offsets, gpu.dispatch);
		swapchain.commandBuffers[i]->bindVertexBuffers(
		    1, renderer.vertexBuffer.buffer.get(), offsets, gpu.dispatch);

		swapchain.commandBuffers[i]->bindDescriptorSets(
		    vk::PipelineBindPoint::eGraphics,
		    *pipeline.layout,
		    0,
		    1,
		    &*renderPass.descriptorSets[i],
		    0,
		    nullptr,
		    gpu.dispatch);

		swapchain.commandBuffers[i]->draw(
		    4, static_cast<uint32_t>(rects.size()), 0, 0, gpu.dispatch);

		swapchain.commandBuffers[i]->endRenderPass(gpu.dispatch);

		swapchain.commandBuffers[i]->end(gpu.dispatch);
	}
}

Swapchain::Swapchain(VulkanDevice&  gpu,
                     const uint32_t width,
                     const uint32_t height)
    : capabilities{gpu.physicalDevice.getSurfaceCapabilitiesKHR(*gpu.surface,
                                                                gpu.dispatch)}
    , extent{getSwapchainExtent(capabilities, width, height)}
    , rawSwapchain{createRawSwapchain(gpu, capabilities, extent)}
    , images{gpu.device->getSwapchainImagesKHR(*rawSwapchain, gpu.dispatch)}
    , commandBuffers{gpu.device->allocateCommandBuffersUnique(
          vk::CommandBufferAllocateInfo{*gpu.commandPool,
                                        vk::CommandBufferLevel::ePrimary,
                                        static_cast<uint32_t>(images.size())},
          gpu.dispatch)}
{
	for (const auto& image : images) {
		fences.emplace_back(gpu.device->createFenceUnique(
		    {vk::FenceCreateFlagBits::eSignaled}, nullptr, gpu.dispatch));

		imageViews.emplace_back(gpu.device->createImageViewUnique(
		    {{},
		     image,
		     vk::ImageViewType::e2D,
		     gpu.surfaceFormat.format,
		     {},
		     {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}},
		    nullptr,
		    gpu.dispatch));
	}
}

static vk::UniqueRenderPass
createRenderPass(vk::Device&       device,
                 const vk::Format& format,
                 Dispatch&         dispatch)
{
	const vk::AttachmentDescription colorAttachment{
	    {},
	    format,
	    vk::SampleCountFlagBits::e1,
	    vk::AttachmentLoadOp::eClear,
	    vk::AttachmentStoreOp::eStore,
	    vk::AttachmentLoadOp::eDontCare,
	    vk::AttachmentStoreOp::eDontCare,
	    vk::ImageLayout::eUndefined,
	    vk::ImageLayout::ePresentSrcKHR,
	};

	const vk::AttachmentReference colorAttachmentRef{
	    0, vk::ImageLayout::eColorAttachmentOptimal};

	vk::SubpassDescription subpass{{},
	                               vk::PipelineBindPoint::eGraphics,
	                               COUNTED(0, nullptr),
	                               COUNTED(1, &colorAttachmentRef)};

	const vk::SubpassDependency dependency{
	    VK_SUBPASS_EXTERNAL,
	    0,
	    vk::PipelineStageFlagBits::eColorAttachmentOutput,
	    vk::PipelineStageFlagBits::eColorAttachmentOutput,
	    {},
	    (vk::AccessFlagBits::eColorAttachmentRead |
	     vk::AccessFlagBits::eColorAttachmentWrite)};

	return device.createRenderPassUnique({{},
	                                      COUNTED(1, &colorAttachment),
	                                      COUNTED(1, &subpass),
	                                      COUNTED(1, &dependency)},
	                                     nullptr,
	                                     dispatch);
}

RenderPass::RenderPass(VulkanDevice&                 gpu,
                       Swapchain&                    swapchain,
                       vk::UniqueDescriptorSetLayout descriptorSetLayout)
    : renderPass{createRenderPass(*gpu.device,
                                  gpu.surfaceFormat.format,
                                  gpu.dispatch)}
    , layout{std::move(descriptorSetLayout)}
{
	const uint32_t nImages = static_cast<uint32_t>(swapchain.images.size());

	frameBuffers.reserve(nImages);
	for (uint32_t i = 0; i < nImages; ++i) {
		const vk::ImageView attachments[] = {*swapchain.imageViews[i]};

		const vk::FramebufferCreateInfo framebufferInfo{{},
		                                                *renderPass,
		                                                COUNTED(1, attachments),
		                                                swapchain.extent.width,
		                                                swapchain.extent.height,
		                                                1};

		frameBuffers.emplace_back(gpu.device->createFramebufferUnique(
		    framebufferInfo, nullptr, gpu.dispatch));
	}

	// Create uniform buffers
	uniformBuffers.reserve(swapchain.imageViews.size());
	for (uint32_t i = 0; i < nImages; ++i) {
		uniformBuffers.emplace_back(
		    gpu,
		    sizeof(UniformBufferObject),
		    vk::BufferUsageFlagBits::eUniformBuffer,
		    vk::MemoryPropertyFlagBits::eHostVisible |
		        vk::MemoryPropertyFlagBits::eHostCoherent);
	}

	// Create layout descriptor pool

	const vk::DescriptorPoolSize poolSize{vk::DescriptorType::eUniformBuffer,
	                                      nImages};

	descriptorPool = gpu.device->createDescriptorPoolUnique(
	    {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	     nImages,
	     1u,
	     &poolSize},
	    nullptr,
	    gpu.dispatch);

	const std::vector<vk::DescriptorSetLayout> layouts(nImages, *layout);

	descriptorSets = gpu.device->allocateDescriptorSetsUnique(
	    {*descriptorPool, nImages, layouts.data()}, gpu.dispatch);
	descriptorSets.reserve(swapchain.images.size());
	for (uint32_t i = 0; i < nImages; ++i) {
		const vk::DescriptorBufferInfo bufferInfo{*uniformBuffers[i].buffer,
		                                          0,
		                                          sizeof(UniformBufferObject)};

		const vk::WriteDescriptorSet descriptorWrite{
		    *descriptorSets[i],
		    0,
		    0,
		    1,
		    vk::DescriptorType::eUniformBuffer,
		    nullptr,
		    &bufferInfo,
		    nullptr};

		gpu.device->updateDescriptorSets(descriptorWrite,
		                                 nullptr,
		                                 gpu.dispatch);
	}
}

static vk::UniqueShaderModule
createShaderModule(VulkanDevice& gpu, const std::vector<char>& code)
{
	return gpu.device->createShaderModuleUnique(
	    {{}, code.size(), reinterpret_cast<const uint32_t*>(code.data())},
	    nullptr,
	    gpu.dispatch);
}

static std::vector<char>
readFile(const std::string& filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file " + filename);
	}

	size_t            fileSize = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

	file.close();

	return buffer;
}

Pipeline::Pipeline(VulkanDevice&                  gpu,
                   const vk::DescriptorSetLayout& descriptorSetLayout,
                   const vk::RenderPass&          renderPass,
                   const vk::Extent2D             extent)
{
	auto vertShaderCode = readFile("build/shaders/rect.vert.spv");
	auto fragShaderCode = readFile("build/shaders/rect.frag.spv");

	auto vertShaderModule = createShaderModule(gpu, vertShaderCode);
	auto fragShaderModule = createShaderModule(gpu, fragShaderCode);

	const vk::PipelineShaderStageCreateInfo shaderStages[] = {
	    {{}, vk::ShaderStageFlagBits::eVertex, *vertShaderModule, "main"},
	    {{}, vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "main"}};

	auto modelAttributeDescriptions    = getModelAttributeDescriptions();
	auto instanceAttributeDescriptions = getRectAttributeDescriptions();

	std::array<vk::VertexInputBindingDescription, 2> bindingDescriptions{
	    getModelBindingDescription(), getRectBindingDescription()};

	std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	attributeDescriptions.insert(attributeDescriptions.end(),
	                             modelAttributeDescriptions.begin(),
	                             modelAttributeDescriptions.end());
	attributeDescriptions.insert(attributeDescriptions.end(),
	                             instanceAttributeDescriptions.begin(),
	                             instanceAttributeDescriptions.end());

	const vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
	    {},
	    COUNTED(static_cast<uint32_t>(bindingDescriptions.size()),
	            bindingDescriptions.data()),
	    COUNTED(static_cast<uint32_t>(attributeDescriptions.size()),
	            attributeDescriptions.data())};

	const vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
	    {}, vk::PrimitiveTopology::eTriangleStrip};

	const vk::Viewport viewport{
	    0.0f, 0.0f, float(extent.width), float(extent.height), 0.0f, 1.0f};

	const vk::Rect2D scissor{{0, 0}, extent};

	const vk::PipelineViewportStateCreateInfo viewportState{
	    {}, COUNTED(1, &viewport), COUNTED(1, &scissor)};

	const vk::PipelineRasterizationStateCreateInfo rasterizer{
	    {},
	    0,
	    0,
	    vk::PolygonMode::eFill,
	    vk::CullModeFlagBits::eBack,
	    vk::FrontFace::eClockwise,
	    0,
	    0,
	    0,
	    0,
	    1.0f};

	const vk::PipelineMultisampleStateCreateInfo multisampling{
	    {}, vk::SampleCountFlagBits::e1};

	const vk::PipelineColorBlendAttachmentState colorBlendAttachment{
	    true,
	    vk::BlendFactor::eSrcAlpha,
	    vk::BlendFactor::eOneMinusSrcAlpha,
	    vk::BlendOp::eAdd,
	    vk::BlendFactor::eOne,
	    vk::BlendFactor::eZero,
	    vk::BlendOp::eAdd,
	    (vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
	     vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)};

	const vk::PipelineColorBlendStateCreateInfo colorBlending{
	    {}, false, vk::LogicOp::eCopy, COUNTED(1, &colorBlendAttachment)};

	layout = gpu.device->createPipelineLayoutUnique(
	    {{}, 1, &descriptorSetLayout}, nullptr, gpu.dispatch);

	const vk::GraphicsPipelineCreateInfo pipelineInfo{{},
	                                                  COUNTED(2, shaderStages),
	                                                  &vertexInputInfo,
	                                                  &inputAssembly,
	                                                  nullptr,
	                                                  &viewportState,
	                                                  &rasterizer,
	                                                  &multisampling,
	                                                  nullptr,
	                                                  &colorBlending,
	                                                  nullptr,
	                                                  *layout,
	                                                  renderPass};

	pipeline = std::move(gpu.device->createGraphicsPipelinesUnique(
	    {}, pipelineInfo, nullptr, gpu.dispatch)[0]);
}

static pugl::Status
onConfigure(View& view, const pugl::ConfigureEvent& event)
{
	PuglTestApp& app = *view.getData();

	// We just record the size here and lazily resize the surface when exposed
	app.width  = static_cast<uint32_t>(event.width);
	app.height = static_cast<uint32_t>(event.height);

	return pugl::Status::success;
}

static void
recreateSwapchain(VulkanContext& vk,
                  const uint32_t width,
                  const uint32_t height)
{
	vk.gpu->device->waitIdle(vk.dispatch);

	vk.renderer.reset();
	vk.renderer = std::unique_ptr<Renderer>(
	    new Renderer(*vk.gpu, width, height));

	recordCommandBuffers(*vk.gpu,
	                     vk.renderer->swapchain,
	                     vk.renderer->renderPass,
	                     vk.renderer->pipeline,
	                     *vk.renderer);
}

////////////

/////////////////

static void
updateUniformBuffer(vk::UniqueDevice& device,
                    Swapchain&        swapChain,
                    RenderPass&       renderPass,
                    const uint32_t    imageIndex,
                    Dispatch&         dispatch)
{
	UniformBufferObject ubo = {{}};
	mat4Ortho(ubo.projection,
	          0.0f,
	          float(swapChain.extent.width),
	          0.0f,
	          float(swapChain.extent.height),
	          -1.0f,
	          1.0f);

	Buffer& buffer = renderPass.uniformBuffers[imageIndex];
	void*   data =
	    device->mapMemory(*buffer.deviceMemory, 0, sizeof(ubo), {}, dispatch);

	memcpy(data, &ubo, sizeof(ubo));

	device->unmapMemory(*buffer.deviceMemory, dispatch);
}

static pugl::Status
onUpdate(View& view, const pugl::UpdateEvent&)
{
	return view.postRedisplay();
}

static pugl::Status
onExpose(View& view, const pugl::ExposeEvent&)
{
	PuglTestApp&   app = *view.getData();
	VulkanContext& vk  = app.vk;
	if (!vk.gpu) {
		return pugl::Status::unknownError;
	}

	VulkanDevice& gpu      = *vk.gpu;
	Renderer&     renderer = *vk.renderer;

	/* If running continuously and with an asynchronous presentation engine,
	 * Vulkan can blast the queue with rendering work faster than the GPU
	 * can execute it, causing RAM usage to grow indefinitely. We use fences
	 * to limit the number of submitted frames to the number of swapchain
	 * images. These fences will be required later anyway when flushing
	 * persistently mapped uniform buffer ranges.
	 */
	gpu.device->waitForFences(
	    *vk.renderer->swapchain.fences[vk.renderer->swapchain.currentFrame],
	    VK_TRUE,
	    UINT64_MAX,
	    vk.dispatch);

	const auto   rectsSize = sizeof(rects[0]) * rects.size();
	const double time      = view.getWorld().getTime();

	for (size_t i = 0; i < rects.size(); ++i) {
		moveRect(&rects[i], i, rects.size(), app.width, app.height, time);
	}

	void* vertexData;
	gpu.device->mapMemory(*renderer.stagingBuffer.deviceMemory,
	                      0,
	                      static_cast<vk::DeviceSize>(rectsSize),
	                      {},
	                      &vertexData,
	                      vk.dispatch);

	memcpy(vertexData, rects.data(), rectsSize);
	gpu.device->unmapMemory(*renderer.stagingBuffer.deviceMemory, vk.dispatch);

	copyBuffer(*gpu.device,
	           *gpu.commandPool,
	           gpu.graphicsQueue,
	           *renderer.stagingBuffer.buffer,
	           *renderer.vertexBuffer.buffer,
	           rectsSize,
	           vk.dispatch);

	vk::Result result;
	uint32_t   imageIndex;
	while ((result = gpu.device->acquireNextImageKHR(
	            *vk.renderer->swapchain.rawSwapchain,
	            UINT64_MAX,
	            *gpu.presentComplete,
	            nullptr,
	            &imageIndex,
	            vk.dispatch)) != vk::Result::eSuccess) {
		switch (result) {
		case vk::Result::eSuboptimalKHR:
		case vk::Result::eErrorOutOfDateKHR:
			recreateSwapchain(vk, app.width, app.height);
			continue;
		default:
			logError("Could not acquire swapchain image: %d\n", result);
			return pugl::Status::unknownError;
		}
	}

	updateUniformBuffer(gpu.device,
	                    vk.renderer->swapchain,
	                    vk.renderer->renderPass,
	                    imageIndex,
	                    gpu.dispatch);

	gpu.device->resetFences(
	    *vk.renderer->swapchain.fences[vk.renderer->swapchain.currentFrame],
	    vk.dispatch);

	const vk::PipelineStageFlags waitStage =
	    vk::PipelineStageFlagBits::eTransfer;

	const vk::SubmitInfo submitInfo{
	    COUNTED(1, &gpu.presentComplete.get()),
	    &waitStage,
	    COUNTED(1, &*vk.renderer->swapchain.commandBuffers[imageIndex]),
	    COUNTED(1, &*gpu.renderFinished)};
	gpu.graphicsQueue.submit(
	    submitInfo,
	    *vk.renderer->swapchain.fences[vk.renderer->swapchain.currentFrame],
	    vk.dispatch);

	const vk::PresentInfoKHR presentInfo{
	    COUNTED(1, &*gpu.renderFinished),
	    COUNTED(1, &*vk.renderer->swapchain.rawSwapchain, &imageIndex),
	};

	result = gpu.graphicsQueue.presentKHR(&presentInfo, vk.dispatch);
	switch (result) {
	case vk::Result::eSuccess:           // All good
	case vk::Result::eSuboptimalKHR:     // Probably a resize race, ignore
	case vk::Result::eErrorOutOfDateKHR: // Probably a resize race, ignore
		break;
	default:
		vk::throwResultException(result, "vk::Queue::presentKHR");
	}

	++app.framesDrawn;

	vk.renderer->swapchain.currentFrame = (vk.renderer->swapchain.currentFrame +
	                                       1) %
	                                      vk.renderer->swapchain.images.size();

	return pugl::Status::success;
}

static pugl::Status
onKeyPress(View& view, const pugl::KeyPressEvent& event)
{
	if (event.key == PUGL_KEY_ESCAPE || event.key == 'q') {
		view.getData()->quit = true;
	}

	return pugl::Status::success;
}

static pugl::Status
onClose(View& view, const pugl::CloseEvent&)
{
	view.getData()->quit = true;

	return pugl::Status::success;
}

static int
run(const PuglTestOptions opts)
{
	PuglTestApp    app{opts};
	VulkanContext& vk    = app.vk;
	const PuglRect frame = {0.0, 0.0, double(app.width), double(app.height)};

	// Create Vulkan instance
	createInstance(app);

	// Create window
	app.view.setFrame(frame);
	app.view.setBackend(puglVulkanBackend());
	app.view.setHint(pugl::ViewHint::resizable, false);
	const pugl::Status st = app.view.createWindow("Pugl Vulkan");
	if (st != pugl::Status::success) {
		return logError("Failed to create window (%s)\n", pugl::strerror(st));
	}

	// Enable debug logging first so we can get reports during setup
	enableDebugging(vk, app.opts.verbose);

	// Create Vulkan surface for Window

	vk::UniqueSurfaceKHR surface{pugl::createSurfaceUnique(
	    app.loader, app.view, vk.instance.get(), nullptr, vk.dispatch)};

	auto physicalDevice = selectPhysicalDevice(vk.instance,
	                                           *surface,
	                                           vk.dispatch);
	if (!physicalDevice.physicalDevice) {
		return logError("Failed to select a suitable physical device\n");
	}

	vk.gpu = std::unique_ptr<VulkanDevice>(
	    new VulkanDevice(vk.dispatch, std::move(surface), physicalDevice));

	vk.renderer = std::unique_ptr<Renderer>(
	    new Renderer(*vk.gpu,
	                 static_cast<uint32_t>(frame.width),
	                 static_cast<uint32_t>(frame.height)));

	recordCommandBuffers(*vk.gpu,
	                     vk.renderer->swapchain,
	                     vk.renderer->renderPass,
	                     vk.renderer->pipeline,
	                     *vk.renderer);

	app.view.setEventFunc(onConfigure);
	app.view.setEventFunc(onUpdate);
	app.view.setEventFunc(onExpose);
	app.view.setEventFunc(onClose);
	app.view.setEventFunc(onKeyPress);

	PuglFpsPrinter fpsPrinter = {app.world.getTime()};
	app.view.showWindow();
	while (!app.quit) {
		app.world.update(0.0);
		puglPrintFps(app.world.cobj(), &fpsPrinter, &app.framesDrawn);
	}

	if (vk.gpu) {
		vk.gpu->device->waitIdle(vk.dispatch);
	}

	return 0;
}

} // namespace

int
main(int argc, char** argv)
{
	// Parse command line options
	const PuglTestOptions opts = puglParseTestOptions(&argc, &argv);
	if (opts.help) {
		puglPrintTestUsage(argv[0], "");
		return 0;
	}

	try {
		// Run application
		return run(opts);
	} catch (const std::exception& e) {
		std::cerr << "error: " << e.what() << std::endl;
	}
}
