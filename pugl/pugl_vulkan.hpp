/*
  Copyright 2012-2019 David Robillard <http://drobilla.net>

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

/**
   @file pugl_vulkan.hpp Vulkan C++ interface.
*/

#ifndef PUGL_VULKAN_BACKEND_HPP
#define PUGL_VULKAN_BACKEND_HPP

#include "pugl/pugl.hpp"
#include "pugl/pugl_vulkan.h"

#include <vulkan/vulkan.hpp>

namespace pugl {

class PUGL_API VulkanLoader
{
public:
	explicit VulkanLoader(World& world)
	    : _loader{puglNewVulkanLoader(world.cobj())}
	{
	}

	VulkanLoader(const VulkanLoader&) = delete;
	VulkanLoader(VulkanLoader&&)      = delete;
	VulkanLoader& operator=(const VulkanLoader&) = delete;
	VulkanLoader& operator=(VulkanLoader&&) = delete;

	~VulkanLoader() { puglFreeVulkanLoader(_loader); }

	inline PFN_vkGetInstanceProcAddr getInstanceProcAddrFunc()
	{
		return puglGetInstanceProcAddrFunc(_loader);
	}

	inline PFN_vkGetDeviceProcAddr getDeviceProcAddrFunc()
	{
		return puglGetDeviceProcAddrFunc(_loader);
	}

	const PuglVulkanLoader* cobj() const { return _loader; }
	PuglVulkanLoader*       cobj() { return _loader; }

private:
	PuglVulkanLoader* _loader;
};

/**
   Return the Vulkan instance extensions required to draw to a PuglView.

   If successful, the returned array always contains "VK_KHR_surface", along
   with whatever other platform-specific extensions are required.

   @return A vector of extension name strings.
*/
inline std::vector<const char*>
getInstanceExtensions()
{
	uint32_t                 count      = 0;
	const char* const* const extensions = puglGetInstanceExtensions(&count);

	return std::vector<const char*>(extensions, extensions + count);
}

inline vk::SurfaceKHR
createSurface(pugl::VulkanLoader&                         loader,
              pugl::ViewBase&                             view,
              vk::Instance&                               instance,
              vk::Optional<const vk::AllocationCallbacks> allocator = nullptr)
{
	VkSurfaceKHR   surface;
	const VkResult vr = puglCreateSurface(
	        loader.cobj(),
	        view.cobj(),
	        instance,
	        allocator ? &static_cast<const VkAllocationCallbacks&>(*allocator)
	                  : nullptr,
	        &surface);

	if (vr) {
		vk::throwResultException(static_cast<vk::Result>(vr),
		                         "pugl::createSurface");
	}

	return surface;
}

template <class Dispatch>
inline vk::UniqueHandle<vk::SurfaceKHR, Dispatch>
createSurfaceUnique(pugl::VulkanLoader&                         loader,
                    pugl::ViewBase&                             view,
                    vk::Instance&                               instance,
                    vk::Optional<const vk::AllocationCallbacks> allocator,
                    const Dispatch&                             dispatch)
{
	vk::SurfaceKHR surface = createSurface(loader, view, instance, allocator);

	return vk::UniqueHandle<vk::SurfaceKHR, Dispatch>{
	        surface,
	        vk::ObjectDestroy<vk::Instance, Dispatch>{
	                instance, allocator, dispatch}};
}

} // namespace pugl

#endif // PUGL_VULKAN_BACKEND_HPP
