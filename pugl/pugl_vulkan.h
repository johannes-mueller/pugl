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
   @file pugl_vulkan.h Declaration of Vulkan backend accessor.
*/

#ifndef PUGL_VULKAN_H
#define PUGL_VULKAN_H

#include "pugl/pugl.h"

#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
   @defgroup vulkan Vulkan
   @ingroup pugl
   @{
*/

/**
   Dynamic Vulkan loader.
*/
typedef struct PuglVulkanLoaderImpl PuglVulkanLoader;

/**
   Create a new dynamic loader for Vulkan functions.

   This dynamically loads the Vulkan library.

   @return A new Vulkan loader, or null on failure.
*/
PUGL_API PuglVulkanLoader*
puglNewVulkanLoader(PuglWorld* world);

/**
   Free a loader created with puglNewVulkanLoader().
*/
PUGL_API void
puglFreeVulkanLoader(PuglVulkanLoader* loader);

/**
   Return the vkGetInstanceProcAddr function.
*/
PUGL_API PFN_vkGetInstanceProcAddr
puglGetInstanceProcAddrFunc(PuglVulkanLoader* loader);

/**
   Return the vkGetDeviceProcAddr function.
*/
PUGL_API PFN_vkGetDeviceProcAddr
puglGetDeviceProcAddrFunc(PuglVulkanLoader* loader);

/**
   Return the Vulkan instance extensions required to draw to a PuglView.

   This function simply returns fixed strings, it does not load or access
   Vulkan or the window system.  The returned array always contains at least
   "VK_KHR_surface".

   @param[out] count The number of extensions in the returned array.
   @return An array of extension name strings.
*/
PUGL_API const char* const*
puglGetInstanceExtensions(uint32_t* count);

/**
   Create a Vulkan surface for a PuglView.

   This function dynamically loads the vulkan library, so the application does
   not need to be linked to it.  If loading fails,
   VK_ERROR_INITIALIZATION_FAILED is returned.

   @param loader The loader for Vulkan functions.
   @param view The view the surface is to be displayed on.
   @param instance The Vulkan instance.
   @param allocator Vulkan allocation callbacks, may be NULL.
   @param[out] surface Pointed to a newly created Vulkan surface.
   @return VK_SUCCESS on success, or a Vulkan error code.
*/
PUGL_API VkResult
puglCreateSurface(PuglVulkanLoader*            loader,
                  PuglView*                    view,
                  VkInstance                   instance,
                  const VkAllocationCallbacks* allocator,
                  VkSurfaceKHR*                surface);

/**
   Vulkan graphics backend.

   Pass the return value to puglInitBackend() to draw to a view with Vulkan.
*/
PUGL_API const PuglBackend*
puglVulkanBackend(void);

/**
   @}
*/

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // PUGL_VULKAN_H
