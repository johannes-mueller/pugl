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
   @file pugl.hpp Pugl C++ API wrapper.
*/

#ifndef PUGL_PUGL_HPP
#define PUGL_PUGL_HPP

#include "pugl/pugl.h"
#include "pugl/pugl_gl.h" // FIXME

#include <chrono>
#include <functional>
#include <ratio>
#include <stdexcept>
#include <utility>

/**
   @defgroup puglxx C++

   C++ API wrapper.

   @ingroup pugl_api
   @{
*/

/**
   Pugl C++ API namespace.
*/
namespace pugl {

enum class Status {
	success             = PUGL_SUCCESS,
	failure             = PUGL_FAILURE,
	unknownError        = PUGL_UNKNOWN_ERROR,
	badBackend          = PUGL_BAD_BACKEND,
	backendFailed       = PUGL_BACKEND_FAILED,
	registrationFailed  = PUGL_REGISTRATION_FAILED,
	createWindowFailed  = PUGL_CREATE_WINDOW_FAILED,
	setFormatFailed     = PUGL_SET_FORMAT_FAILED,
	createContextFailed = PUGL_CREATE_CONTEXT_FAILED,
	unsupportedType     = PUGL_UNSUPPORTED_TYPE,
};

enum class ViewHint {
	useCompatProfile,    ///< Use compatible (not core) OpenGL profile
	useDebugContext,     ///< True to use a debug OpenGL context
	contextVersionMajor, ///< OpenGL context major version
	contextVersionMinor, ///< OpenGL context minor version
	redBits,             ///< Number of bits for red channel
	greenBits,           ///< Number of bits for green channel
	blueBits,            ///< Number of bits for blue channel
	alphaBits,           ///< Number of bits for alpha channel
	depthBits,           ///< Number of bits for depth buffer
	stencilBits,         ///< Number of bits for stencil buffer
	samples,             ///< Number of samples per pixel (AA)
	doubleBuffer,        ///< True if double buffering should be used
	swapInterval,        ///< Number of frames between buffer swaps
	resizable,           ///< True if window should be resizable
	ignoreKeyRepeat,     ///< True if key repeat events are ignored
};

enum class WorldType { program = PUGL_PROGRAM, module = PUGL_MODULE };

enum class WorldFlag { threads = PUGL_WORLD_THREADS };

using WorldFlags = PuglWorldFlags;

using Rect         = PuglRect;
using NativeWindow = PuglNativeWindow;
using GlFunc       = PuglGlFunc;
using Event        = PuglEvent;

template<PuglEventType t, class Base>
struct TypedEvent : public Base {
	static constexpr const PuglEventType type = t;
};

/* Strong types for every event type. */

using CreateEvent        = TypedEvent<PUGL_CREATE, PuglEventAny>;
using DestroyEvent       = TypedEvent<PUGL_DESTROY, PuglEventAny>;
using ConfigureEvent     = TypedEvent<PUGL_CONFIGURE, PuglEventConfigure>;
using MapEvent           = TypedEvent<PUGL_MAP, PuglEventAny>;
using UnmapEvent         = TypedEvent<PUGL_UNMAP, PuglEventAny>;
using UpdateEvent        = TypedEvent<PUGL_UPDATE, PuglEventAny>;
using ExposeEvent        = TypedEvent<PUGL_EXPOSE, PuglEventExpose>;
using CloseEvent         = TypedEvent<PUGL_CLOSE, PuglEventAny>;
using FocusInEvent       = TypedEvent<PUGL_FOCUS_IN, PuglEventFocus>;
using FocusOutEvent      = TypedEvent<PUGL_FOCUS_OUT, PuglEventFocus>;
using KeyPressEvent      = TypedEvent<PUGL_KEY_PRESS, PuglEventKey>;
using KeyReleaseEvent    = TypedEvent<PUGL_KEY_RELEASE, PuglEventKey>;
using TextEvent          = TypedEvent<PUGL_TEXT, PuglEventText>;
using EnterEvent         = TypedEvent<PUGL_POINTER_IN, PuglEventCrossing>;
using LeaveEvent         = TypedEvent<PUGL_POINTER_OUT, PuglEventCrossing>;
using ButtonPressEvent   = TypedEvent<PUGL_BUTTON_PRESS, PuglEventButton>;
using ButtonReleaseEvent = TypedEvent<PUGL_BUTTON_RELEASE, PuglEventButton>;
using MotionEvent        = TypedEvent<PUGL_MOTION, PuglEventMotion>;
using ScrollEvent        = TypedEvent<PUGL_SCROLL, PuglEventScroll>;
using ClientEvent        = TypedEvent<PUGL_CLIENT, PuglEventClient>;
using TimerEvent         = TypedEvent<PUGL_TIMER, PuglEventTimer>;

static inline const char*
strerror(pugl::Status status)
{
	return puglStrerror(static_cast<PuglStatus>(status));
}

static inline GlFunc
getProcAddress(const char* name)
{
	return puglGetProcAddress(name);
}

class World;

/**
   A `std::chrono` compatible clock that uses Pugl time.
*/
class Clock
{
public:
	using rep        = double;                         ///< Time representation
	using duration   = std::chrono::duration<double>;  ///< Duration in seconds
	using time_point = std::chrono::time_point<Clock>; ///< A Pugl time point

	static constexpr bool is_steady = true; ///< Steady clock flag, always true

	/// Construct a clock that uses time from puglGetTime()
	explicit Clock(World& world)
	    : _world{world}
	{}

	/// Return the current time
	time_point now() const;

private:
	const pugl::World& _world;
};

class World
{
public:
	explicit World(WorldType type, WorldFlags flags = {})
	    : _clock(*this)
	    , _world(puglNewWorld(static_cast<PuglWorldType>(type), flags))
	{
		if (!_world) {
			throw std::runtime_error("Failed to create pugl::World");
		}
	}

	~World() { puglFreeWorld(_world); }

	World(const World&) = delete;
	World& operator=(const World&) = delete;
	World(World&&)                 = delete;
	World&& operator=(World&&) = delete;

	Status setClassName(const char* const name)
	{
		return static_cast<Status>(puglSetClassName(_world, name));
	}

	double getTime() const { return puglGetTime(_world); }

	Status update(const double timeout)
	{
		return static_cast<Status>(puglUpdate(_world, timeout));
	}

	const PuglWorld* cobj() const { return _world; }
	PuglWorld*       cobj() { return _world; }

	const Clock& clock() { return _clock; }

private:
	Clock            _clock;
	PuglWorld* const _world;
};

inline Clock::time_point
Clock::now() const
{
	return time_point{duration{_world.getTime()}};
}

class ViewBase
{
public:
	explicit ViewBase(World& world)
	    : _world(world)
	    , _view(puglNewView(world.cobj()))
	{
		if (!_view) {
			throw std::runtime_error("Failed to create pugl::View");
		}

		puglSetHandle(_view, this);
	}

	~ViewBase() { puglFreeView(_view); }

	ViewBase(const ViewBase&) = delete;
	ViewBase(ViewBase&&)      = delete;
	ViewBase&  operator=(const ViewBase&) = delete;
	ViewBase&& operator=(ViewBase&&) = delete;

	Status setHint(ViewHint hint, int value)
	{
		return static_cast<Status>(
		    puglSetViewHint(_view, static_cast<PuglViewHint>(hint), value));
	}

	bool getVisible() const { return puglGetVisible(_view); }

	Status postRedisplay()
	{
		return static_cast<Status>(puglPostRedisplay(_view));
	}

	const pugl::World& getWorld() const { return _world; }
	pugl::World&       getWorld() { return _world; }

	Rect getFrame() const { return puglGetFrame(_view); }

	Status setFrame(Rect frame)
	{
		return static_cast<Status>(puglSetFrame(_view, frame));
	}

	Status setMinSize(int width, int height)
	{
		return static_cast<Status>(puglSetMinSize(_view, width, height));
	}

	Status setAspectRatio(int minX, int minY, int maxX, int maxY)
	{
		return static_cast<Status>(
		    puglSetAspectRatio(_view, minX, minY, maxX, maxY));
	}

	Status setWindowTitle(const char* title)
	{
		return static_cast<Status>(puglSetWindowTitle(_view, title));
	}

	Status setParentWindow(NativeWindow parent)
	{
		return static_cast<Status>(puglSetParentWindow(_view, parent));
	}

	Status setTransientFor(NativeWindow parent)
	{
		return static_cast<Status>(puglSetTransientFor(_view, parent));
	}

	Status createWindow(const char* title)
	{
		return static_cast<Status>(puglCreateWindow(_view, title));
	}

	Status showWindow() { return static_cast<Status>(puglShowWindow(_view)); }

	Status hideWindow() { return static_cast<Status>(puglHideWindow(_view)); }

	NativeWindow getNativeWindow() { return puglGetNativeWindow(_view); }

	Status setBackend(const PuglBackend* backend)
	{
		return static_cast<Status>(puglSetBackend(_view, backend));
	}

	void* getContext() { return puglGetContext(_view); }

	bool hasFocus() const { return puglHasFocus(_view); }

	Status grabFocus() { return static_cast<Status>(puglGrabFocus(_view)); }

	Status requestAttention()
	{
		return static_cast<Status>(puglRequestAttention(_view));
	}

	PuglView* cobj() { return _view; }

protected:
	World&    _world;
	PuglView* _view;
};

/**
   A drawable region that receives events.

   This is a thin wrapper for a PuglView that contains only a pointer.

   @ingroup puglxx
*/
template<typename Data>
class View : public ViewBase
{
public:
	explicit View(World& world)
	    : ViewBase{world}
	    , _data{}
	{
		puglSetEventFunc(_view, _onEvent);
	}

	View(World& world, Data data)
	    : ViewBase{world}
	    , _data{data}
	{
		puglSetEventFunc(_view, _onEvent);
	}

	template<class E>
	using TypedEventFunc = std::function<pugl::Status(View&, const E&)>;

	template<class HandledEvent>
	Status setEventFunc(TypedEventFunc<HandledEvent> handler)
	{
		std::get<HandledEvent::type>(_eventFuncs) = handler;

		return Status::success;
	}

	template<class E>
	using TypedEventHandler = pugl::Status (*)(View&, const E&);

	template<class HandledEvent>
	Status setEventFunc(TypedEventHandler<HandledEvent> handler)
	{
		std::get<HandledEvent::type>(_eventFuncs) = handler;

		return Status::success;
	}

	const Data& getData() const { return _data; }
	Data&       getData() { return _data; }

private:
	using NothingEvent = TypedEvent<PUGL_NOTHING, PuglEvent>;

	/**
	   A tuple of event handlers, one for each event type.

	   Note that the indices here must correspond to PuglEventType.
	*/
	using EventFuncs = std::tuple<TypedEventFunc<NothingEvent>,
	                              TypedEventFunc<CreateEvent>,
	                              TypedEventFunc<DestroyEvent>,
	                              TypedEventFunc<ConfigureEvent>,
	                              TypedEventFunc<MapEvent>,
	                              TypedEventFunc<UnmapEvent>,
	                              TypedEventFunc<UpdateEvent>,
	                              TypedEventFunc<ExposeEvent>,
	                              TypedEventFunc<CloseEvent>,
	                              TypedEventFunc<FocusInEvent>,
	                              TypedEventFunc<FocusOutEvent>,
	                              TypedEventFunc<KeyPressEvent>,
	                              TypedEventFunc<KeyReleaseEvent>,
	                              TypedEventFunc<TextEvent>,
	                              TypedEventFunc<EnterEvent>,
	                              TypedEventFunc<LeaveEvent>,
	                              TypedEventFunc<ButtonPressEvent>,
	                              TypedEventFunc<ButtonReleaseEvent>,
	                              TypedEventFunc<MotionEvent>,
	                              TypedEventFunc<ScrollEvent>,
	                              TypedEventFunc<ClientEvent>,
	                              TypedEventFunc<TimerEvent>>;

	using EventFunc = std::function<pugl::Status(View&, const PuglEvent&)>;

	static PuglStatus _onEvent(PuglView* view, const PuglEvent* event) noexcept
	{
		View* self = static_cast<View*>(puglGetHandle(view));

		return static_cast<PuglStatus>(self->dispatchEvent(*event));
	}

	Status dispatchEvent(const PuglEvent& event)
	{
		switch (event.type) {
		case PUGL_NOTHING:
			return Status::success;
		case PUGL_CREATE:
			return dispatchTypedEvent(
			    static_cast<const CreateEvent&>(event.any));
		case PUGL_DESTROY:
			return dispatchTypedEvent(
			    static_cast<const DestroyEvent&>(event.any));
		case PUGL_CONFIGURE:
			return dispatchTypedEvent(
			    static_cast<const ConfigureEvent&>(event.configure));
		case PUGL_MAP:
			return dispatchTypedEvent(static_cast<const MapEvent&>(event.any));
		case PUGL_UNMAP:
			return dispatchTypedEvent(
			    static_cast<const UnmapEvent&>(event.any));
		case PUGL_UPDATE:
			return dispatchTypedEvent(
			    static_cast<const UpdateEvent&>(event.any));
		case PUGL_EXPOSE:
			return dispatchTypedEvent(
			    static_cast<const ExposeEvent&>(event.expose));
		case PUGL_CLOSE:
			return dispatchTypedEvent(
			    static_cast<const CloseEvent&>(event.any));
		case PUGL_FOCUS_IN:
			return dispatchTypedEvent(
			    static_cast<const FocusInEvent&>(event.focus));
		case PUGL_FOCUS_OUT:
			return dispatchTypedEvent(
			    static_cast<const FocusOutEvent&>(event.focus));
		case PUGL_KEY_PRESS:
			return dispatchTypedEvent(
			    static_cast<const KeyPressEvent&>(event.key));
		case PUGL_KEY_RELEASE:
			return dispatchTypedEvent(
			    static_cast<const KeyReleaseEvent&>(event.key));
		case PUGL_TEXT:
			return dispatchTypedEvent(
			    static_cast<const TextEvent&>(event.text));
		case PUGL_POINTER_IN:
			return dispatchTypedEvent(
			    static_cast<const EnterEvent&>(event.crossing));
		case PUGL_POINTER_OUT:
			return dispatchTypedEvent(
			    static_cast<const LeaveEvent&>(event.crossing));
		case PUGL_BUTTON_PRESS:
			return dispatchTypedEvent(
			    static_cast<const ButtonPressEvent&>(event.button));
		case PUGL_BUTTON_RELEASE:
			return dispatchTypedEvent(
			    static_cast<const ButtonReleaseEvent&>(event.button));
		case PUGL_MOTION:
			return dispatchTypedEvent(
			    static_cast<const MotionEvent&>(event.motion));
		case PUGL_SCROLL:
			return dispatchTypedEvent(
			    static_cast<const ScrollEvent&>(event.scroll));
		case PUGL_CLIENT:
			return dispatchTypedEvent(
			    static_cast<const ClientEvent&>(event.client));
		case PUGL_TIMER:
			return dispatchTypedEvent(
			    static_cast<const TimerEvent&>(event.timer));
		}

		return Status::failure;
	}

	template<class E>
	Status dispatchTypedEvent(const E& event)
	{
		auto& handler = std::get<E::type>(_eventFuncs);
		if (handler) {
			return handler(*this, event);
		}

		return Status::success;
	}

	Data       _data;
	EventFuncs _eventFuncs;
};

} // namespace pugl

/**
   @}
*/

#endif /* PUGL_PUGL_HPP */
