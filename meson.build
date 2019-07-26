project('pugl', 'c',
        version: '0.3.0',
        default_options: [
          'buildtype=debugoptimized',
          'warning_level=3',
          'c_std=c99',
        ],
        meson_version : '>= 0.48.0',
        license: 'ISC')

cc = meson.get_compiler('c')

add_project_arguments(
  cc.get_supported_arguments(
    [
      '-Wall',
      '-Wcast-align',
      '-Wcast-qual',
      '-Wextra',
      '-Wlogical-op',
      '-Wmissing-declarations',
      '-Wmissing-prototypes',
      '-Wno-unused-parameter',
      '-Wredundant-decls',
      '-Wshadow',
      '-Wstrict-overflow',
      '-Wstrict-prototypes',
      '-Wsuggest-attribute=noreturn',
      '-Wundef',
      '-Wunsafe-loop-optimizations',
      '-Wwrite-strings',
      '-fstrict-overflow',
      '-pedantic',
    ]),
  language: 'c')

headers = [
  'pugl/detail/implementation.h',
  'pugl/detail/mac.h',
  'pugl/detail/types.h',
  'pugl/detail/win.h',
  'pugl/detail/x11.h',
  'pugl/glew.h',
  'pugl/gl.h',
  'pugl/glu.h',
  'pugl/pugl_cairo_backend.h',
  'pugl/pugl_cairo.h',
  'pugl/pugl_gl_backend.h',
  'pugl/pugl_gl.h',
  'pugl/pugl.h',
  'pugl/pugl_stub_backend.h',
  'pugl/pugl_stub.h',
]

sources = ['pugl/detail/implementation.c']

cairo_dep = dependency('cairo', required: false)
m_dep = cc.find_library('m', required: false)
x11_dep = dependency('x11', required: false)
gl_dep = dependency('GL')

if gl_dep.found()
  add_project_arguments(['-DPUGL_HAVE_GL=1'], language: 'c')
endif
if cairo_dep.found()
  add_project_arguments(['-DPUCAIRO_HAVE_CAIRO=1'], language: 'c')
endif

if x11_dep.found()
  sources += 'pugl/detail/x11.c'
  if gl_dep.found()
    sources += 'pugl/detail/x11_gl.c'
  endif
  if cairo_dep.found()
    sources += 'pugl/detail/x11_cairo.c'
  endif
endif

if build_machine.system() == 'darwin'
  sources += 'pugl/pugl_osx.m'
elif build_machine.system() == 'windows'
  sources += 'pugl/pugl_win.c'
endif

dependencies = [x11_dep, gl_dep, m_dep, cairo_dep]

libpugl = library('pugl',
                  headers + sources,
                  c_args: ['-DPUGL_SHARED', '-DPUGL_INTERNAL'],
                  dependencies: dependencies)

pugl_dep = declare_dependency(link_with: libpugl,
                              include_directories: ['pugl'],
                              dependencies: dependencies)

examples = ['pugl_cairo_demo',
            'pugl_embed_demo',
            # 'pugl_gl3_demo',
            'pugl_print_events',
            'pugl_window_demo']

foreach example : examples
  executable(example,
             'examples/' + example + '.c',
             dependencies: [pugl_dep])
endforeach
