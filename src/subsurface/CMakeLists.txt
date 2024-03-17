set(MI_PLUGIN_PREFIX "subsurface")

add_plugin(dipole dipole.cpp)
# TODO
#add_plugin(betterDipole betterDipole.cpp)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
