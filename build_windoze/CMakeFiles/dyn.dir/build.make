# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\14167\CDT\src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\14167\CDT\build_windoze

# Include any dependencies generated for this target.
include CMakeFiles/dyn.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dyn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dyn.dir/flags.make

CMakeFiles/dyn.dir/dynlib.c.obj: CMakeFiles/dyn.dir/flags.make
CMakeFiles/dyn.dir/dynlib.c.obj: C:/Users/14167/CDT/src/dynlib.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\14167\CDT\build_windoze\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/dyn.dir/dynlib.c.obj"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles\dyn.dir\dynlib.c.obj   -c C:\Users\14167\CDT\src\dynlib.c

CMakeFiles/dyn.dir/dynlib.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/dyn.dir/dynlib.c.i"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\14167\CDT\src\dynlib.c > CMakeFiles\dyn.dir\dynlib.c.i

CMakeFiles/dyn.dir/dynlib.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/dyn.dir/dynlib.c.s"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\14167\CDT\src\dynlib.c -o CMakeFiles\dyn.dir\dynlib.c.s

# Object files for target dyn
dyn_OBJECTS = \
"CMakeFiles/dyn.dir/dynlib.c.obj"

# External object files for target dyn
dyn_EXTERNAL_OBJECTS =

libdyn.dll: CMakeFiles/dyn.dir/dynlib.c.obj
libdyn.dll: CMakeFiles/dyn.dir/build.make
libdyn.dll: CMakeFiles/dyn.dir/linklibs.rsp
libdyn.dll: CMakeFiles/dyn.dir/objects1.rsp
libdyn.dll: CMakeFiles/dyn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\14167\CDT\build_windoze\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C shared library libdyn.dll"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\dyn.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dyn.dir/build: libdyn.dll

.PHONY : CMakeFiles/dyn.dir/build

CMakeFiles/dyn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\dyn.dir\cmake_clean.cmake
.PHONY : CMakeFiles/dyn.dir/clean

CMakeFiles/dyn.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\14167\CDT\src C:\Users\14167\CDT\src C:\Users\14167\CDT\build_windoze C:\Users\14167\CDT\build_windoze C:\Users\14167\CDT\build_windoze\CMakeFiles\dyn.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dyn.dir/depend

