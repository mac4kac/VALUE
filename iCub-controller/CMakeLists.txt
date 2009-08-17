SET(PROJECTNAME Grasp)

PROJECT(${PROJECTNAME})

FIND_PACKAGE(YARP)
FIND_PACKAGE(ICUB)
FIND_PACKAGE(OpenCV)

# Find .cpp and .h files automatically.  This is a bit lazy,
# and in principle it would be better to list these files manually.
FILE(GLOB folder_source *.cpp)
FILE(GLOB folder_header *.h)

# Declare groups of source and header files -- makes things pretty in MSVC.
SOURCE_GROUP("Source Files" FILES ${folder_source})
SOURCE_GROUP("Header Files" FILES ${folder_header})

# Add our include files into our compiler's search path.
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
INCLUDE_DIRECTORIES(
	/usr/local/include/nnfw
	/usr/include/qt4/Qt
	/usr/include/qt4/QtCore
	/usr/include/qt4/QtXml
	/usr/share/qt4/include
)

LINK_LIBRARIES(${OPENCV_LIBRARIES})
LINK_LIBRARIES(
	/usr/local/lib64/libnnfw.a
	/usr/lib/libQtXml.so
	/usr/lib/libQtCore.so
)

# Create everything needed to build our executable.
ADD_EXECUTABLE(${PROJECTNAME} ${folder_source} ${folder_header})
INSTALL_TARGETS(/bin Grasp) 


