SET(PROJECTNAME SOMTest)
PROJECT(${PROJECTNAME})

FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(SDL REQUIRED)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}

	/usr/local/include/nnfw
	/usr/include/qt4/Qt
	/usr/include/qt4/QtCore
	/usr/include/qt4/QtXml
	/usr/share/qt4/include
	
	)
INCLUDE_DIRECTORIES(${SDL_INCLUDE_DIR})
LINK_LIBRARIES(${SDL_LIBRARY}
	
	/usr/local/lib64/libnnfw.a
	/usr/lib/libQtXml.so
	/usr/lib/libQtCore.so
	
	)

FILE(GLOB folder_source *.cpp)
FILE(GLOB folder_header *.h)


ADD_EXECUTABLE(${PROJECTNAME} ${folder_source} ${folder_header})

INSTALL(TARGETS ${PROJECTNAME} RUNTIME DESTINATION bin)
