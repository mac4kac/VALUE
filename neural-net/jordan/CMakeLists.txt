
INCLUDE_DIRECTORIES(
	/usr/local/include/nnfw
	/usr/include/qt4/Qt
	/usr/include/qt4/QtCore
	/usr/include/qt4/QtXml
	/usr/share/qt4/include
)

LINK_LIBRARIES(
	/usr/local/lib64/libnnfw.a
	/usr/lib/libQtXml.so
	/usr/lib/libQtCore.so
)

ADD_EXECUTABLE(jordanTest 
	jordanTest.cpp
	)


