QT += core
QT -= gui

CONFIG += c++11

TARGET = Mytextonboost
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app
INCLUDEPATH += C:/opencv3.1.0/necessary_file/include/opencv \
             C:/opencv3.1.0/necessary_file/include/opencv2 \
             C:/opencv3.1.0/necessary_file/include
LIBS += -L C:/opencv3.1.0/necessary_file/lib/libopencv_*.a

SOURCES += main.cpp
