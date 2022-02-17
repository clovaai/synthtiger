#!/bin/bash
set -euxo pipefail

# Pillow

pacman --noconfirm -S \
    mingw-w64-x86_64-libjpeg-turbo \
    mingw-w64-x86_64-zlib \
    mingw-w64-x86_64-libtiff \
    mingw-w64-x86_64-freetype \
    mingw-w64-x86_64-lcms2 \
    mingw-w64-x86_64-libwebp \
    mingw-w64-x86_64-openjpeg2 \
    mingw-w64-x86_64-libimagequant \
    mingw-w64-x86_64-libraqm
