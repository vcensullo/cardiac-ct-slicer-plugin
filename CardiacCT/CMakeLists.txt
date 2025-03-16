cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(Cardiac CT Tools)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/CardiacCT")
set(EXTENSION_CONTRIBUTORS "Vittorio Censullo, AITeRTC, Carlo Zebele, Gianluca Spidalieri")
set(EXTENSION_DESCRIPTION "Cardiac CT 4D Analysis Plugin
A plugin specifically designed for 4D Cardiac CT imaging analysis, currently consisting of two modules. The plugin provides cardiologists and radiologists with tools for cardiac function evaluation, with capacity for additional analytical modules in future updates.

Volume Quantification Module

Measures right and left ventricular volumes through chamber segmentation, calculates ejection fraction, and generates temporal volume curves for functional assessment, exporting data in CSV and PDF

3D Wall Motion analysis

It analyzes myocardial wall motion in three dimensions,and identifies wall motion abnormalities through motion tracking and visualization.
")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/CardiacCT.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/CardiacCT/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(CardiacVolumeAnalysis)
add_subdirectory(CardiacMotionAnalysis)
add_subdirectory(CTOptimizer)
add_subdirectory(CoronarySegmentation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
