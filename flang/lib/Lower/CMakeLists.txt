set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-unused-parameter")

add_flang_library(FortranLower
  Bridge.cpp
  ConvertExpr.cpp
  ConvertType.cpp
  FIRBuilder.cpp
  Intrinsics.cpp
  IO.cpp
  Mangler.cpp
  PFTBuilder.cpp
  Runtime.cpp
)

target_link_libraries(FortranLower
  FIRSupport
  MLIRAffine
  MLIRLLVMIR
  MLIRLoopToStandard
  MLIRStandardOps

  LINK_COMPONENTS
  Support
)
