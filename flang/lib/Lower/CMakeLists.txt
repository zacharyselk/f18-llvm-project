set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-unused-parameter")

add_flang_library(FortranLower
  Bridge.cpp
  CharRT.cpp
  ConvertExpr.cpp
  ConvertType.cpp
  FIRBuilder.cpp
  Intrinsics.cpp
  IO.cpp
  Mangler.cpp
  OpenMP.cpp
  PFTBuilder.cpp
  Runtime.cpp

  DEPENDS
  MLIROpAsmInterfacesIncGen
  MLIRControlFlowInterfaces
  MLIRLinalgStructuredOpsInterfaceIncGen
)

target_link_libraries(FortranLower
  FIRSupport
  MLIRAffineToStandard
  MLIRLLVMIR
  MLIRLoopToStandard
  MLIRStandardOps
)
