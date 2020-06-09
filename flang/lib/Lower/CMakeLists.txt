set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-unused-parameter")

add_flang_library(FortranLower
  Bridge.cpp
  CallInterface.cpp
  CharacterExpr.cpp
  CharacterRuntime.cpp
  ComplexExpr.cpp
  ConvertExpr.cpp
  ConvertType.cpp
  DoLoopHelper.cpp
  FIRBuilder.cpp
  Image.cpp
  IntrinsicCall.cpp
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
  FIROptimizer
  MLIRAffineToStandard
  MLIRLLVMIR
  MLIRSCFToStandard
  MLIRStandardOps
)
