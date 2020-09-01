//===-- Optimizer/Dialect/FIRDialect.h -- FIR dialect -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_DIALECT_FIRDIALECT_H
#define OPTIMIZER_DIALECT_FIRDIALECT_H

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"

namespace fir {

/// FIR dialect
class FIROpsDialect final : public mlir::Dialect {
public:
  explicit FIROpsDialect(mlir::MLIRContext *ctx);
  virtual ~FIROpsDialect();

  static llvm::StringRef getDialectNamespace() { return "fir"; }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type ty, mlir::DialectAsmPrinter &p) const override;

  mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                 mlir::Type type) const override;
  void printAttribute(mlir::Attribute attr,
                      mlir::DialectAsmPrinter &p) const override;
};

/// Register and load all the dialects used by flang.
inline void registerAndLoadDialects(mlir::MLIRContext &ctx) {
  auto registry = ctx.getDialectRegistry();
  // clang-format off
  registry.insert<mlir::AffineDialect,
                  FIROpsDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::acc::OpenACCDialect,
                  mlir::omp::OpenMPDialect,
                  mlir::scf::SCFDialect,
                  mlir::StandardOpsDialect,
                  mlir::vector::VectorDialect>();
  // clang-format on
  registry.loadAll(&ctx);
}

/// Register the standard passes we use. This comes from registerAllPasses(),
/// but is a smaller set since we aren't using many of the passes found there.
inline void registerGeneralPasses() {
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerAffineLoopFusionPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerLoopCoalescingPass();
  mlir::registerStripDebugInfoPass();
  mlir::registerPrintOpStatsPass();
  mlir::registerInlinerPass();
  mlir::registerSCCPPass();
  mlir::registerMemRefDataFlowOptPass();
  mlir::registerSymbolDCEPass();
  mlir::registerLocationSnapshotPass();
  mlir::registerAffinePipelineDataTransferPass();

  mlir::registerAffineVectorizePass();
  mlir::registerAffineLoopUnrollPass();
  mlir::registerAffineLoopUnrollAndJamPass();
  mlir::registerSimplifyAffineStructuresPass();
  mlir::registerAffineLoopInvariantCodeMotionPass();
  mlir::registerAffineLoopTilingPass();
  mlir::registerAffineDataCopyGenerationPass();

  mlir::registerConvertAffineToStandardPass();
}

inline void registerFIRPasses() { registerGeneralPasses(); }

} // namespace fir

#endif // OPTIMIZER_DIALECT_FIRDIALECT_H
