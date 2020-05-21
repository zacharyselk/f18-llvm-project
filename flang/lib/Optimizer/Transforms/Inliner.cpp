//===-- Inliner.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<bool>
    enableInliningPass("enable-inlining",
                       llvm::cl::desc("enable FIR inlining pass"),
                       llvm::cl::init(false));

/// Should we inline the callable `op` into region `reg`?
bool fir::canLegallyInline(mlir::Operation *op, mlir::Region *reg,
                           mlir::BlockAndValueMapping &map) {
  // TODO: this should _not_ just return true.
  return true;
}

/// Should an inlining pass be instantiated?
bool fir::inlinerIsEnabled() { return enableInliningPass; }

/// Instantiate an inlining pass. NB: If inlining is disabled, the compiler will
/// abort if an inlining pass is instantiated.
std::unique_ptr<mlir::Pass> fir::createInlinerPass() {
  if (enableInliningPass)
    return mlir::createInlinerPass();
  llvm::report_fatal_error("inlining is disabled");
}
