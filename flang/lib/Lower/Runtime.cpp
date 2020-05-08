//===-- Runtime.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Runtime.h"
#include "RTBuilder.h"
#include "flang/Lower/FIRBuilder.h"
#include "llvm/ADT/SmallVector.h"

#include "../runtime/stop.h"

struct RuntimeEntry {
  using Key = Fortran::lower::RuntimeEntryCode;
  Key key;
  llvm::StringRef symbol;
  Fortran::lower::FuncTypeBuilderFunc typeBuilder;
};

#define QUOTE_HELPER(X) #X
#define QUOTE(X) QUOTE_HELPER(X)
#define QUOTE_RTNAME(X) QUOTE(RTNAME(X))

#define MakeEntry(X)                                                           \
  {                                                                            \
    Fortran::lower::RuntimeEntryCode::X, QUOTE_RTNAME(X),                      \
        Fortran::lower::RuntimeTableKey<decltype(RTNAME(X))>::getTypeModel()   \
  }

static constexpr RuntimeEntry runtimeTable[]{
    MakeEntry(StopStatement), MakeEntry(StopStatementText),
    MakeEntry(FailImageStatement), MakeEntry(ProgramEndStatement)};

static constexpr Fortran::lower::StaticMultimapView runtimeMap(runtimeTable);

mlir::FuncOp
Fortran::lower::genRuntimeFunction(RuntimeEntryCode code,
                                   Fortran::lower::FirOpBuilder &builder) {
  auto entry = runtimeMap.find(code);
  assert(entry != runtimeMap.end());
  auto func = builder.getNamedFunction(entry->symbol);
  if (func)
    return func;
  auto funTy = entry->typeBuilder(builder.getContext());
  func = builder.createFunction(entry->symbol, funTy);
  func.setAttr("fir.runtime", builder.getUnitAttr());
  return func;
}
