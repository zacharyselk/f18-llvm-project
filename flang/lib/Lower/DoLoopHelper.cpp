//===-- DoLoopHelper.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/DoLoopHelper.h"

//===----------------------------------------------------------------------===//
// DoLoopHelper implementation
//===----------------------------------------------------------------------===//

void Fortran::lower::DoLoopHelper::createLoop(
    mlir::Value lb, mlir::Value ub, mlir::Value step,
    const BodyGenerator &bodyGenerator) {
  auto lbi = builder.convertToIndexType(lb);
  auto ubi = builder.convertToIndexType(ub);
  assert(step && "step must be an actual Value");
  auto inc = builder.convertToIndexType(step);
  auto loop = builder.create<fir::LoopOp>(loc, lbi, ubi, inc);
  auto insertPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(loop.getBody());
  auto index = loop.getInductionVar();
  bodyGenerator(builder, index);
  builder.restoreInsertionPoint(insertPt);
}

void Fortran::lower::DoLoopHelper::createLoop(
    mlir::Value lb, mlir::Value ub, const BodyGenerator &bodyGenerator) {
  createLoop(lb, ub, builder.createIntegerConstant(builder.getIndexType(), 1),
             bodyGenerator);
}

void Fortran::lower::DoLoopHelper::createLoop(
    mlir::Value count, const BodyGenerator &bodyGenerator) {
  auto indexType = builder.getIndexType();
  auto zero = builder.createIntegerConstant(indexType, 0);
  auto one = builder.createIntegerConstant(count.getType(), 1);
  auto up = builder.create<mlir::SubIOp>(loc, count, one);
  createLoop(zero, up, one, bodyGenerator);
}
