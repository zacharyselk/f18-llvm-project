//===-- OpenMP.cpp -- Open MP directive lowering --------------------------===//
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

#include "flang/Lower/OpenMP.h"
#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Todo.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

static const Fortran::parser::Name *
getDesignatorNameIfDataRef(const Fortran::parser::Designator &designator) {
  const auto *dataRef = std::get_if<Fortran::parser::DataRef>(&designator.u);
  return dataRef ? std::get_if<Fortran::parser::Name>(&dataRef->u) : nullptr;
}

static void genObjectList(const Fortran::parser::OmpObjectList &objectList,
                          Fortran::lower::AbstractConverter &converter,
                          SmallVectorImpl<Value> &operands) {
  for (const auto &ompObject : objectList.v) {
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::Designator &designator) {
              if (const auto *name = getDesignatorNameIfDataRef(designator)) {
                const auto variable = converter.getSymbolAddress(*name->symbol);
                operands.push_back(variable);
              }
            },
            [&](const Fortran::parser::Name &name) {
              const auto variable = converter.getSymbolAddress(*name.symbol);
              operands.push_back(variable);
            }},
        ompObject.u);
  }
}

template <typename Op>
static void createBodyOfOp(Op &op, Fortran::lower::FirOpBuilder &firOpBuilder,
                           mlir::Location &loc) {
  firOpBuilder.createBlock(&op.getRegion());
  auto &block = op.getRegion().back();
  firOpBuilder.setInsertionPointToStart(&block);
  // Ensure the block is well-formed.
  firOpBuilder.create<mlir::omp::TerminatorOp>(loc);
  // Reset the insertion point to the start of the first block.
  firOpBuilder.setInsertionPointToStart(&block);
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPSimpleStandaloneConstruct
                       &simpleStandaloneConstruct) {
  const auto &directive =
      std::get<Fortran::parser::OmpSimpleStandaloneDirective>(
          simpleStandaloneConstruct.t);
  switch (directive.v) {
  default:
    break;
  case llvm::omp::Directive::OMPD_barrier:
    converter.getFirOpBuilder().create<mlir::omp::BarrierOp>(
        converter.getCurrentLocation());
    break;
  case llvm::omp::Directive::OMPD_taskwait:
    converter.getFirOpBuilder().create<mlir::omp::TaskwaitOp>(
        converter.getCurrentLocation());
    break;
  case llvm::omp::Directive::OMPD_taskyield:
    converter.getFirOpBuilder().create<mlir::omp::TaskyieldOp>(
        converter.getCurrentLocation());
    break;
  case llvm::omp::Directive::OMPD_target_enter_data:
    TODO("");
  case llvm::omp::Directive::OMPD_target_exit_data:
    TODO("");
  case llvm::omp::Directive::OMPD_target_update:
    TODO("");
  case llvm::omp::Directive::OMPD_ordered:
    TODO("");
  }
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPStandaloneConstruct &standaloneConstruct) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPSimpleStandaloneConstruct
                  &simpleStandaloneConstruct) {
            genOMP(converter, eval, simpleStandaloneConstruct);
          },
          [&](const Fortran::parser::OpenMPFlushConstruct &flushConstruct) {
            SmallVector<Value, 4> operandRange;
            if (auto &ompObjectList =
                    std::get<std::optional<Fortran::parser::OmpObjectList>>(
                        flushConstruct.t))
              genObjectList(*ompObjectList, converter, operandRange);
            if (auto &flushMemoryClause = std::get<
                    std::optional<Fortran::parser::OmpFlushMemoryClause>>(
                    flushConstruct.t))
              TODO("Handle OmpFlushMemoryClause");
            converter.getFirOpBuilder().create<mlir::omp::FlushOp>(
                converter.getCurrentLocation(), operandRange);
          },
          [&](const Fortran::parser::OpenMPCancelConstruct &cancelConstruct) {
            TODO("");
          },
          [&](const Fortran::parser::OpenMPCancellationPointConstruct
                  &cancellationPointConstruct) { TODO(""); },
      },
      standaloneConstruct.u);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPBlockConstruct &blockConstruct) {
  const auto &beginBlockDirective =
      std::get<Fortran::parser::OmpBeginBlockDirective>(blockConstruct.t);
  const auto &blockDirective =
      std::get<Fortran::parser::OmpBlockDirective>(beginBlockDirective.t);

  auto &firOpBuilder = converter.getFirOpBuilder();
  auto currentLocation = converter.getCurrentLocation();
  llvm::ArrayRef<mlir::Type> argTy;
  if (blockDirective.v == llvm::omp::OMPD_parallel) {

    mlir::Value ifClauseOperand, numThreadsClauseOperand;
    SmallVector<Value, 4> privateClauseOperands, firstprivateClauseOperands,
        sharedClauseOperands, copyinClauseOperands;
    Attribute defaultClauseOperand, procBindClauseOperand;

    const auto &parallelOpClauseList =
        std::get<Fortran::parser::OmpClauseList>(beginBlockDirective.t);
    for (const auto &clause : parallelOpClauseList.v) {
      if (const auto &ifClause =
              std::get_if<Fortran::parser::OmpIfClause>(&clause.u)) {
        auto &expr = std::get<Fortran::parser::ScalarLogicalExpr>(ifClause->t);
        ifClauseOperand = fir::getBase(
            converter.genExprValue(*Fortran::semantics::GetExpr(expr)));
      } else if (const auto &numThreadsClause =
                     std::get_if<Fortran::parser::OmpClause::NumThreads>(
                         &clause.u)) {
        // OMPIRBuilder expects `NUM_THREAD` clause as a `Value`.
        numThreadsClauseOperand = fir::getBase(converter.genExprValue(
            *Fortran::semantics::GetExpr(numThreadsClause->v)));
      } else if (const auto &privateClause =
                     std::get_if<Fortran::parser::OmpClause::Private>(
                         &clause.u)) {
        const Fortran::parser::OmpObjectList &ompObjectList = privateClause->v;
        genObjectList(ompObjectList, converter, privateClauseOperands);
      } else if (const auto &firstprivateClause =
                     std::get_if<Fortran::parser::OmpClause::Firstprivate>(
                         &clause.u)) {
        const Fortran::parser::OmpObjectList &ompObjectList =
            firstprivateClause->v;
        genObjectList(ompObjectList, converter, firstprivateClauseOperands);
      } else if (const auto &sharedClause =
                     std::get_if<Fortran::parser::OmpClause::Shared>(
                         &clause.u)) {
        const Fortran::parser::OmpObjectList &ompObjectList = sharedClause->v;
        genObjectList(ompObjectList, converter, sharedClauseOperands);
      } else if (const auto &copyinClause =
                     std::get_if<Fortran::parser::OmpClause::Copyin>(
                         &clause.u)) {
        const Fortran::parser::OmpObjectList &ompObjectList = copyinClause->v;
        genObjectList(ompObjectList, converter, copyinClauseOperands);
      }
    }
    // Create and insert the operation.
    auto parallelOp = firOpBuilder.create<mlir::omp::ParallelOp>(
        currentLocation, argTy, Value(), numThreads,
        defaultValue.dyn_cast_or_null<StringAttr>(), ValueRange(), ValueRange(),
        ValueRange(), ValueRange(), ValueRange(), ValueRange(),
        procBindValue.dyn_cast_or_null<StringAttr>());
    firOpBuilder.createBlock(&parallelOp.getRegion());
    auto &block = parallelOp.getRegion().back();
    firOpBuilder.setInsertionPointToStart(&block);
    // Ensure the block is well-formed.
    firOpBuilder.create<mlir::omp::TerminatorOp>(currentLocation);

    // Place the insertion point to the start of the first block.
    firOpBuilder.setInsertionPointToStart(&block);
  }
}

void Fortran::lower::genOpenMPConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPConstruct &ompConstruct) {

  std::visit(
      common::visitors{
          [&](const Fortran::parser::OpenMPStandaloneConstruct
                  &standaloneConstruct) {
            genOMP(converter, eval, standaloneConstruct);
          },
          [&](const Fortran::parser::OpenMPSectionsConstruct
                  &sectionsConstruct) { TODO(""); },
          [&](const Fortran::parser::OpenMPLoopConstruct &loopConstruct) {
            TODO("");
          },
          [&](const Fortran::parser::OpenMPBlockConstruct &blockConstruct) {
            genOMP(converter, eval, blockConstruct);
          },
          [&](const Fortran::parser::OpenMPAtomicConstruct &atomicConstruct) {
            TODO("");
          },
          [&](const Fortran::parser::OpenMPCriticalConstruct
                  &criticalConstruct) { TODO(""); },
      },
      ompConstruct.u);
}

void Fortran::lower::genOpenMPEndLoop(
    Fortran::lower::AbstractConverter &, Fortran::lower::pft::Evaluation &,
    const Fortran::parser::OmpEndLoopDirective &) {
  TODO("");
}
