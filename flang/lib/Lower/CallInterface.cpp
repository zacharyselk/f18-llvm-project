//===-- CallInterface.cpp -- Procedure call interface ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/CallInterface.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"

//===----------------------------------------------------------------------===//
// Caller side interface implementation
//===----------------------------------------------------------------------===//

bool Fortran::lower::CallerInterface::hasAlternateReturns() const {
  return procRef.HasAlternateReturns();
}

std::string Fortran::lower::CallerInterface::getMangledName() const {
  const auto &proc = procRef.proc();
  if (const auto *symbol = proc.GetSymbol())
    return converter.mangleName(*symbol);
  assert(proc.GetSpecificIntrinsic() &&
         "expected intrinsic procedure in designator");
  return proc.GetName();
}

mlir::Location Fortran::lower::CallerInterface::getCalleeLocation() const {
  const auto &proc = procRef.proc();
  if (const auto *symbol = proc.GetSymbol()) {
    const auto &details = symbol->get<Fortran::semantics::ProcEntityDetails>();
    if (const auto *interfaceSymbol = details.interface().symbol())
      symbol = interfaceSymbol;
    return converter.genLocation(symbol->name());
  }
  // Unknown location for intrinsics.
  return converter.genLocation();
}

Fortran::evaluate::characteristics::Procedure
Fortran::lower::CallerInterface::characterize() const {
  // FIXME: get actual IntrinsicProcTable.
  auto characteristic =
      Fortran::evaluate::characteristics::Procedure::Characterize(
          procRef,
          *static_cast<const Fortran::evaluate::IntrinsicProcTable *>(nullptr));
  assert(characteristic && "Fail to get characteristic from procRef");
  return *characteristic;
}

void Fortran::lower::CallerInterface::placeInput(
    const PassedEntity &passedEntity, mlir::Value arg) {
  assert(static_cast<int>(actualInputs.size()) >= passedEntity.firArgument &&
         passedEntity.firArgument >= 0 &&
         passedEntity.passBy != CallInterface::PassEntityBy::AddressAndLength &&
         "bad arg position");
  actualInputs[passedEntity.firArgument] = arg;
}

void Fortran::lower::CallerInterface::placeAddressAndLengthInput(
    const PassedEntity &passedEntity, mlir::Value addr, mlir::Value len) {
  assert(static_cast<int>(actualInputs.size()) >= passedEntity.firArgument &&
         static_cast<int>(actualInputs.size()) >= passedEntity.firLength &&
         passedEntity.firArgument >= 0 && passedEntity.firLength >= 0 &&
         passedEntity.passBy == CallInterface::PassEntityBy::AddressAndLength &&
         "bad arg position");
  actualInputs[passedEntity.firArgument] = addr;
  actualInputs[passedEntity.firLength] = len;
}

bool Fortran::lower::CallerInterface::verifyActualInputs() const {
  if (getNumFIRArguments() != actualInputs.size())
    return false;
  for (auto arg : actualInputs) {
    if (!arg)
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Callee side interface implementation
//===----------------------------------------------------------------------===//

bool Fortran::lower::CalleeInterface::hasAlternateReturns() const {
  return !funit.isMainProgram() &&
         Fortran::semantics::HasAlternateReturns(funit.getSubprogramSymbol());
}

std::string Fortran::lower::CalleeInterface::getMangledName() const {
  return funit.isMainProgram()
             ? fir::NameUniquer::doProgramEntry().str()
             : converter.mangleName(funit.getSubprogramSymbol());
}

mlir::Location Fortran::lower::CalleeInterface::getCalleeLocation() const {
  // FIXME: do NOT use unknown for the anonymous PROGRAM case. We probably
  // should just stash the location in the funit regardless.
  return converter.genLocation(funit.getStartingSourceLoc());
}

Fortran::evaluate::characteristics::Procedure
Fortran::lower::CalleeInterface::characterize() const {
  // FIXME: get actual IntrinsicProcTable.
  auto characteristic =
      Fortran::evaluate::characteristics::Procedure::Characterize(
          funit.getSubprogramSymbol(),
          *static_cast<const Fortran::evaluate::IntrinsicProcTable *>(nullptr));
  assert(characteristic && "Fail to get characteristic from symbol");
  return *characteristic;
}

bool Fortran::lower::CalleeInterface::isMainProgram() const {
  return funit.isMainProgram();
}

//===----------------------------------------------------------------------===//
// CallInterface implementation: this part is common to both caller and caller
// sides.
//===----------------------------------------------------------------------===//

template <typename T>
void Fortran::lower::CallInterface<T>::init() {
  if (!impl().isMainProgram()) {
    auto characteristic = impl().characterize();
    if (characteristic.CanBeCalledViaImplicitInterface())
      buildImplicitInterface(characteristic);
    else
      buildExplicitInterface(characteristic);
  }
  // No input/output for main program

  auto name = impl().getMangledName();
  auto module = converter.getModuleOp();
  func = Fortran::lower::FirOpBuilder::getNamedFunction(module, name);
  if (!func) {
    mlir::Location loc = impl().getCalleeLocation();
    mlir::FunctionType ty = genFunctionType();
    func = Fortran::lower::FirOpBuilder::createFunction(loc, module, name, ty);
  }
  // TODO: re-map fir inputs to passedArguments and results

  // for (const auto& pair: llvm::zip(inputs, func.front.getArguments())) {
  //   const auto& placeHolder = std::get<0>(pair)
  //   if ()
  // }
}

/// Helper to access ActualArgument/Symbols
static const Fortran::evaluate::ActualArguments &
getEntityContainer(const Fortran::evaluate::ProcedureRef &proc) {
  return proc.arguments();
}

static const std::vector<Fortran::semantics::Symbol *> &
getEntityContainer(Fortran::lower::pft::FunctionLikeUnit &funit) {
  return funit.getSubprogramSymbol()
      .get<Fortran::semantics::SubprogramDetails>()
      .dummyArgs();
}

static const Fortran::evaluate::ActualArgument *getDataObjectEntity(
    const std::optional<Fortran::evaluate::ActualArgument> &arg) {
  if (arg)
    return &*arg;
  return nullptr;
}

static const Fortran::semantics::Symbol &
getDataObjectEntity(const Fortran::semantics::Symbol *arg) {
  assert(arg && "expect symbol for data object entity");
  return *arg;
}

static const Fortran::evaluate::ActualArgument *
getResultEntity(const Fortran::evaluate::ProcedureRef &) {
  return nullptr;
}

static const Fortran::semantics::Symbol &
getResultEntity(Fortran::lower::pft::FunctionLikeUnit &funit) {
  const auto &details =
      funit.getSubprogramSymbol().get<Fortran::semantics::SubprogramDetails>();
  return details.result();
}

template <typename T>
struct EmptyFirValue {};
template <>
struct EmptyFirValue<Fortran::lower::CalleeInterface> {
  static mlir::Value get() { return {}; }
};
template <>
struct EmptyFirValue<Fortran::lower::CallerInterface> {
  static int get() { return -1; }
};

template <typename T>
void Fortran::lower::CallInterface<T>::buildImplicitInterface(
    const Fortran::evaluate::characteristics::Procedure &procedure) {
  auto &mlirContext = converter.getMLIRContext();
  FirValue emptyValue = EmptyFirValue<T>::get();
  // Handle result
  auto resultPosition = FirPlaceHolder::resultEntityPosition;
  if (const auto &result = procedure.functionResult) {
    if (result->IsProcedurePointer())
      // TODO
      llvm_unreachable("procedure pointer result not yet handled");
    const auto *typeAndShape = result->GetTypeAndShape();
    assert(typeAndShape && "expect type for non proc pointer result");
    auto dynamicType = typeAndShape->type();
    // Character result allocated by caller and passed has hidden arguments
    if (dynamicType.category() == common::TypeCategory::Character) {
      passedResult = PassedEntity{PassEntityBy::AddressAndLength,
                                  getResultEntity(impl().getCallDescription()),
                                  emptyValue, emptyValue};
      // FIXME: there is a bunch of type helpers in the FiROpBuilder that are
      // not accessible here because there is no live FirOpBuilder yet on the
      // callee side. Most of these helpers only really need a MLIRContext, it
      // would be nice to allow using them here.
      auto lenTy = mlir::IndexType::get(&mlirContext);
      auto charRefTy = fir::ReferenceType::get(
          fir::CharacterType::get(&mlirContext, dynamicType.kind()));
      auto boxCharTy = fir::BoxCharType::get(&mlirContext, dynamicType.kind());
      inputs.emplace_back(
          FirPlaceHolder{charRefTy, resultPosition, Property::CharAddress});
      inputs.emplace_back(
          FirPlaceHolder{lenTy, resultPosition, Property::CharLength});
      /// For now, still also return it by boxchar
      outputs.emplace_back(
          FirPlaceHolder{boxCharTy, resultPosition, Property::CharLength});
    } else {
      // All result other than characters are simply returned by value in
      // implicit interfaces
      auto mlirType =
          converter.genType(dynamicType.category(), dynamicType.kind());
      outputs.emplace_back(
          FirPlaceHolder{mlirType, resultPosition, Property::Value});
    }
  } else if (impl().hasAlternateReturns()) {
    outputs.emplace_back(FirPlaceHolder{mlir::IndexType::get(&mlirContext),
                                        resultPosition, Property::Value});
  }
  // Handle arguments
  const auto &argumentEntities =
      getEntityContainer(impl().getCallDescription());
  for (const auto &pair :
       llvm::zip(procedure.dummyArguments, argumentEntities)) {
    const auto &dummy = std::get<0>(pair);
    std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::characteristics::DummyDataObject
                    &obj) {
              auto dynamicType = obj.type.type();
              const auto &entity = getDataObjectEntity(std::get<1>(pair));
              if (dynamicType.category() ==
                  Fortran::common::TypeCategory::Character) {
                auto boxCharTy =
                    fir::BoxCharType::get(&mlirContext, dynamicType.kind());
                inputs.emplace_back(FirPlaceHolder{
                    boxCharTy, passedArguments.size(), Property::BoxChar});
                passedArguments.emplace_back(
                    PassedEntity{PassEntityBy::BoxChar, entity, emptyValue});
              } else {
                auto refType = fir::ReferenceType::get(converter.genType(
                    dynamicType.category(), dynamicType.kind()));
                inputs.emplace_back(FirPlaceHolder{
                    refType, passedArguments.size(), Property::BaseAddress});
                passedArguments.emplace_back(PassedEntity{
                    PassEntityBy::BaseAddress, entity, emptyValue});
              }
            },
            [&](const Fortran::evaluate::characteristics::DummyProcedure &) {
              // TODO
              llvm_unreachable("dummy procedure pointer not yet handled");
            },
            [&](const Fortran::evaluate::characteristics::AlternateReturn &) {
              // do nothing
            },
        },
        dummy.u);
  }
}

template <typename T>
void Fortran::lower::CallInterface<T>::buildExplicitInterface(
    const Fortran::evaluate::characteristics::Procedure &) {
  llvm_unreachable("Explicit interface lowering TODO");
}

template <typename T>
mlir::FunctionType Fortran::lower::CallInterface<T>::genFunctionType() const {
  llvm::SmallVector<mlir::Type, 1> returnTys;
  llvm::SmallVector<mlir::Type, 4> inputTys;
  for (const auto &placeHolder : outputs)
    returnTys.emplace_back(placeHolder.type);
  for (const auto &placeHolder : inputs)
    inputTys.emplace_back(placeHolder.type);
  return mlir::FunctionType::get(inputTys, returnTys,
                                 &converter.getMLIRContext());
}

template <typename T>
llvm::SmallVector<mlir::Type, 1>
Fortran::lower::CallInterface<T>::getResultType() const {
  llvm::SmallVector<mlir::Type, 1> types;
  for (const auto &out : outputs)
    types.emplace_back(out.type);
  return types;
}

template class Fortran::lower::CallInterface<Fortran::lower::CalleeInterface>;
template class Fortran::lower::CallInterface<Fortran::lower::CallerInterface>;
