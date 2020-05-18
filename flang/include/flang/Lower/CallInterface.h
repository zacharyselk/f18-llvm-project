//===-- Lower/CallInterface.h -- Procedure call interface ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define procedure call interface to be used both on callee and caller side
// while lowering.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CALLINTERFACE_H
#define FORTRAN_LOWER_CALLINTERFACE_H

#include "flang/Common/reference.h"
#include "mlir/IR/Function.h"

namespace Fortran::semantics {
class Symbol;
}

namespace Fortran::evaluate {
struct ProcedureRef;
class ActualArgument;
namespace characteristics {
struct Procedure;
}
} // namespace Fortran::evaluate

namespace Fortran::lower {
class AbstractConverter;
class SymMap;
namespace pft {
struct FunctionLikeUnit;
}

class CallerInterface;
class CalleeInterface;
template <typename T>
struct PassedEntityTypes {};
template <>
struct PassedEntityTypes<CallerInterface> {
  using FortranEntity = const Fortran::evaluate::ActualArgument *;
  using FirValue = int;
};
template <>
struct PassedEntityTypes<CalleeInterface> {
  using FortranEntity = common::Reference<const semantics::Symbol>;
  using FirValue = mlir::Value;
};

template <typename T>
class CallInterface {
public:
  /// Different properties of an entity that can be passed/returned.
  enum class Property {
    BaseAddress,
    BoxChar,
    CharAddress,
    CharLength,
    Descriptor,
    Value
  };
  /// Enum the different ways an entity can be passed-by
  enum class PassEntityBy {
    BaseAddress,
    BoxChar,
    Descriptor,
    AddressAndLength,
    Value
  };

  using FortranEntity = typename PassedEntityTypes<T>::FortranEntity;
  using FirValue = typename PassedEntityTypes<T>::FirValue;
  struct FirPlaceHolder {
    static constexpr int resultEntityPosition = -1;
    mlir::Type type;
    /// Position of related FortranPassInfo in fortranArguments.
    int FortranPosition;
    /// Indicate property of the entity at FortranPosition that must be passed
    /// through this argument.
    Property property;
  };

  struct PassedEntity {
    PassEntityBy passBy;
    FortranEntity entity;
    FirValue firArgument;
    FirValue firLength; /* only for AddressAndLength */
  };

  mlir::FuncOp getFuncOp() const { return func; }
  std::size_t getNumFIRArguments() const { return inputs.size(); }
  std::size_t getNumFIRResults() const { return outputs.size(); }
  llvm::SmallVector<mlir::Type, 1> getResultType() const;

  llvm::ArrayRef<PassedEntity> getPassedArguments() const {
    return passedArguments;
  }
  std::optional<PassedEntity> getPassedResult() const { return passedResult; }

private:
  T &impl() { return *static_cast<T *>(this); }
  void
  buildImplicitInterface(const Fortran::evaluate::characteristics::Procedure &);
  void
  buildExplicitInterface(const Fortran::evaluate::characteristics::Procedure &);
  mlir::FunctionType genFunctionType() const;

  llvm::SmallVector<FirPlaceHolder, 1> outputs;
  llvm::SmallVector<FirPlaceHolder, 4> inputs;
  mlir::FuncOp func;

  llvm::SmallVector<PassedEntity, 4> passedArguments;
  std::optional<PassedEntity> passedResult;

protected:
  CallInterface(Fortran::lower::AbstractConverter &c) : converter{c} {}
  void init();
  Fortran::lower::AbstractConverter &converter;
};

//===----------------------------------------------------------------------===//
// Caller side interface
//===----------------------------------------------------------------------===//

class CallerInterface : public CallInterface<CallerInterface> {
public:
  CallerInterface(const Fortran::evaluate::ProcedureRef &p,
                  Fortran::lower::AbstractConverter &c)
      : CallInterface{c}, procRef{p} {
    init();
  }
  /// CRTP callbacks
  bool hasAlternateReturns() const;
  std::string getMangledName() const;
  mlir::Location getCalleeLocation() const;
  Fortran::evaluate::characteristics::Procedure characterize() const;
  const Fortran::evaluate::ProcedureRef &getCallDescription() const {
    return procRef;
  };
  bool isMainProgram() const { return false; }

  /// Specific to Caller
  void placeInput(const PassedEntity &passedEntity, mlir::Value arg);
  void placeAddressAndLengthInput(const PassedEntity &passedEntity,
                                  mlir::Value addr, mlir::Value len);
  const llvm::SmallVector<mlir::Value, 3> &getInputs() const {
    assert(verifyActualInputs() && "lowered arguments are incomplete");
    return actualInputs;
  }

private:
  bool verifyActualInputs() const;
  const Fortran::evaluate::ProcedureRef &procRef;
  llvm::SmallVector<mlir::Value, 3> actualInputs;
};

//===----------------------------------------------------------------------===//
// Callee side interface
//===----------------------------------------------------------------------===//

class CalleeInterface : public CallInterface<CalleeInterface> {
public:
  CalleeInterface(Fortran::lower::pft::FunctionLikeUnit &f,
                  Fortran::lower::AbstractConverter &c)
      : CallInterface{c}, funit{f} {
    init();
  }
  bool hasAlternateReturns() const;
  std::string getMangledName() const;
  mlir::Location getCalleeLocation() const;
  Fortran::evaluate::characteristics::Procedure characterize() const;
  bool isMainProgram() const;
  Fortran::lower::pft::FunctionLikeUnit &getCallDescription() const {
    return funit;
  };

private:
  Fortran::lower::pft::FunctionLikeUnit &funit;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_FIRBUILDER_H
