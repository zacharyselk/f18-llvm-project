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

#include "mlir/IR/Function.h"

namespace Fortran::semantics {
class Symbol;
}

namespace Fortran::lower {
class SymMap;
namespace pft {
struct FunctionLikeUnit;
}

class CallInterface {
public:
  mlir::FuncOp getFuncOp() const { return func; }
  int getNumFIRArguments() const { return inputs.size(); }
  int getNumFIRResults() const { return outputs.size(); }
  llvm::StringRef getMangledName() const { return mangledName; }
  llvm::SmallVector<mlir::Type, 3> getResultType() const;
  /// Different properties of an entity that can be passed/returned.
  enum class Property {
    BaseAddress,
    BoxChar,
    CharAddress,
    CharLength,
    Descriptor,
    Value
  };
  /// Indicate position is used to pass information about result as result.
  static constexpr int resultPosition = -1;
  struct PlaceHolder {
    mlir::Type type;
    /// Position of symbol in callee or ActualArgument in caller.
    int FortranPosition;
    /// Indicate property of the entity at FortranPosition that must be passed
    /// through this argument.
    Property property;
  };

protected:
  llvm::SmallVector<PlaceHolder, 1> outputs;
  llvm::SmallVector<PlaceHolder, 4> inputs;
  mlir::FuncOp func;
  bool calleeMustPassResult = false;
  std::string mangledName;
};

//===----------------------------------------------------------------------===//
// Callee side interface
//===----------------------------------------------------------------------===//

class CalleeInterface : public CallInterface {
public:
  CalleeInterface(Fortran::lower::pft::FunctionLikeUnit &f);
  void mapDummyAndResults(Fortran::lower::SymMap &map);
  bool hasAlternateReturns() const;

private:
  Fortran::lower::pft::FunctionLikeUnit &funit;
};

//===----------------------------------------------------------------------===//
// Caller side interface
//===----------------------------------------------------------------------===//

// Enum the different ways an entity can be passed-by
enum class PassEntityBy {
  BaseAddress,
  BoxChar,
  Descriptor,
  AddressAndLength,
  Value
};
struct PassInfo {
  template <typename T>
  void placeArgument(mlir::Value arg, T &args) const {
    assert(static_cast<int>(args.size()) >= firArgumentPosition &&
           "bad arg position");
    args[firArgumentPosition] = arg;
  }
  template <typename T>
  void placeAddressAndLength(mlir::Value addr, mlir::Value len, T &args) const {
    assert(static_cast<int>(args.size()) >= firArgumentPosition &&
           static_cast<int>(args.size()) >= firLengthPosition &&
           "bad arg position");
    args[firArgumentPosition] = addr;
    args[firLengthPosition] = addr;
  }
  PassEntityBy passBy;

private:
  int firArgumentPosition;
  int firLengthPosition; /* only for AddressAndLength */
  // Caller action (i.e, make it contiguous....) could be added here as a set.
};

struct PassedActual : PassInfo {
  const std::optional<Fortran::evaluate::ActualArgument> &actual;
};

class CallerInterface : public CallInterface {
public:
  CallerInterface(const Fortran::evaluate::ProcedureRef &f);
  mlir::FuncOp getFuncOp() const { return func; }
  llvm::ArrayRef<PassedActual> getPassedActuals() const {
    return passedActuals;
  }
  std::optional<PassInfo> getPassResult() const { return passResult; }
  bool hasAlternateReturns() const;

private:
  const Fortran::evaluate::ProcedureRef &procRef;
  llvm::SmallVector<PassedActual, 3> passedActuals;
  std::optional<PassInfo> passResult;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_FIRBUILDER_H
