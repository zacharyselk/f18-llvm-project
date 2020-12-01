//===-- ConvertType.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertType.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/shape.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Todo.h"
#include "flang/Lower/Utils.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

//===--------------------------------------------------------------------===//
// Intrinsic type translation helpers
//===--------------------------------------------------------------------===//

static mlir::Type genRealType(mlir::MLIRContext *context, int kind) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(
          Fortran::common::TypeCategory::Real, kind)) {
    switch (kind) {
    case 2:
      return mlir::FloatType::getF16(context);
    case 3:
      return mlir::FloatType::getBF16(context);
    case 4:
      return mlir::FloatType::getF32(context);
    case 8:
      return mlir::FloatType::getF64(context);
    case 10:
      return fir::RealType::get(context, 10);
    case 16:
      return fir::RealType::get(context, 16);
    }
  }
  llvm_unreachable("REAL type translation not implemented");
}

template <int KIND>
int getIntegerBits() {
  return Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer,
                                 KIND>::Scalar::bits;
}
static mlir::Type genIntegerType(mlir::MLIRContext *context, int kind) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(
          Fortran::common::TypeCategory::Integer, kind)) {
    switch (kind) {
    case 1:
      return mlir::IntegerType::get(getIntegerBits<1>(), context);
    case 2:
      return mlir::IntegerType::get(getIntegerBits<2>(), context);
    case 4:
      return mlir::IntegerType::get(getIntegerBits<4>(), context);
    case 8:
      return mlir::IntegerType::get(getIntegerBits<8>(), context);
    case 16:
      return mlir::IntegerType::get(getIntegerBits<16>(), context);
    }
  }
  llvm_unreachable("INTEGER type translation not implemented");
}

static mlir::Type genLogicalType(mlir::MLIRContext *context, int KIND) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(
          Fortran::common::TypeCategory::Logical, KIND))
    return fir::LogicalType::get(context, KIND);
  return {};
}

static mlir::Type genCharacterType(
    mlir::MLIRContext *context, int KIND,
    Fortran::lower::LenParameterTy len = fir::CharacterType::unknownLen()) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(
          Fortran::common::TypeCategory::Character, KIND))
    return fir::CharacterType::get(context, KIND, len);
  return {};
}

static mlir::Type genComplexType(mlir::MLIRContext *context, int KIND) {
  if (Fortran::evaluate::IsValidKindOfIntrinsicType(
          Fortran::common::TypeCategory::Complex, KIND))
    return fir::ComplexType::get(context, KIND);
  return {};
}

static mlir::Type
genFIRType(mlir::MLIRContext *context, Fortran::common::TypeCategory tc,
           int kind,
           llvm::ArrayRef<Fortran::lower::LenParameterTy> lenParameters) {
  switch (tc) {
  case Fortran::common::TypeCategory::Real:
    return genRealType(context, kind);
  case Fortran::common::TypeCategory::Integer:
    return genIntegerType(context, kind);
  case Fortran::common::TypeCategory::Complex:
    return genComplexType(context, kind);
  case Fortran::common::TypeCategory::Logical:
    return genLogicalType(context, kind);
  case Fortran::common::TypeCategory::Character:
    if (!lenParameters.empty())
      return genCharacterType(context, kind, lenParameters[0]);
    else
      return genCharacterType(context, kind);
  default:
    break;
  }
  llvm_unreachable("unhandled type category");
}

//===--------------------------------------------------------------------===//
// Symbol and expression type translation
//===--------------------------------------------------------------------===//

/// TypeBuilder translates expression and symbol type taking into account
/// their shape and length parameters. For symbols, attributes such as
/// ALLOCATABLE or POINTER are reflected in the fir type.
/// It uses evaluate::DynamicType and evaluate::Shape when possible to
/// avoid re-implementing type/shape analysis here.
/// Do not use the FirOpBuilder from the AbstractConverter to get fir/mlir types
/// since it is not guaranteed to exist yet when we lower types.
namespace {
struct TypeBuilder {

  TypeBuilder(Fortran::lower::AbstractConverter &converter)
      : converter{converter}, context{&converter.getMLIRContext()} {}

  mlir::Type genExprType(const Fortran::lower::SomeExpr &expr) {
    auto dynamicType = expr.GetType();
    if (!dynamicType)
      return genTypelessExprType(expr);
    auto category = dynamicType->category();
    if (category == Fortran::common::TypeCategory::Derived)
      TODO("derived types lowering");
    auto shapeExpr =
        Fortran::evaluate::GetShape(converter.getFoldingContext(), expr);
    if (!shapeExpr)
      TODO("Assumed rank expression type lowering");

    // LOGICAL, INTEGER, REAL, COMPLEX, CHARACTER
    llvm::SmallVector<Fortran::lower::LenParameterTy, 2> params;
    translateLenParameters(params, category, expr);
    auto baseType = genFIRType(context, category, dynamicType->kind(), params);
    fir::SequenceType::Shape shape;
    translateShape(shape, std::move(*shapeExpr));
    if (!shape.empty())
      return fir::SequenceType::get(shape, baseType);
    return baseType;
  }

  template <typename A>
  void translateShape(A &shape, Fortran::evaluate::Shape &&shapeExpr) {
    for (auto extentExpr : shapeExpr) {
      auto extent = fir::SequenceType::getUnknownExtent();
      if (auto constantExtent = toInt64(std::move(extentExpr)))
        extent = *constantExtent;
      shape.push_back(extent);
    }
  }

  template <typename A>
  std::optional<std::int64_t> toInt64(A &&expr) {
    return Fortran::evaluate::ToInt64(Fortran::evaluate::Fold(
        converter.getFoldingContext(), std::move(expr)));
  }

  mlir::Type genTypelessExprType(const Fortran::lower::SomeExpr &expr) {
    return std::visit(
        Fortran::common::visitors{
            [&](const Fortran::evaluate::BOZLiteralConstant &) -> mlir::Type {
              return mlir::NoneType::get(context);
            },
            [&](const Fortran::evaluate::NullPointer &) -> mlir::Type {
              return fir::ReferenceType::get(mlir::NoneType::get(context));
            },
            [&](const Fortran::evaluate::ProcedureDesignator &proc)
                -> mlir::Type {
              return Fortran::lower::translateSignature(proc, converter);
            },
            [&](const Fortran::evaluate::ProcedureRef &) -> mlir::Type {
              return mlir::NoneType::get(context);
            },
            [](const auto &x) -> mlir::Type {
              using T = std::decay_t<decltype(x)>;
              static_assert(!Fortran::common::HasMember<
                                T, Fortran::evaluate::TypelessExpression>,
                            "missing typeless expr handling in type lowering");
              llvm::report_fatal_error("not a typeless expression");
            },
        },
        expr.u);
  }

  mlir::Type genSymbolType(const Fortran::semantics::Symbol &symbol,
                           bool isAlloc = false, bool isPtr = false) {
    auto loc = converter.genLocation(symbol.name());
    mlir::Type ty;
    // If the symbol is not the same as the ultimate one (i.e, it is host or use
    // associated), all the symbol properties are the ones of the ultimate
    // symbol but the volatile and asynchronous attributes that may differ. To
    // avoid issues with helper functions that would not follow association
    // links, the fir type is built based on the ultimate symbol. This relies
    // on the fact volatile and asynchronous are not reflected in fir types.
    const auto &ultimate = symbol.GetUltimate();
    if (auto *type{ultimate.GetType()}) {
      if (auto *tySpec{type->AsIntrinsic()}) {
        int kind = toInt64(Fortran::common::Clone(tySpec->kind())).value();
        llvm::SmallVector<Fortran::lower::LenParameterTy, 2> params;
        translateLenParameters(params, tySpec->category(), ultimate);
        ty = genFIRType(context, tySpec->category(), kind, params);
      } else if (auto *tySpec = type->AsDerived()) {
        std::vector<std::pair<std::string, mlir::Type>> ps;
        std::vector<std::pair<std::string, mlir::Type>> cs;
        auto &symbol = tySpec->typeSymbol();
        auto rec = fir::RecordType::get(context, toStringRef(symbol.name()));
        // TODO: use Fortran::semantics::ComponentIterator to go through
        // components. or use similar mechanism. We probably want to go through
        // the Ordered components.
        TODO("lower derived type to fir types");
        rec.finalize(ps, cs);
        ty = rec;
      } else {
        mlir::emitError(loc, "symbol's type must have a type spec");
        return {};
      }
    } else {
      mlir::emitError(loc, "symbol must have a type");
      return {};
    }
    if (ultimate.IsObjectArray()) {
      auto shapeExpr = Fortran::evaluate::GetShapeHelper{
          converter.getFoldingContext()}(ultimate);
      if (!shapeExpr)
        TODO("assumed rank symbol type lowering");
      fir::SequenceType::Shape shape;
      translateShape(shape, std::move(*shapeExpr));
      ty = fir::SequenceType::get(shape, ty);
    }

    if (isPtr || Fortran::semantics::IsPointer(ultimate))
      ty = fir::PointerType::get(ty);
    else if (isAlloc || Fortran::semantics::IsAllocatable(ultimate))
      ty = fir::HeapType::get(ty);
    return ty;
  }

  // To get the character length from a symbol, make an fold a designator for
  // the symbol to cover the case where the symbol is an assumed length named
  // constant and its length comes from its init expression length.
  template <int Kind>
  fir::SequenceType::Extent
  getCharacterLengthHelper(const Fortran::semantics::Symbol &symbol) {
    using TC =
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Character, Kind>;
    auto designator = Fortran::evaluate::Fold(
        converter.getFoldingContext(),
        Fortran::evaluate::Expr<TC>{Fortran::evaluate::Designator<TC>{symbol}});
    if (auto len = toInt64(std::move(designator.LEN())))
      return *len;
    return fir::SequenceType::getUnknownExtent();
  }

  template <typename T>
  void translateLenParameters(
      llvm::SmallVectorImpl<Fortran::lower::LenParameterTy> &params,
      Fortran::common::TypeCategory category, const T &exprOrSym) {
    if (category == Fortran::common::TypeCategory::Character)
      params.push_back(getCharacterLength(exprOrSym));
    else if (category == Fortran::common::TypeCategory::Derived)
      TODO("lowering derived type length parameters");
    return;
  }
  Fortran::lower::LenParameterTy
  getCharacterLength(const Fortran::semantics::Symbol &symbol) {
    auto *type = symbol.GetType();
    if (!type ||
        type->category() != Fortran::semantics::DeclTypeSpec::Character ||
        !type->AsIntrinsic())
      llvm::report_fatal_error("not a character symbol");
    int kind =
        toInt64(Fortran::common::Clone(type->AsIntrinsic()->kind())).value();
    switch (kind) {
    case 1:
      return getCharacterLengthHelper<1>(symbol);
    case 2:
      return getCharacterLengthHelper<2>(symbol);
    case 4:
      return getCharacterLengthHelper<4>(symbol);
    }
    llvm_unreachable("unknown character kind");
  }
  Fortran::lower::LenParameterTy
  getCharacterLength(const Fortran::lower::SomeExpr &expr) {
    // Do not use dynamic type length here. We would miss constant
    // lengths opportunities because dynamic type only has the length
    // if it comes from a declaration.
    auto charExpr =
        std::get<Fortran::evaluate::Expr<Fortran::evaluate::SomeCharacter>>(
            expr.u);
    if (auto constantLen = toInt64(charExpr.LEN()))
      return *constantLen;
    return fir::SequenceType::getUnknownExtent();
  }

  mlir::Type genVariableType(const Fortran::lower::pft::Variable &var) {
    return genSymbolType(var.getSymbol(), var.isHeapAlloc(), var.isPointer());
  }

  Fortran::lower::AbstractConverter &converter;
  mlir::MLIRContext *context;
};
} // namespace

mlir::Type Fortran::lower::getFIRType(mlir::MLIRContext *context,
                                      Fortran::common::TypeCategory tc,
                                      int kind,
                                      llvm::ArrayRef<LenParameterTy> params) {
  return genFIRType(context, tc, kind, params);
}

mlir::Type Fortran::lower::translateSomeExprToFIRType(
    Fortran::lower::AbstractConverter &converter, const SomeExpr &expr) {
  return TypeBuilder{converter}.genExprType(expr);
}

mlir::Type Fortran::lower::translateSymbolToFIRType(
    Fortran::lower::AbstractConverter &converter, const SymbolRef symbol) {
  return TypeBuilder{converter}.genSymbolType(symbol);
}

mlir::Type Fortran::lower::translateVariableToFIRType(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable &var) {
  return TypeBuilder{converter}.genVariableType(var);
}

mlir::Type Fortran::lower::convertReal(mlir::MLIRContext *context, int kind) {
  return genRealType(context, kind);
}
