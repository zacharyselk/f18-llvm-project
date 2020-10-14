//===-- CharacterExpr.cpp -------------------------------------------------===//
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

#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/DoLoopHelper.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/Todo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-character"

//===----------------------------------------------------------------------===//
// CharacterExprHelper implementation
//===----------------------------------------------------------------------===//

template <bool checkForScalar>
static fir::CharacterType recoverCharacterType(mlir::Type type) {
  if (auto boxType = type.dyn_cast<fir::BoxCharType>())
    return boxType.getEleTy();
  if (auto refType = type.dyn_cast<fir::ReferenceType>())
    type = refType.getEleTy();
  if (auto seqType = type.dyn_cast<fir::SequenceType>()) {
    // In a context where `type` may be a sequence, we want to opt out of this
    // assertion by setting `checkForScalar` to `false`.
    assert((!checkForScalar || seqType.getShape().size() == 1) &&
           "rank must be 1 for a scalar CHARACTER");
    type = seqType.getEleTy();
  }
  if (auto charType = type.dyn_cast<fir::CharacterType>())
    return charType;
  llvm_unreachable("Invalid character value type");
}

/// Get fir.char<kind> type with the same kind as inside str.
fir::CharacterType
Fortran::lower::CharacterExprHelper::getCharacterType(mlir::Type type) {
  return recoverCharacterType<true>(type);
}

fir::CharacterType Fortran::lower::CharacterExprHelper::getCharacterType(
    const fir::CharBoxValue &box) {
  return getCharacterType(box.getBuffer().getType());
}

fir::CharacterType
Fortran::lower::CharacterExprHelper::getCharacterType(mlir::Value str) {
  return getCharacterType(str.getType());
}

/// Determine the static size of the character. Returns the computed size, not
/// an IR Value.
static std::optional<fir::SequenceType::Extent>
getCompileTimeLength(const fir::CharBoxValue &box) {
  auto type = box.getBuffer().getType();
  if (type.isa<fir::CharacterType>())
    return 1;
  if (auto refType = type.dyn_cast<fir::ReferenceType>())
    type = refType.getEleTy();
  if (auto seqType = type.dyn_cast<fir::SequenceType>()) {
    auto shape = seqType.getShape();
    assert(shape.size() == 1 && "only scalar character supported");
    if (shape[0] != fir::SequenceType::getUnknownExtent())
      return shape[0];
  }
  return {};
}

/// Detect the precondition that the value `str` does not reside in memory. Such
/// values will have a type `!fir.array<...x!fir.char<N>>` or `!fir.char<N>`.
static bool needToMaterialize(mlir::Value str) {
  return str.getType().isa<fir::SequenceType>() ||
         str.getType().isa<fir::CharacterType>();
}

/// This is called only if `str` does not reside in memory. Such a bare string
/// value will be converted into a memory-based temporary and an extended
/// boxchar value returned.
fir::CharBoxValue
Fortran::lower::CharacterExprHelper::materializeValue(mlir::Value str) {
  assert(needToMaterialize(str));
  mlir::Type eleTy;
  fir::SequenceType::Shape shape;
  if (auto seqTy = str.getType().dyn_cast<fir::SequenceType>()) {
    // (1) `str` is a sequence type
    assert(seqTy.getDimension() == 1 && "character is not a scalar");
    eleTy = seqTy.getEleTy();
    auto lenVal = seqTy.getShape()[0];
    assert(lenVal != fir::SequenceType::getUnknownExtent());
    shape.push_back(lenVal);
  } else {
    eleTy = str.getType();
    if (auto chTy = eleTy.dyn_cast<fir::CharacterType>()) {
      shape.push_back(chTy.getLen());
    } else {
      LLVM_DEBUG(llvm::dbgs() << "cannot materialize: " << str << '\n');
      llvm_unreachable("must be a !fir.char<N> type");
    }
  }
  assert(eleTy && shape.size() == 1);
  auto len =
      builder.createIntegerConstant(loc, builder.getIndexType(), shape[0]);
  auto charTy = fir::SequenceType::get(shape, eleTy);
  auto temp = builder.create<fir::AllocaOp>(loc, charTy);
  builder.create<fir::StoreOp>(loc, str, temp);
  LLVM_DEBUG(llvm::dbgs() << "materialized as local: " << str << " -> (" << temp
                          << ", " << len << ")\n");
  return {temp, len};
}

/// Use toExtendedValue to convert `character` to an extended value. This
/// assumes `character` is scalar and unwraps the extended value into a CharBox
/// value. This should not be used if `character` is an array.
fir::CharBoxValue
Fortran::lower::CharacterExprHelper::toDataLengthPair(mlir::Value character) {
  auto *charBox = toExtendedValue(character).getCharBox();
  assert(charBox && "Array unsupported in character lowering helper");
  return *charBox;
}

fir::ExtendedValue
Fortran::lower::CharacterExprHelper::toExtendedValue(mlir::Value character,
                                                     mlir::Value len) {
  auto lenType = getLengthType();
  auto type = character.getType();
  auto base = fir::isa_passbyref_type(type) ? character : mlir::Value{};
  auto resultLen = len;
  llvm::SmallVector<mlir::Value, 2> extents;

  if (auto eleType = fir::dyn_cast_ptrEleTy(type))
    type = eleType;

  if (auto arrayType = type.dyn_cast<fir::SequenceType>()) {
    auto shape = arrayType.getShape();
    auto cstLen = shape[0];
    if (!resultLen && cstLen != fir::SequenceType::getUnknownExtent())
      resultLen = builder.createIntegerConstant(loc, lenType, cstLen);
    // FIXME: only allow `?` in last dimension ?
    auto typeExtents =
        llvm::ArrayRef<fir::SequenceType::Extent>{shape}.drop_front();
    auto indexType = builder.getIndexType();
    for (auto extent : typeExtents) {
      if (extent == fir::SequenceType::getUnknownExtent())
        break;
      extents.emplace_back(
          builder.createIntegerConstant(loc, indexType, extent));
    }
    // Last extent might be missing in case of assumed-size. If more extents
    // could not be deduced from type, that's an error (a fir.box should
    // have been used in the interface).
    if (extents.size() + 1 < typeExtents.size())
      mlir::emitError(loc, "cannot retrieve array extents from type");
  } else if (type.isa<fir::CharacterType>()) {
    if (!resultLen)
      resultLen = builder.createIntegerConstant(loc, lenType, 1);
  } else if (auto boxCharType = type.dyn_cast<fir::BoxCharType>()) {
    auto refType = builder.getRefType(boxCharType.getEleTy());
    // If the embox is accessible, use its operand to avoid filling
    // the generated fir with embox/unbox.
    mlir::Value boxCharLen;
    if (auto definingOp = character.getDefiningOp()) {
      if (auto box = dyn_cast<fir::EmboxCharOp>(definingOp)) {
        base = box.memref();
        boxCharLen = box.len();
      }
    }
    if (!boxCharLen) {
      auto unboxed =
          builder.create<fir::UnboxCharOp>(loc, refType, lenType, character);
      base = unboxed.getResult(0);
      boxCharLen = unboxed.getResult(1);
    }
    if (!resultLen) {
      resultLen = boxCharLen;
    }
  } else if (type.isa<fir::BoxType>()) {
    mlir::emitError(loc, "descriptor or derived type not yet handled");
  } else {
    llvm_unreachable("Cannot translate mlir::Value to character ExtendedValue");
  }

  if (!base) {
    if (auto load =
            mlir::dyn_cast_or_null<fir::LoadOp>(character.getDefiningOp())) {
      base = load.getOperand();
    } else {
      return materializeValue(fir::getBase(character));
    }
  }
  if (!resultLen)
    mlir::emitError(loc, "no dynamic length found for character");
  if (!extents.empty())
    return fir::CharArrayBoxValue{base, resultLen, extents};
  return fir::CharBoxValue{base, resultLen};
}

/// Get canonical `!fir.ref<!fir.char<kind>>` type.
mlir::Type
Fortran::lower::CharacterExprHelper::getReferenceType(mlir::Value str) const {
  return builder.getRefType(getCharacterType(str));
}

mlir::Type Fortran::lower::CharacterExprHelper::getReferenceType(
    const fir::CharBoxValue &box) const {
  return getReferenceType(box.getBuffer());
}

/// Get canonical `!fir.array<len x !fir.char<kind>>` type.
mlir::Type
Fortran::lower::CharacterExprHelper::getSeqTy(mlir::Value str) const {
  if (auto ty = str.getType().dyn_cast<fir::SequenceType>())
    return ty;
  return builder.getRefType(builder.getVarLenSeqTy(getCharacterType(str)));
}

mlir::Type Fortran::lower::CharacterExprHelper::getSeqTy(
    const fir::CharBoxValue &box) const {
  return getSeqTy(box.getBuffer());
}

mlir::Value
Fortran::lower::CharacterExprHelper::createEmbox(const fir::CharBoxValue &box) {
  // BoxChar require a reference. Base CharBoxValue of CharArrayBoxValue
  // are ok here (do not require a scalar type)
  auto charTy = recoverCharacterType<false /* checkForScalar */>(
      box.getBuffer().getType());
  auto boxCharType =
      fir::BoxCharType::get(builder.getContext(), charTy.getFKind());
  auto refType = fir::ReferenceType::get(charTy);
  auto buff = builder.createConvert(loc, refType, box.getBuffer());
  // Convert in case the provided length is not of the integer type that must
  // be used in boxchar.
  auto lenType = getLengthType();
  auto len = builder.createConvert(loc, lenType, box.getLen());
  return builder.create<fir::EmboxCharOp>(loc, boxCharType, buff, len);
}

fir::CharBoxValue Fortran::lower::CharacterExprHelper::toScalarCharacter(
    const fir::CharArrayBoxValue &box) {
  if (box.getBuffer().getType().isa<fir::PointerType>())
    TODO("concatenating non contiguous character array into a scalar");

  // TODO: add a fast path multiplying new length at compile time if the info is
  // in the array type.
  auto lenType = getLengthType();
  auto len = builder.createConvert(loc, lenType, box.getLen());
  for (auto extent : box.getExtents())
    len = builder.create<mlir::MulIOp>(
        loc, len, builder.createConvert(loc, lenType, extent));

  // TODO: typeLen can be improved in compiled constant cases
  // TODO: allow bare fir.array<> (no ref) conversion here ?
  auto typeLen = fir::SequenceType::getUnknownExtent();
  auto baseType = recoverCharacterType<false /* do not check for scalar arg*/>(
      box.getBuffer().getType());
  auto charTy = fir::SequenceType::get({typeLen}, baseType);
  auto type = fir::ReferenceType::get(charTy);
  auto buffer = builder.createConvert(loc, type, box.getBuffer());
  return {buffer, len};
}

mlir::Value Fortran::lower::CharacterExprHelper::createEmbox(
    const fir::CharArrayBoxValue &box) {
  // Use same embox as for scalar. It's losing the actual data size information
  // (We do not multiply the length by the array size), but that is what Fortran
  // call interfaces using boxchar expect.
  return createEmbox(static_cast<const fir::CharBoxValue &>(box));
}

/// Load a character out of `buff` from offset `index`.
/// `buff` must be a reference to memory.
mlir::Value
Fortran::lower::CharacterExprHelper::createLoadCharAt(mlir::Value buff,
                                                      mlir::Value index) {
  LLVM_DEBUG(llvm::dbgs() << "load a char: " << buff << " type: "
                          << buff.getType() << " at: " << index << '\n');
  assert(fir::isa_ref_type(buff.getType()));
  auto coor = builder.createConvert(loc, getSeqTy(buff), buff);
  auto addr = builder.create<fir::CoordinateOp>(loc, getReferenceType(buff),
                                                coor, index);
  return builder.create<fir::LoadOp>(loc, addr);
}

/// Store the character `c` to `str` at offset `index`.
/// `str` must be a reference to memory.
void Fortran::lower::CharacterExprHelper::createStoreCharAt(mlir::Value str,
                                                            mlir::Value index,
                                                            mlir::Value c) {
  LLVM_DEBUG(llvm::dbgs() << "store the char: " << c << " into: " << str
                          << " type: " << str.getType() << " at: " << index
                          << '\n');
  assert(fir::isa_ref_type(str.getType()));
  auto buff = builder.createConvert(loc, getSeqTy(str), str);
  auto addr = builder.create<fir::CoordinateOp>(loc, getReferenceType(str),
                                                buff, index);
  builder.create<fir::StoreOp>(loc, c, addr);
}

// FIXME: this temp is useless... either fir.coordinate_of needs to
// work on "loaded" characters (!fir.array<len x fir.char<kind>>) or
// character should never be loaded.
// If this is a fir.array<>, allocate and store the value so that
// fir.cooridnate_of can be use on the value.
mlir::Value Fortran::lower::CharacterExprHelper::getCharBoxBuffer(
    const fir::CharBoxValue &box) {
  auto buff = box.getBuffer();
  if (buff.getType().isa<fir::SequenceType>()) {
    auto newBuff = builder.create<fir::AllocaOp>(loc, buff.getType());
    builder.create<fir::StoreOp>(loc, buff, newBuff);
    return newBuff;
  }
  return buff;
}

/// Create a loop to copy `count` characters from `src` to `dest`.
void Fortran::lower::CharacterExprHelper::createCopy(
    const fir::CharBoxValue &dest, const fir::CharBoxValue &src,
    mlir::Value count) {
  auto fromBuff = getCharBoxBuffer(src);
  auto toBuff = getCharBoxBuffer(dest);
  LLVM_DEBUG(llvm::dbgs() << "create char copy from: "; src.dump();
             llvm::dbgs() << " to: "; dest.dump();
             llvm::dbgs() << " count: " << count << '\n');
  Fortran::lower::DoLoopHelper{builder, loc}.createLoop(
      count, [&](Fortran::lower::FirOpBuilder &, mlir::Value index) {
        auto charVal = createLoadCharAt(fromBuff, index);
        createStoreCharAt(toBuff, index, charVal);
      });
}

void Fortran::lower::CharacterExprHelper::createPadding(
    const fir::CharBoxValue &str, mlir::Value lower, mlir::Value upper) {
  auto blank = createBlankConstant(getCharacterType(str));
  // Always create the loop, if upper < lower, no iteration will be
  // executed.
  auto toBuff = getCharBoxBuffer(str);
  Fortran::lower::DoLoopHelper{builder, loc}.createLoop(
      lower, upper, [&](Fortran::lower::FirOpBuilder &, mlir::Value index) {
        createStoreCharAt(toBuff, index, blank);
      });
}

fir::CharBoxValue
Fortran::lower::CharacterExprHelper::createCharacterTemp(mlir::Type type,
                                                         mlir::Value len) {
  assert(type.isa<fir::CharacterType>() && "expected fir character type");
  auto typeLen = fir::SequenceType::getUnknownExtent();
  // If len is a constant, reflect the length in the type.
  if (auto lenDefiningOp = len.getDefiningOp())
    if (auto constantOp = dyn_cast<mlir::ConstantOp>(lenDefiningOp))
      typeLen = constantOp.getValue().cast<::mlir::IntegerAttr>().getInt();
  auto charTy = fir::SequenceType::get({typeLen}, type);
  llvm::SmallVector<mlir::Value, 3> sizes{len};
  auto ref = builder.allocateLocal(loc, charTy, llvm::StringRef{}, sizes);
  return {ref, len};
}

// Simple length one character assignment without loops.
void Fortran::lower::CharacterExprHelper::createLengthOneAssign(
    const fir::CharBoxValue &lhs, const fir::CharBoxValue &rhs) {
  auto addr = lhs.getBuffer();
  mlir::Value val = builder.create<fir::LoadOp>(loc, rhs.getBuffer());
  auto valTy = val.getType();
  // Precondition is rhs is size 1, but it may be wrapped in a fir.array.
  if (auto seqTy = valTy.dyn_cast<fir::SequenceType>()) {
    auto zero =
        builder.createIntegerConstant(loc, builder.getIntegerType(32), 0);
    valTy = seqTy.getEleTy();
    val = builder.create<fir::ExtractValueOp>(loc, valTy, val, zero);
  }
  auto addrTy = fir::ReferenceType::get(valTy);
  addr = builder.createConvert(loc, addrTy, addr);
  assert(fir::dyn_cast_ptrEleTy(addr.getType()) == val.getType());
  builder.create<fir::StoreOp>(loc, val, addr);
}

void Fortran::lower::CharacterExprHelper::createAssign(
    const fir::CharBoxValue &lhs, const fir::CharBoxValue &rhs) {
  auto rhsCstLen = getCompileTimeLength(rhs);
  auto lhsCstLen = getCompileTimeLength(lhs);
  bool compileTimeSameLength =
      lhsCstLen && rhsCstLen && *lhsCstLen == *rhsCstLen;

  if (compileTimeSameLength && *lhsCstLen == 1) {
    createLengthOneAssign(lhs, rhs);
    return;
  }

  // Copy the minimum of the lhs and rhs lengths and pad the lhs remainder
  // if needed.
  auto copyCount = lhs.getLen();
  auto idxTy = builder.getIndexType();
  if (!compileTimeSameLength) {
    auto lhsLen = builder.createConvert(loc, idxTy, lhs.getLen());
    auto rhsLen = builder.createConvert(loc, idxTy, rhs.getLen());
    copyCount = Fortran::lower::genMin(builder, loc, {lhsLen, rhsLen});
  }

  // If rhs is in memory, always assumes rhs might overlap with lhs
  // in a way that require a temp for the copy. That can be optimize later.
  // Only create a temp of copyCount size because we do not need more from
  // rhs.
  // TODO: It should be rare that the assignment is between overlapping
  // substrings of the same variable. So this extra copy is pessimistic in the
  // common case.
  auto temp = createCharacterTemp(getCharacterType(rhs), copyCount);
  createCopy(temp, rhs, copyCount);

  // Actual copy
  createCopy(lhs, temp, copyCount);

  // Pad if needed.
  if (!compileTimeSameLength) {
    auto one = builder.createIntegerConstant(loc, lhs.getLen().getType(), 1);
    auto maxPadding = builder.create<mlir::SubIOp>(loc, lhs.getLen(), one);
    createPadding(lhs, copyCount, maxPadding);
  }
}

fir::CharBoxValue Fortran::lower::CharacterExprHelper::createConcatenate(
    const fir::CharBoxValue &lhs, const fir::CharBoxValue &rhs) {
  auto lhsLen = builder.createConvert(loc, getLengthType(), lhs.getLen());
  auto rhsLen = builder.createConvert(loc, getLengthType(), rhs.getLen());
  mlir::Value len = builder.create<mlir::AddIOp>(loc, lhsLen, rhsLen);
  auto temp = createCharacterTemp(getCharacterType(rhs), len);
  createCopy(temp, lhs, lhsLen);
  auto one = builder.createIntegerConstant(loc, len.getType(), 1);
  auto upperBound = builder.create<mlir::SubIOp>(loc, len, one);
  auto lhsLenIdx = builder.createConvert(loc, builder.getIndexType(), lhsLen);
  auto fromBuff = getCharBoxBuffer(rhs);
  auto toBuff = getCharBoxBuffer(temp);
  Fortran::lower::DoLoopHelper{builder, loc}.createLoop(
      lhsLenIdx, upperBound, one,
      [&](Fortran::lower::FirOpBuilder &bldr, mlir::Value index) {
        auto rhsIndex = bldr.create<mlir::SubIOp>(loc, index, lhsLenIdx);
        auto charVal = createLoadCharAt(fromBuff, rhsIndex);
        createStoreCharAt(toBuff, index, charVal);
      });
  return temp;
}

fir::CharBoxValue Fortran::lower::CharacterExprHelper::createSubstring(
    const fir::CharBoxValue &box, llvm::ArrayRef<mlir::Value> bounds) {
  // Constant need to be materialize in memory to use fir.coordinate_of.
  auto nbounds = bounds.size();
  if (nbounds < 1 || nbounds > 2) {
    mlir::emitError(loc, "Incorrect number of bounds in substring");
    return {mlir::Value{}, mlir::Value{}};
  }
  mlir::SmallVector<mlir::Value, 2> castBounds;
  // Convert bounds to length type to do safe arithmetic on it.
  for (auto bound : bounds)
    castBounds.push_back(builder.createConvert(loc, getLengthType(), bound));
  auto lowerBound = castBounds[0];
  // FIR CoordinateOp is zero based but Fortran substring are one based.
  auto one = builder.createIntegerConstant(loc, lowerBound.getType(), 1);
  auto offset = builder.create<mlir::SubIOp>(loc, lowerBound, one).getResult();
  auto idxType = builder.getIndexType();
  if (offset.getType() != idxType)
    offset = builder.createConvert(loc, idxType, offset);
  auto buff = builder.createConvert(loc, getSeqTy(box), box.getBuffer());
  auto substringRef = builder.create<fir::CoordinateOp>(
      loc, getReferenceType(box), buff, offset);

  // Compute the length.
  mlir::Value substringLen;
  if (nbounds < 2) {
    substringLen =
        builder.create<mlir::SubIOp>(loc, box.getLen(), castBounds[0]);
  } else {
    substringLen =
        builder.create<mlir::SubIOp>(loc, castBounds[1], castBounds[0]);
  }
  substringLen = builder.create<mlir::AddIOp>(loc, substringLen, one);

  // Set length to zero if bounds were reversed (Fortran 2018 9.4.1)
  auto zero = builder.createIntegerConstant(loc, substringLen.getType(), 0);
  auto cdt = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt,
                                          substringLen, zero);
  substringLen = builder.create<mlir::SelectOp>(loc, cdt, zero, substringLen);

  return {substringRef, substringLen};
}

mlir::Value Fortran::lower::CharacterExprHelper::createLenTrim(
    const fir::CharBoxValue &str) {
  // Note: Runtime for LEN_TRIM should also be available at some
  // point. For now use an inlined implementation.
  auto indexType = builder.getIndexType();
  auto len = builder.createConvert(loc, indexType, str.getLen());
  auto one = builder.createIntegerConstant(loc, indexType, 1);
  auto minusOne = builder.createIntegerConstant(loc, indexType, -1);
  auto zero = builder.createIntegerConstant(loc, indexType, 0);
  auto trueVal = builder.createIntegerConstant(loc, builder.getI1Type(), 1);
  auto blank = createBlankConstantCode(getCharacterType(str));
  mlir::Value lastChar = builder.create<mlir::SubIOp>(loc, len, one);

  auto iterWhile =
      builder.create<fir::IterWhileOp>(loc, lastChar, zero, minusOne, trueVal,
                                       /*returnFinalCount=*/false, lastChar);
  auto insPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(iterWhile.getBody());
  auto index = iterWhile.getInductionVar();
  // Look for first non-blank from the right of the character.
  auto fromBuff = getCharBoxBuffer(str);
  auto c = createLoadCharAt(fromBuff, index);
  c = builder.createConvert(loc, blank.getType(), c);
  auto isBlank =
      builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, blank, c);
  llvm::SmallVector<mlir::Value, 2> results = {isBlank, index};
  builder.create<fir::ResultOp>(loc, results);
  builder.restoreInsertionPoint(insPt);
  // Compute length after iteration (zero if all blanks)
  mlir::Value newLen =
      builder.create<mlir::AddIOp>(loc, iterWhile.getResult(1), one);
  auto result =
      builder.create<SelectOp>(loc, iterWhile.getResult(0), zero, newLen);
  return builder.createConvert(loc, getLengthType(), result);
}

fir::CharBoxValue
Fortran::lower::CharacterExprHelper::createCharacterTemp(mlir::Type type,
                                                         int len) {
  assert(type.isa<fir::CharacterType>() && "expected fir character type");
  assert(len >= 0 && "expected positive length");
  fir::SequenceType::Shape shape{len};
  auto seqType = fir::SequenceType::get(shape, type);
  auto addr = builder.create<fir::AllocaOp>(loc, seqType);
  auto mlirLen = builder.createIntegerConstant(loc, getLengthType(), len);
  return {addr, mlirLen};
}

// Returns integer with code for blank. The integer has the same
// size as the character. Blank has ascii space code for all kinds.
mlir::Value Fortran::lower::CharacterExprHelper::createBlankConstantCode(
    fir::CharacterType type) {
  auto bits = builder.getKindMap().getCharacterBitsize(type.getFKind());
  auto intType = builder.getIntegerType(bits);
  return builder.createIntegerConstant(loc, intType, ' ');
}

mlir::Value Fortran::lower::CharacterExprHelper::createBlankConstant(
    fir::CharacterType type) {
  return builder.createConvert(loc, type, createBlankConstantCode(type));
}

void Fortran::lower::CharacterExprHelper::createAssign(
    const fir::ExtendedValue &lhs, const fir::ExtendedValue &rhs) {
  if (auto *str = rhs.getBoxOf<fir::CharBoxValue>()) {
    if (auto *to = lhs.getBoxOf<fir::CharBoxValue>()) {
      createAssign(*to, *str);
    } else {
      auto lhsPair = toDataLengthPair(fir::getBase(lhs));
      createAssign(lhsPair, *str);
    }
  } else {
    auto lhsPair = toDataLengthPair(fir::getBase(lhs));
    auto rhsPair = toDataLengthPair(fir::getBase(rhs));
    createAssign(lhsPair, rhsPair);
  }
}

mlir::Value
Fortran::lower::CharacterExprHelper::createLenTrim(mlir::Value str) {
  return createLenTrim(toDataLengthPair(str));
}

mlir::Value
Fortran::lower::CharacterExprHelper::createEmboxChar(mlir::Value addr,
                                                     mlir::Value len) {
  return createEmbox(fir::CharBoxValue{addr, len});
}

std::pair<mlir::Value, mlir::Value>
Fortran::lower::CharacterExprHelper::createUnboxChar(mlir::Value boxChar) {
  auto box = toDataLengthPair(boxChar);
  return {box.getBuffer(), box.getLen()};
}

bool Fortran::lower::CharacterExprHelper::isCharacterLiteral(mlir::Type type) {
  if (auto seqType = type.dyn_cast<fir::SequenceType>())
    return (seqType.getShape().size() == 1) &&
           seqType.getEleTy().isa<fir::CharacterType>();
  return false;
}

bool Fortran::lower::CharacterExprHelper::isCharacterScalar(mlir::Type type) {
  if (type.isa<fir::BoxCharType>())
    return true;
  if (auto refType = type.dyn_cast<fir::ReferenceType>())
    type = refType.getEleTy();
  if (auto seqType = type.dyn_cast<fir::SequenceType>())
    if (seqType.getShape().size() == 1)
      type = seqType.getEleTy();
  return type.isa<fir::CharacterType>();
}

fir::KindTy
Fortran::lower::CharacterExprHelper::getCharacterKind(mlir::Type type) {
  return recoverCharacterType<true>(type).getFKind();
}

fir::KindTy Fortran::lower::CharacterExprHelper::getCharacterOrSequenceKind(
    mlir::Type type) {
  return recoverCharacterType<false>(type).getFKind();
}

bool Fortran::lower::CharacterExprHelper::isArray(mlir::Type type) {
  if (auto boxTy = type.dyn_cast<fir::BoxType>())
    type = boxTy.getEleTy();
  if (auto eleTy = fir::dyn_cast_ptrEleTy(type))
    type = eleTy;
  if (auto seqTy = type.dyn_cast<fir::SequenceType>()) {
    auto charTy = seqTy.getEleTy().dyn_cast<fir::CharacterType>();
    assert(charTy);
    return (!charTy.singleton()) || (seqTy.getDimension() > 1);
  }
  return false;
}
