! RUN: bbc -emit-fir %s -o - | FileCheck %s

! ABS
! CHECK-LABEL: abs_testi
subroutine abs_testi(a, b)
  integer :: a, b
  ! CHECK: shift_right_signed
  ! CHECK: xor
  ! CHECK: subi
  b = abs(a)
end subroutine

! CHECK-LABEL: abs_testr
subroutine abs_testr(a, b)
  real :: a, b
  ! CHECK: call @llvm.fabs.f32
  b = abs(a)
end subroutine

! CHECK-LABEL: abs_testz
subroutine abs_testz(a, b)
  complex :: a
  real :: b
  ! CHECK: fir.extract_value
  ! CHECK: fir.extract_value
  ! CHECK: call @{{.*}}hypot
  b = abs(a)
end subroutine

! AIMAG
! CHECK-LABEL: aimag_test
subroutine aimag_test(a, b)
  complex :: a
  real :: b
  ! CHECK: fir.extract_value
  b = aimag(a)
end subroutine

! AINT
! CHECK-LABEL: aint_test
subroutine aint_test(a, b)
  real :: a, b
  ! CHECK: call @llvm.trunc.f32
  b = aint(a)
end subroutine

! ANINT
! CHECK-LABEL: anint_test
subroutine anint_test(a, b)
  real :: a, b
  ! CHECK: call @llvm.round.f32
  b = anint(a)
end subroutine

! DBLE
! CHECK-LABEL: dble_test
subroutine dble_test(a)
  real :: a
  ! CHECK: fir.convert {{.*}} : (f32) -> f64
  print *, dble(a)
end subroutine

! DIM
! CHECK-LABEL: dim_testr
subroutine dim_testr(x, y, z)
  real :: x, y, z
  ! CHECK-DAG: %[[x:.*]] = fir.load %arg0
  ! CHECK-DAG: %[[y:.*]] = fir.load %arg1
  ! CHECK-DAG: %[[zero:.*]] = constant 0.0
  ! CHECK-DAG: %[[diff:.*]] = fir.subf %[[x]], %[[y]]
  ! CHECK: %[[cmp:.*]] = fir.cmpf "ogt", %[[diff]], %[[zero]]
  ! CHECK: %[[res:.*]] = select %[[cmp]], %[[diff]], %[[zero]]
  ! CHECK: fir.store %[[res]] to %arg2
  z = dim(x, y)
end subroutine
! CHECK-LABEL: dim_testi
subroutine dim_testi(i, j, k)
  integer :: i, j, k
  ! CHECK-DAG: %[[i:.*]] = fir.load %arg0
  ! CHECK-DAG: %[[j:.*]] = fir.load %arg1
  ! CHECK-DAG: %[[zero:.*]] = constant 0
  ! CHECK-DAG: %[[diff:.*]] = subi %[[i]], %[[j]]
  ! CHECK: %[[cmp:.*]] = cmpi "sgt", %[[diff]], %[[zero]]
  ! CHECK: %[[res:.*]] = select %[[cmp]], %[[diff]], %[[zero]]
  ! CHECK: fir.store %[[res]] to %arg2
  k = dim(i, j)
end subroutine

! DPROD
! CHECK-LABEL: dprod_test
subroutine dprod_test (x, y, z)
  real :: x,y
  double precision :: z
  z = dprod(x,y)
  ! CHECK-DAG: %[[x:.*]] = fir.load %arg0
  ! CHECK-DAG: %[[y:.*]] = fir.load %arg1
  ! CHECK-DAG: %[[a:.*]] = fir.convert %[[x]] : (f32) -> f64 
  ! CHECK-DAG: %[[b:.*]] = fir.convert %[[y]] : (f32) -> f64 
  ! CHECK: %[[res:.*]] = fir.mulf %[[a]], %[[b]]
  ! CHECK: fir.store %[[res]] to %arg2
end subroutine

! CEILING
! CHECK-LABEL: ceiling_test1
subroutine ceiling_test1(i, a)
  integer :: i
  real :: a
  i = ceiling(a)
  ! CHECK: %[[f:.*]] = call @llvm.ceil.f32
  ! CHECK: fir.convert %[[f]] : (f32) -> i32
end subroutine
! CHECK-LABEL: ceiling_test2
subroutine ceiling_test2(i, a)
  integer(8) :: i
  real :: a
  i = ceiling(a, 8)
  ! CHECK: %[[f:.*]] = call @llvm.ceil.f32
  ! CHECK: fir.convert %[[f]] : (f32) -> i64
end subroutine


! CONJG
! CHECK-LABEL: conjg_test
subroutine conjg_test(z1, z2)
  complex :: z1, z2
  ! CHECK: fir.extract_value
  ! CHECK: fir.negf
  ! CHECK: fir.insert_value
  z2 = conjg(z1)
end subroutine

! FLOOR
! CHECK-LABEL: floor_test1
subroutine floor_test1(i, a)
  integer :: i
  real :: a
  i = floor(a)
  ! CHECK: %[[f:.*]] = call @llvm.floor.f32
  ! CHECK: fir.convert %[[f]] : (f32) -> i32
end subroutine
! CHECK-LABEL: floor_test2
subroutine floor_test2(i, a)
  integer(8) :: i
  real :: a
  i = floor(a, 8)
  ! CHECK: %[[f:.*]] = call @llvm.floor.f32
  ! CHECK: fir.convert %[[f]] : (f32) -> i64
end subroutine

! IABS
! CHECK-LABEL: iabs_test
subroutine iabs_test(a, b)
  integer :: a, b
  ! CHECK: shift_right_signed
  ! CHECK: xor
  ! CHECK: subi
  b = iabs(a)
end subroutine

! IABS - Check if the return type (RT) has default kind.
! CHECK-LABEL: iabs_test
subroutine iabs_testRT(a, b)
  integer(KIND=4) :: a
  integer(KIND=16) :: b
  ! CHECK: shift_right_signed
  ! CHECK: xor
  ! CHECK: %[[RT:.*]] =  subi
  ! CHECK: fir.convert %[[RT]] : (i32)
  b = iabs(a)
end subroutine

! IAND
! CHECK-LABEL: iand_test
subroutine iand_test(a, b)
  integer :: a, b
  print *, iand(a, b)
  ! CHECK: %{{[0-9]+}} = and %{{[0-9]+}}, %{{[0-9]+}} : i{{(8|16|32|64|128)}}
end subroutine iand_test

! ICHAR
! CHECK-LABEL: ichar_test
subroutine ichar_test(c)
  character(1) :: c
  character :: str(10)
  ! CHECK: fir.load {{.*}} : !fir.ref<!fir.char<1>>
  ! CHECK: fir.convert {{.*}} : (!fir.char<1>) -> i32
  print *, ichar(c)

  ! CHECK: fir.extract_value {{.*}} : (!fir.array<1x!fir.char<1>>, i32) -> !fir.char<1>
  ! CHECK: fir.convert {{.*}} : (!fir.char<1>) -> i32
  print *, ichar(str(J))
end subroutine

! IEOR
! CHECK-LABEL: ieor_test
subroutine ieor_test(a, b)
  integer :: a, b
  print *, ieor(a, b)
  ! CHECK: %{{[0-9]+}} = xor %{{[0-9]+}}, %{{[0-9]+}} : i{{(8|16|32|64|128)}}
end subroutine ieor_test

! IOR
! CHECK-LABEL: ior_test
subroutine ior_test(a, b)
  integer :: a, b
  print *, ior(a, b)
  ! CHECK: %{{[0-9]+}} = or %{{[0-9]+}}, %{{[0-9]+}} : i{{(8|16|32|64|128)}}
end subroutine ior_test

! LEN
! CHECK-LABEL: len_test
subroutine len_test(i, c)
  integer :: i
  character(*) :: c
  ! CHECK: fir.boxchar_len
  i = len(c)
end subroutine

! LEN_TRIM
! CHECK-LABEL: len_trim_test
integer function len_trim_test(c)
  character(*) :: c
  ltrim = len_trim(c)
  ! CHECK-DAG: %[[c0:.*]] = constant 0 : index
  ! CHECK-DAG: %[[c1:.*]] = constant 1 : index
  ! CHECK-DAG: %[[cm1:.*]] = constant -1 : index
  ! CHECK-DAG: %[[lastChar:.*]] = subi {{.*}}, %[[c1]]
  ! CHECK: %[[iterateResult:.*]], %[[lastIndex:.*]] = fir.iterate_while (%[[index:.*]] = %[[lastChar]] to %[[c0]] step %[[cm1]]) and ({{.*}}) iter_args({{.*}}) {
    ! CHECK: %[[addr:.*]] = fir.coordinate_of {{.*}}, %[[index]]
    ! CHECK: %[[char:.*]] = fir.load %[[addr]]
    ! CHECK: %[[code:.*]] = fir.convert %[[char]]
    ! CHECK: %[[bool:.*]] = cmpi "eq"
    !CHECK fir.result %[[bool]], %[[index]]
  ! CHECK }
  ! CHECK-DAG: %[[len:.*]] = addi %[[lastIndex]], %[[c1]]
  ! CHECK: select %[[iterateResult]], %[[c0]], %[[len]]
end function

! NINT
! CHECK-LABEL: nint_test1
subroutine nint_test1(i, a)
  integer :: i
  real :: a
  i = nint(a)
  ! CHECK: call @llvm.lround.i32.f32
end subroutine
! CHECK-LABEL: nint_test2
subroutine nint_test2(i, a)
  integer(8) :: i
  real(8) :: a
  i = nint(a, 8)
  ! CHECK: call @llvm.lround.i64.f64
end subroutine



! SIGN
! CHECK-LABEL: sign_testi
subroutine sign_testi(a, b, c)
  integer a, b, c
  ! CHECK: shift_right_signed
  ! CHECK: xor
  ! CHECK: subi
  ! CHECK-DAG: subi
  ! CHECK-DAG: cmpi "slt"
  ! CHECK: select
  c = sign(a, b)
end subroutine

! CHECK-LABEL: sign_testr
subroutine sign_testr(a, b, c)
  real a, b, c
  ! CHECK-DAG: call {{.*}}fabs
  ! CHECK-DAG: fir.negf
  ! CHECK-DAG: fir.cmpf "olt"
  ! CHECK: select
  c = sign(a, b)
end subroutine

! SQRT
! CHECK-LABEL: sqrt_testr
subroutine sqrt_testr(a, b)
  real :: a, b
  ! CHECK: call {{.*}}sqrt
  b = sqrt(a)
end subroutine
