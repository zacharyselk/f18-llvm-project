! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPformatassign
function formatAssign()
    real :: pi
    integer :: label
    logical :: flag

    ! CHECK: select
    if (flag) then
       assign 100 to label
    else
       assign 200 to label
    end if

    ! CHECK: fir.select
    ! CHECK-COUNT-3: br ^bb{{.*}})
    ! CHECK: call{{.*}}BeginExternalFormattedOutput
    ! CHECK-DAG: call{{.*}}OutputAscii
    ! CHECK-DAG: call{{.*}}OutputReal32
    ! CHECK: call{{.*}}EndIoStatement
    pi = 3.141592653589
    write(*, label) "PI=", pi
 

100 format (A, F10.3)
200 format (A,E8.1)
300 format (A, E2.4)

    end function
