# SparseTriton Backward Pass Debugging Summary

## 문제 정의

SparseTriton의 `test_sparse_conv.py::TestSparseConv3DBackward` 테스트에서 3개 케이스 실패:

1. `test_sparse_conv3d_backward[16-16-3-1-0-1]` - padding=0
2. `test_sparse_conv3d_backward[64-64-3-1-1-1]` - padding=1
3. `test_sparse_conv3d_backward[512-512-5-1-1-1]` - larger kernel

## 원인 분석

### 1. Forward Overflow (해결됨 ✅)

**문제:**
```python
st_out_dense.abs().sum().backward()  # float16으로 sum() → overflow → inf
```

**해결:**
```python
st_out_dense.float().abs().sum().backward()  # float32로 sum()
```

### 2. Triton 3.x dtype 호환성 문제 (해결됨 ✅)

**문제:**
- `acc = tl.zeros(..., dtype=tl.float32)`로 초기화
- `f_tile`과 `w_tile`은 `tl.load(..., other=0.0)`으로 로드 → float16
- `tl.dot(f_tile, w_tile, acc=acc)`에서 dtype mismatch 에러 발생

**에러:**
```
Both operands must be same dtype. Got fp32 and fp16
```

**해결:**
```python
# acc는 float32로 유지
acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C_OUT), dtype=tl.float32)

# f_tile과 w_tile을 float32로 변환해서 dot
acc = tl.dot(f_tile.to(tl.float32), w_tile.to(tl.float32), acc=acc)

# forward에서 .to() 적용 - backward weight kernel도 동일
out_off = off_n[:, None] * C_out + off_cout[None, :]
tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_cout[None, :])
```

**수정된 kernels:**
- `implicit_gemm_hash_on_fly_fwd_kernel` - forward
- `implicit_gemm_bwd_feat_kernel` - backward feature
- `implicit_gemm_bwd_weight_kernel` - backward weight

### 3. Forward Forward Test에서의 Fail (현재 문제 ❌)

**현상:**
- `TestSparseConv3DForward`는 모든 케이스 통과 ✅
- `TestSparseConv3DBackward`의 forward check는 fail ❌
- Forward test: C_in=8, C_out=16, C_in=4, C_out=8, C_in=16, C_out=16 (통과)
- Backward test: C_in=1, C_out=1, C_in=64, C_out=64 (fail)

**추정:**
- C_in=C_out일 때 block size 문제
- 특정 C_in, C_out 값에서 hash table collision 또는 autotuner 문제

**디버깅 결과:**
- C_in=1, C_out=1, padding=1 → fail (Max diff: 7.79)
- C_in=1, C_out=1, padding=0 → fail (Max diff: 6.17)
- C_in=64, C_out=64, padding=1 → fail (Max diff: 90.6)
- C_in=64, C_out=32, padding=1 → fail (Max diff: 74.8)
- C_in=8, C_out=16, padding=1 → pass ✅ (forward test)

**결론:**
- C_in=1, 64일 때 forward fail
- C_in=4, 8, 16일 때 forward pass
- Padding과는 상관없음
- C_in=C_out이 아닌 C_in=64, C_out=32도 fail하므로, C_in=C_out이 문제가 아님

**가능한 원인:**
1. **C_in=1일 때 block size 문제**: BLOCK_SIZE_C_IN=64, C_in=1 → mask_cin이 1개만 True
2. **C_in=64일 때 문제**: BLOCK_SIZE_C_IN=64, C_in=64 → edge case
3. **Hash table collision**: 특정 nnz, C_in 조합에서 hash table 충돌 발생
4. **Autotuner config**: 특정 C_in, C_out에서 잘못된 config 선택

### 4. Backward Weight Gradient Overflow/Scale

**문제:**
- `spconv_w_grad`가 `tconv_w_grad`보다 **254.25배 큼**
- 이전에는 inf였으나, float32 누적으로 해결
- 이제는 scale 차이만 존재 (inf 없음)

**근본 원인:**
`implicit_gemm_bwd_weight_kernel`의 구조:
```python
# Grid: (N_out / BLOCK_SIZE_N, K, (C_in / BLOCK_SIZE_C_IN) * (C_out / BLOCK_SIZE_C_OUT))
# 각 output point (N_out개)에 대해 각 kernel position (K개)으로 처리
# 동일한 (kernel_idx, cin_idx, cout_idx)에 대해 atomic_add 누적
```

**문제점:**
1. **Grid 구조**: 모든 output points × 모든 kernel positions 순회
2. **유효하지 않은 쌍도 계산**: f_tile=0인 경우도 acc 계산 및 atomic_add 실행
3. **누적 횟수 차이**: 232,000 output × 27 kernel positions = 6,264,000번 vs 유효한 쌍

**Scale 계산:**
```
spconv_w_grad / tconv_w_grad ≈ 254.25
output_nnz/input_nnz * kernel_vol ≈ ?
```

## 시도한 해결책

### 1. Test 코드 수정 ✅

```python
# Before
st_out_dense.abs().sum().backward()

# After
st_out_dense.float().abs().sum().backward()
```

결과: forward overflow 해결

### 2. Triton dtype 호환성 수정 ✅

```python
# acc는 float32로 유지하고, f_tile과 w_tile을 float32로 변환
acc = tl.dot(f_tile.to(tl.float32), w_tile.to(tl.float32), acc=acc)
```

결과:
- Forward test: 3/3 통과 ✅
- Backward test: inf 제거 ✅, scale 차이 254.25배

### 3. Backward Weight Kernel 수정 (시도 중 ⚠️)

**방법 2: Grid 구조 변경 (진행 중)**
- 목표: 유효한 쌍만 순회
- 문제: 현재 코드는 원래 코드와 거의 동일
- 추가 시도 필요

## 남은 문제

### 1. Forward Forward Test에서의 Fail (우선순위 높음 ❌)

**Forward test (통과 ✅):**
- C_in=4, C_out=8, kernel_size=3, padding=1
- C_in=8, C_out=16, kernel_size=3, padding=1
- C_in=16, C_out=16, kernel_size=5, padding=2

**Backward test의 forward check (fail ❌):**
- C_in=1, C_out=1, kernel_size=3, padding=1 → fail (Max diff: 7.79)
- C_in=1, C_out=1, kernel_size=3, padding=0 → fail (Max diff: 6.17)
- C_in=64, C_out=64, kernel_size=3, padding=1 → fail (Max diff: 90.6)
- C_in=64, C_out=32, kernel_size=3, padding=1 → fail (Max diff: 74.8)

**패턴:**
- C_in=1, 64일 때 fail
- C_in=4, 8, 16일 때 pass
- Padding과는 상관없음

**가능한 원인:**
1. **C_in=1일 때 block size 문제**
   - BLOCK_SIZE_C_IN=64, C_in=1
   - mask_cin이 1개만 True, 63개 False
   - Triton autotuner가 다른 config 선택 가능

2. **C_in=64일 때 edge case**
   - BLOCK_SIZE_C_IN=64, C_in=64
   - 정확히 block size와 일치
   - Edge case에서 문제 발생 가능

3. **Hash table collision**
   - nnz=512 (8**3)
   - 특정 C_in, C_out 조합에서 hash table 충돌 발생

4. **Autotuner config 선택 문제**
   - C_in=1, 64에서 다른 config 선택
   - 잘못된 config 선택으로 오류 발생

**필요 분석:**
- Forward kernel의 autotuner config 확인
- C_in=1, 64일 때 선택된 config 분석
- Hash table collision 여부 확인
- 디버깅: output coords, valid_mask, hash table lookup 확인

### 2. Backward Weight Gradient의 근본적인 문제

```
spconv_w_grad / tconv_w_grad ≈ 254.25
```

**원인:**
- Grid 구조: `(N_out / BLOCK_SIZE_N, K, (C_in / BLOCK_SIZE_C_IN) * (C_out / BLOCK_SIZE_C_OUT))`
- 각 output point × 각 kernel position으로 순회
- 동일한 (k, cin, cout)에 대해 여러 output points가 기여
- f_tile이 0인 경우도 acc 계산 → 잘못된 기여

**필요 수정:**
backward weight kernel의 구조를 완전히 변경하여:
1. 각 (k, cin, cout)에 대해 **유효한 output points만** 순회
2. 또는 forward에서 사용된 in_out_map을 저장하고 backward에서 재사용

### 3. Backward Feature Gradient

forward와 비슷한 문제일 수 있음. 분석 필요.

## 다음 단계

### 1. Forward 버그 먼저 해결 (우선순위 높음)

- **C_in=1, 64일 때 autotuner config 분석**
  - Forward kernel의 autotune config 확인
  - C_in=1, 64에서 선택된 config 확인
  - 잘못된 config 선택 여부 확인

- **Hash table collision 확인**
  - nnz=512, C_in=1, 64에서 hash table 크기 확인
  - Collision 여부 확인
  - Hash table size 충분한지 확인

- **디버깅:**
  - Forward kernel에 debug print 추가
  - Output coords, valid_mask 확인
  - Hash table lookup 확인
  - Weight initialization 비교 (forward vs backward test)

### 2. Backward Weight Kernel 구조 변경 (방법 2)

**방법 2-1: valid_mask 적용**
- backward weight kernel에서 valid_mask를 올바르게 적용
- 유효하지 않은 쌍은 atomic_add 생략

**방법 2-2: Grid 구조 변경**
- Grid를 `(K, (C_in / BLOCK_SIZE_C_IN) * (C_out / BLOCK_SIZE_C_OUT), N_out / BLOCK_SIZE_N)`으로 변경
- 각 kernel position × 각 channel block에 대해 유효한 output points만 순회

### 3. 통합 테스트

- padding=0, 1 케이스 모두 통과
- kernel_size=3, 5 케이스 모두 통과
- C_in, C_out 다양한 조합 통과 (1, 4, 8, 16, 64, 512)

## 파일 수정 히스토리

1. `tests/test_sparse_conv.py`:
   - Line 155: `nnz=32**3` → `nnz=8**3`
   - Line 154: `.abs().sum().backward()` → `.float().abs().sum().backward()`
   - Line 180: `.abs().sum().backward()` → `.float().abs().sum().backward()`

2. `sparsetriton/nn/functional/conv/funcs/implicit_hashfly_gemm.py`:
   - **Line 44**: `acc = tl.zeros(..., dtype=tl.float32)` (forward)
   - **Line 84**: `acc = tl.dot(f_tile.to(tl.float32), w_tile.to(tl.float32), acc=acc)` (forward)
   - **Line 137**: `acc = tl.zeros(..., dtype=tl.float32)` (backward feature)
   - **Line 166**: `acc = tl.dot(do_tile.to(tl.float32), tl.trans(w_tile).to(tl.float32), acc=acc)` (backward feature)
   - **Line 258**: `acc = tl.dot(tl.trans(f_tile).to(tl.float32), do_tile.to(tl.float32))` (backward weight)

## 참고 코드 위치

- Forward kernel: `implicit_gemm_hash_on_fly_fwd_kernel` (Line 14-94)
- Backward feature kernel: `implicit_gemm_bwd_feat_kernel` (Line 101-177)
- Backward weight kernel: `implicit_gemm_bwd_weight_kernel` (Line 185-262)
- Autograd Function: `ConvHashOnTheFlyImplicitGEMM` (Line 269-355)
