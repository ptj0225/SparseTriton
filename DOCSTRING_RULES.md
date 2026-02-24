# Docstring 규칙 (Docstring Rules)

SparseTriton 프로젝트에서는 다음과 같은 docstring 형식을 따라야 합니다.

## 기본 형식

### 함수/메서드

```python
def function_name(arg1: type, arg2: type) -> return_type:
    """One-line summary of what the function does.

    Additional detailed description if needed (optional).

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Example:
        >>> import module
        >>> result = function_name(arg1_val, arg2_val)
        >>> result.shape
        torch.Size([...])
    """
```

### 클래스

```python
class ClassName:
    """One-line summary of the class.

    Additional detailed description.

    Attributes:
        attr1: Description of attr1
        attr2: Description of attr2

    Example:
        >>> obj = ClassName(param1=1, param2=2)
        >>> obj.method()
        result
    """
```

## 규칙

1. **요약**: 첫 줄은 간결한 한 줄 요약 (마침표로 끝)
2. **설명**: 필요한 경우 추가 설명 (빈 줄로 구분)
3. **섹션 순서**:
   - Attributes (클래스만 해당)
   - Args
   - Returns (선택)
   - Example
4. **포맷팅**:
   - 각 섹션 사이에 빈 줄
   - 섹션명 뒤에 콜론
   - 인자 설명: `arg_name: Description`
   - **타입 어노테이션은 docstring에 쓰지 않고 함수 시그니처에만 작성**
5. **Example**: 필요한 경우 예시 코드, `>>>` 사용
6. **언어**: 한국어 주석이 있는 파일은 한국어 docstring, 그 외는 영어

## 예시 비교

### ❌ 잘못된 예
```python
def foo(x: int) -> int:
    """Computes foo.

    Args:
        x (int): The input integer - 타입 설명 불필요

    Returns:
        int: Returns the result - 타입 설명 불필요
    """
```

### ✅ 올바른 예
```python
def foo(x: int) -> int:
    """Compute foo from input value.

    Args:
        x: Input integer value

    Returns:
        Computed foo value

    Example:
        >>> foo(5)
        25
    """
```
