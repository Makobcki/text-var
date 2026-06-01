import re
import traceback

def extract_python_code(text: str) -> str | None:
    """
    Извлекает Python-код из markdown-блока.
    Ищет блок вида ```python ... ```.
    """
    # Ищем код внутри ```python и ``` (с учетом возможных пробелов)
    match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Фолбэк: если модель написала просто ``` ... ``` без указания языка
    match_fallback = re.search(r'```\n(.*?)\n```', text, re.DOTALL)
    if match_fallback:
        return match_fallback.group(1)
        
    return None

def check_syntax(code: str) -> tuple[bool, str | None]:
    """
    Проверяет синтаксис переданного Python-кода без его выполнения.
    Возвращает кортеж: (Успех (bool), Ошибка (str | None))
    """
    try:
        # Пытаемся скомпилировать код в AST
        compile(code, '<string>', 'exec')
        return True, None
    except Exception as e:
        # В случае ошибки (SyntaxError, IndentationError и т.д.) возвращаем traceback
        error_msg = traceback.format_exc()
        return False, error_msg
