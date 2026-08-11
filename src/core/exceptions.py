"""
Domain-level exceptions.

These are raised by the service/repository layers and translated into
proper HTTP responses by exception handlers registered in main.py.
Keeping them framework-agnostic (not HTTPException subclasses) means the
service layer stays decoupled from FastAPI and could be reused elsewhere
(CLI scripts, background workers, etc).
"""


class AppException(Exception):
    """Base class for all application-level exceptions."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class UserAlreadyExistsException(AppException):
    def __init__(self, message: str = "A user with this email or username already exists."):
        super().__init__(message)


class InvalidCredentialsException(AppException):
    def __init__(self, message: str = "Incorrect email or password."):
        super().__init__(message)


class UserNotFoundException(AppException):
    def __init__(self, message: str = "User not found."):
        super().__init__(message)


class InactiveUserException(AppException):
    def __init__(self, message: str = "This user account is inactive."):
        super().__init__(message)


class InvalidTokenException(AppException):
    def __init__(self, message: str = "Could not validate credentials."):
        super().__init__(message)


class TokenExpiredException(AppException):
    def __init__(self, message: str = "Token has expired."):
        super().__init__(message)


# --------------------------------------------------------------------------
# Document module exceptions
# --------------------------------------------------------------------------
class UnsupportedFileTypeException(AppException):
    def __init__(self, message: str = "Unsupported file type. Allowed types: PDF, DOCX, TXT."):
        super().__init__(message)


class FileTooLargeException(AppException):
    def __init__(self, message: str = "File exceeds the maximum allowed upload size."):
        super().__init__(message)


class EmptyFileException(AppException):
    def __init__(self, message: str = "Uploaded file is empty."):
        super().__init__(message)


class DocumentNotFoundException(AppException):
    def __init__(self, message: str = "Document not found."):
        super().__init__(message)


class DocumentProcessingException(AppException):
    def __init__(self, message: str = "Document processing failed."):
        super().__init__(message)


class TextExtractionException(AppException):
    def __init__(self, message: str = "Could not extract text from the document."):
        super().__init__(message)


# --------------------------------------------------------------------------
# RAG / conversation module exceptions
# --------------------------------------------------------------------------
class ConversationNotFoundException(AppException):
    def __init__(self, message: str = "Conversation not found."):
        super().__init__(message)


class PromptInjectionException(AppException):
    def __init__(
        self,
        message: str = (
            "Your question contains prompt-injection patterns and was not processed. "
            "Please rephrase it without instructing the system to ignore rules, "
            "reveal prompts, or change its behavior."
        ),
    ):
        super().__init__(message)


class LLMException(AppException):
    def __init__(self, message: str = "The language model could not generate a response."):
        super().__init__(message)
