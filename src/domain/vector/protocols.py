from typing import Protocol, Optional

class VectorDBConfig(Protocol):
    """Protocol for vector database configuration."""
    # Connection parameters
    port: Optional[int] = 6333
    host: Optional[str] = "localhost"
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None

    # Advanced connection parameters
    grpc_port: Optional[int] = None
    https: Optional[bool] = False
    prefix: str = None
    
    # Optional configuration for specific vector databases
    timeout: Optional[int] = None
