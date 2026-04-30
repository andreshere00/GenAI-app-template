from src.application.e2e import rag_app_service


def test_run_rag_use_case_function_defined_returns_coroutine_function() -> None:
    assert callable(rag_app_service.run_rag_use_case)
