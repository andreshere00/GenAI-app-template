"""Pre-generation hook.

Warns the user when embedding/vector-db selections will be ignored
because the ``rag`` service was not chosen.
"""

SERVICE: str = "{{ cookiecutter.services }}"


if __name__ == "__main__":
    if SERVICE != "rag":
        print(
            f"\n  Note: service is '{SERVICE}', not 'rag'."
            "\n  Any embedding provider and vector DB selections"
            " will be ignored.\n"
        )
