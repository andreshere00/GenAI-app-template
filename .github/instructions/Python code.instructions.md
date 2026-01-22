---
applyTo: '*.py'
---

## Documentation Style

I prefer Google docstyle and always in English documentation for methods and classes, including warns, raises, args, and returns, except for private methods or tiny helpers. 

## Code Complexity

I prefer a lower number of code lines and simpler solutions, applying software architecture patterns if needed.

## Naming Conventions

`CamelCase` is the go-to option for classes name, while `snake_case` should be applied for variables. For constants, I prefer to use `UPPER_CASE`. 

## Testing

- Test naming conventions should be `{method_name}_{state_under_test}_{expected_behavior}`.
- Tests should follow the Arrange-Act-Assert pattern. 
- Each test should focus on a single behavior or scenario.
- Test script should be divided into "# ---- Mocks, fixtures & helpers ---- #", "# ---- Happy path ---- #", "# ---- Error paths ---- #" and "# ---- Edge cases ---- #" sections.

## Code Style

- Code lines should not exceed 100 characters per line. 
- I don't want unnecessary comments, all the comments should be presented in the chat as rationales.
- Add typing in methods, constants and variables.
