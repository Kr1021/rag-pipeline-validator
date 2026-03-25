from dataclasses import dataclass
from typing import List, Set


@dataclass
class AuthCheckResult:
    authorised_access: int
    unauthorised_attempts: int
    violations: List[str]
    passed: bool


class AuthControlChecker:
    def __init__(self, allowed_scopes: Set[str]):
        self.allowed_scopes = allowed_scopes

    def check(self, access_log: List[dict]) -> AuthCheckResult:
        violations = [
            entry["resource"] for entry in access_log
            if entry.get("scope") not in self.allowed_scopes
        ]
        authorised = len(access_log) - len(violations)

        return AuthCheckResult(
            authorised_access=authorised,
            unauthorised_attempts=len(violations),
            violations=violations,
            passed=len(violations) == 0,
        )
