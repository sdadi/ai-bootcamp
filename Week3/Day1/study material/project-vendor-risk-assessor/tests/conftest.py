import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assessor.config import SETTINGS  # noqa: E402
from assessor.policy_rag import PolicyLibrary  # noqa: E402
from assessor.schemas import VendorFact, VendorProfile  # noqa: E402

SETTINGS.verbose = False
AS_OF = date(2026, 11, 1)


@pytest.fixture(scope="session")
def library() -> PolicyLibrary:
    return PolicyLibrary()


def fact(field: str, value, level: str = "documented", as_of: date = date(2026, 6, 1)) -> VendorFact:
    return VendorFact(field=field, value=value, display=str(value), confidence=0.8, as_of=as_of, evidence_level=level,
                      sources=[f"https://vendor.example/{field}"])


def profile(**facts) -> VendorProfile:
    p = VendorProfile(vendor="TestVendor", official_domain="vendor.example", as_of=AS_OF)
    for k, v in facts.items():
        p.facts[k] = v
    return p
