"""License boundary that keeps public source checkouts usable."""

from __future__ import annotations


try:
    from jasna.protection import ProtectionError, license_store
except ImportError:
    class ProtectionError(RuntimeError):
        """Raised when supporter activation is unavailable or invalid."""


    class _PublicSourceLicenseStore:
        @staticmethod
        def load_license() -> None:
            return None

        @staticmethod
        def is_licensed() -> bool:
            return False

        @staticmethod
        def set_license(email: str, key: str) -> None:
            del email, key
            raise ProtectionError(
                "Supporter activation is unavailable in the public source checkout"
            )


    license_store = _PublicSourceLicenseStore()


__all__ = ["ProtectionError", "license_store"]
