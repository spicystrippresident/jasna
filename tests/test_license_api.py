import pytest

from jasna.license_api import ProtectionError, license_store


def test_public_source_license_boundary_keeps_free_features_available() -> None:
    assert license_store.load_license() is None
    assert license_store.is_licensed() is False


def test_public_source_license_boundary_rejects_activation_clearly() -> None:
    with pytest.raises(ProtectionError, match="public source checkout"):
        license_store.set_license("user@example.com", "test-key")
