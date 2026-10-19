import socket

import pytest


def _blocked(*args, **kwargs):
    raise RuntimeError("мережа в тестах заборонена: використовуйте підроблений клієнт")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Будь-яка спроба з'єднання в тесті падає одразу — жодних реальних запитів."""
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
