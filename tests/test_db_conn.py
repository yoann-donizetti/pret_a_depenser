import core.db.conn as connmod


class FakeConn:
    def __init__(self, closed=False):
        self.closed = closed
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append(sql)
        return self


def test_get_conn_returns_none_if_no_db_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    connmod._CONN = None
    assert connmod.get_conn() is None


def test_get_conn_connects_when_no_cached_conn(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    connmod._CONN = None

    fake = FakeConn()
    calls = {"n": 0}

    def fake_connect(url, autocommit=True):
        calls["n"] += 1
        return fake

    monkeypatch.setattr(connmod.psycopg, "connect", fake_connect)

    c = connmod.get_conn()
    assert c is fake
    assert calls["n"] == 1


def test_get_conn_reuses_cached_conn_if_open(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    fake = FakeConn(closed=False)
    connmod._CONN = fake

    # si connect était appelé -> erreur
    monkeypatch.setattr(connmod.psycopg, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not connect")))

    c = connmod.get_conn()
    assert c is fake


def test_get_conn_reconnects_if_closed(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")

    fake_closed = FakeConn(closed=True)
    fake_new = FakeConn(closed=False)
    connmod._CONN = fake_closed

    calls = {"n": 0}
    def fake_connect(url, autocommit=True):
        calls["n"] += 1
        return fake_new

    monkeypatch.setattr(connmod.psycopg, "connect", fake_connect)

    c = connmod.get_conn()
    assert c is fake_new
    assert calls["n"] == 1


def test_init_db_returns_if_no_conn(monkeypatch):
    monkeypatch.setattr(connmod, "get_conn", lambda: None)
    connmod.init_db()  # doit juste passer


def test_init_db_executes_expected_sql(monkeypatch):
    fake = FakeConn()
    monkeypatch.setattr(connmod, "get_conn", lambda: fake)

    connmod.init_db()

    sql_all = "\n".join(fake.executed)

    assert "CREATE TABLE IF NOT EXISTS prod_requests" in sql_all
    assert "ALTER TABLE prod_requests" in sql_all
    assert "CREATE INDEX IF NOT EXISTS idx_prod_requests_ts" in sql_all