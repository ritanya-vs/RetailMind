import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from retailmind.data.esci import (
    CHECKSUMS_NAME,
    ESCI_FILES,
    download_all,
    sha256_file,
    verify_all,
)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class _Server:
    def __init__(self, directory: Path):
        handler = functools.partial(_QuietHandler, directory=str(directory))
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def serve(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    servers: list[_Server] = []

    def _serve(directory: Path) -> _Server:
        server = _Server(directory)
        servers.append(server)
        return server

    yield _serve
    for server in servers:
        server.stop()


def _write_fake_dataset(directory: Path, drop: tuple[str, str] | None = None) -> None:
    """Write tiny parquet files with the real column names. drop = (file name, column to omit)."""
    directory.mkdir(parents=True, exist_ok=True)
    for spec in ESCI_FILES:
        columns = [
            c for c in spec.required_columns if drop != (spec.name, c)
        ]
        pd.DataFrame({c: ["x", "y"] for c in columns}).to_parquet(directory / spec.name)


def test_sha256_file_matches_known_value(tmp_path: Path):
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_file(path) == expected


def test_download_all_downloads_verifies_and_records_checksums(tmp_path: Path, serve):
    source = tmp_path / "source"
    _write_fake_dataset(source)
    server = serve(source)
    raw = tmp_path / "raw"

    summary = download_all(raw, base_url=server.url, progress=False)

    for spec in ESCI_FILES:
        info = summary[spec.name]
        assert info["downloaded"] is True
        assert info["rows"] == 2
        assert info["sha256"] == sha256_file(source / spec.name)
        assert (raw / spec.name).exists()
    saved = json.loads((raw / CHECKSUMS_NAME).read_text(encoding="utf-8"))
    assert saved == summary
    assert not list(raw.glob("*.part"))


def test_existing_files_are_not_downloaded_again(tmp_path: Path, serve):
    source = tmp_path / "source"
    _write_fake_dataset(source)
    server = serve(source)
    raw = tmp_path / "raw"
    download_all(raw, base_url=server.url, progress=False)
    server.stop()  # a second run must not need the network

    summary = download_all(raw, base_url=server.url, progress=False)

    assert all(info["downloaded"] is False for info in summary.values())


def test_missing_column_fails_and_removes_the_bad_download(tmp_path: Path, serve):
    source = tmp_path / "source"
    products = ESCI_FILES[1].name
    _write_fake_dataset(source, drop=(products, "product_title"))
    server = serve(source)
    raw = tmp_path / "raw"

    with pytest.raises(ValueError, match="product_title"):
        download_all(raw, base_url=server.url, progress=False)

    assert not (raw / products).exists()


def test_git_lfs_pointer_is_reported(tmp_path: Path, serve):
    source = tmp_path / "source"
    source.mkdir()
    pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n"
    for spec in ESCI_FILES:
        (source / spec.name).write_text(pointer, encoding="utf-8")
    server = serve(source)

    with pytest.raises(ValueError, match="LFS"):
        download_all(tmp_path / "raw", base_url=server.url, progress=False)


def test_http_error_is_an_oserror(tmp_path: Path, serve):
    empty = tmp_path / "empty"
    empty.mkdir()
    server = serve(empty)

    with pytest.raises(OSError):
        download_all(tmp_path / "raw", base_url=server.url, progress=False)


def test_verify_all_works_on_files_already_present(tmp_path: Path):
    raw = tmp_path / "raw"
    _write_fake_dataset(raw)

    summary = verify_all(raw)

    assert set(summary) == {spec.name for spec in ESCI_FILES}
    assert (raw / CHECKSUMS_NAME).exists()


def test_verify_all_requires_the_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        verify_all(tmp_path)
