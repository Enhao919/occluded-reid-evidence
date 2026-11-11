"""Atomic full-head resumes; keep the prior valid file on any failed save."""

import os
from pathlib import Path
import shutil

from checkpoints import digest
from evaluate_checkpoint import require


def _safe_path(out,name):
    out = Path(out).resolve()
    path = out/name
    require(path.parent==out and not path.is_symlink(),'Unsafe checkpoint publication path')
    return path


def write_packet(path,packet,dump):
    path = Path(path)
    require(not path.is_symlink(),'Refuse checkpoint symlink')
    with path.open('wb') as stream:
        dump(packet,stream)
        stream.flush()
        os.fsync(stream.fileno())


def publish_resume(out,packet,dump):
    out = Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    require(shutil.disk_usage(out).free>=20*2**30,'Keep20GiB free before checkpoint publication')
    last = _safe_path(out,'last.pt')
    previous = _safe_path(out,'last.previous.pt')
    temporary = _safe_path(out,'last.pt.tmp')
    link = _safe_path(out,'last.previous.link.tmp')
    write_packet(temporary,packet,dump)
    if last.exists():
        link.unlink(missing_ok=True)
        os.link(last,link)
        os.replace(link,previous)
    os.replace(temporary,last)
    return last
