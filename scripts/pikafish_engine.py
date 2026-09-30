#!/usr/bin/env python3
"""Thin UCI client for the Pikafish xiangqi engine, used only to LABEL training
data (`gold` moves). Never used at serve time — `serve_xiangqi.py` never imports
this module, so playing against NanoJev never depends on Pikafish being installed.

UCI xiangqi squares are `<file><rank>` with file a-i (0-8) and rank 0-9, which is
exactly this repo's (row, col) scheme: file letter <-> col, rank digit <-> row.
"""
import subprocess


def square_to_uci(row, col):
    return f"{chr(ord('a') + col)}{row}"


def move_to_uci(move):
    (fr, fc), (tr, tc) = move
    return square_to_uci(fr, fc) + square_to_uci(tr, tc)


def uci_to_move(text):
    """Ranks are always a single digit (0-9), so a UCI xiangqi move is always 4 characters."""
    fc, fr, tc, tr = ord(text[0]) - ord('a'), int(text[1]), ord(text[2]) - ord('a'), int(text[3])
    return (fr, fc), (tr, tc)


class PikafishEngine:
    def __init__(self, executable_path, threads=1, hash_mb=64):
        self.proc = subprocess.Popen([executable_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        self._send("uci")
        self._read_until("uciok")
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, command):
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def _read_until(self, prefix):
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"Pikafish exited before printing '{prefix}'")
            line = line.strip()
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def best_move_for_ucis(self, uci_moves, movetime_ms=300):
        """uci_moves: list of already-played moves from the start position, in UCI form."""
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")
        cmd = "position startpos" + (" moves " + " ".join(uci_moves) if uci_moves else "")
        self._send(cmd)
        self._send(f"go movetime {movetime_ms}")
        lines = self._read_until("bestmove")
        best_line = lines[-1]
        move = best_line.split()[1]
        if move in ("(none)", "0000"):
            return None
        return move

    def close(self):
        self._send("quit")
        self.proc.wait(timeout=5)
