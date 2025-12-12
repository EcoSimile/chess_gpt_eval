#!/usr/bin/env python3
"""Proxy Play_CGPT through a light-weight UCI logger that PyChess accepts."""

from __future__ import annotations

import argparse
import datetime
import os
import random
import subprocess
import sys
import threading
from typing import Iterable, List, Optional

import chess

from engine_version import ENGINE_VERSION

DEFAULT_ENGINE_CMD = [
    "/usr/bin/python3",
    "/home/chriskar/chess_gpt_eval/Play_CGPT.py",
    "--opponent",
    "nanogpt",
]
DEFAULT_LOG_PATH = "/home/chriskar/chess_gpt_eval/logs/pychess_traffic.log"
DEFAULT_FAILURE_LOG_PATH = "/home/chriskar/chess_gpt_eval/logs/uci_failures.log"
DEFAULT_POSITION_LOG_PATH = (
    "/home/chriskar/chess_gpt_eval/logs/uci_logger/position_summary.log"
)
DEFAULT_FAILED_GAME_LOG_PATH = "/home/chriskar/chess_gpt_eval/logs/uci_logger/failed_games.log"
DEFAULT_ENGINE_VERSION = ENGINE_VERSION
DEFAULT_ENGINE_NAME = "ChessGPT Logger"
DEFAULT_ENGINE_AUTHOR = "ChessGPT"

log_lock = threading.Lock()


def log_line(log_path: str, prefix: str, line: str) -> str:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    if not line.endswith("\n"):
        line = f"{line}\n"
    formatted = f"{timestamp} {prefix} {line}"
    with log_lock, open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(formatted)
    return formatted


class UciLoggingProxy:
    """Small wrapper that spoofs the UCI handshake and rewrites stdout if needed."""

    def __init__(
        self,
        engine_cmd: Iterable[str],
        log_path: str,
        engine_name: str,
        engine_author: str,
        spoof_handshake: bool = True,
        failure_log_path: str = DEFAULT_FAILURE_LOG_PATH,
        position_log_path: str = DEFAULT_POSITION_LOG_PATH,
        failed_game_log_path: str = DEFAULT_FAILED_GAME_LOG_PATH,
        engine_version: str = DEFAULT_ENGINE_VERSION,
        illegal_san_threshold: int = 3,
    ) -> None:
        self.engine_cmd: List[str] = list(engine_cmd)
        self.log_path = log_path
        self.failure_log_path = failure_log_path
        self.position_log_path = position_log_path
        self.failed_game_log_path = failed_game_log_path
        self.engine_version = engine_version
        self.engine_name = engine_name
        self.engine_author = engine_author
        self.spoof_handshake = spoof_handshake
        self.stdout_lock = threading.Lock()
        self.board_lock = threading.Lock()
        self.stopped = threading.Event()
        self.engine_ready = threading.Event()
        self.current_game_events: List[str] = []
        self.current_game_failed = False
        self.capture_active = False
        self.failed_game_counter = self._load_failed_game_counter()
        self.latest_position_cmd: Optional[str] = None
        self.latest_board: Optional[chess.Board] = None
        self.board_at_go: Optional[chess.Board] = None
        self.last_go_command: Optional[str] = None
        self.current_search_id = 0
        self.illegal_san_counter = 0
        self.illegal_san_threshold = max(1, illegal_san_threshold)

        self.engine = subprocess.Popen(
            self.engine_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self.stdout_thread = threading.Thread(
            target=self._pump_engine_stdout, daemon=True
        )
        self.stdout_thread.start()

    def run(self) -> None:
        try:
            for raw_line in sys.stdin:
                if not raw_line:
                    continue
                self._handle_gui_command(raw_line)
        finally:
            self.shutdown()

    def _handle_gui_command(self, raw_line: str) -> None:
        logged_entry = log_line(self.log_path, ">>>", raw_line)
        stripped = raw_line.strip()
        lower = stripped.lower()

        if lower == "ucinewgame":
            self._start_new_game(initial_entry=logged_entry)
        else:
            self._record_game_event(logged_entry)

        if not stripped:
            return

        if lower == "uci":
            self._send_fake_handshake()
        elif lower == "xboard":
            self._write_gui("info string GUI requested xboard, but only UCI is supported")
        elif lower.startswith("position"):
            self._update_board_from_position(stripped)
        elif lower.startswith("go"):
            self._prepare_for_search(stripped)

        self._send_to_engine(raw_line)

    def _send_fake_handshake(self) -> None:
        if not self.spoof_handshake:
            return
        for msg in (
            f"id name {self.engine_name}",
            f"id author {self.engine_author}",
            "uciok",
        ):
            self._write_gui(msg)

    def _send_to_engine(self, data: str) -> None:
        if not self.engine.stdin:
            return
        try:
            self.engine.stdin.write(data)
            if not data.endswith("\n"):
                self.engine.stdin.write("\n")
            self.engine.stdin.flush()
        except BrokenPipeError:
            self._write_gui("info string Backend engine stdin pipe is closed")

    def _pump_engine_stdout(self) -> None:
        if not self.engine.stdout:
            return
        for line in self.engine.stdout:
            if not line:
                continue
            sanitized = self._sanitize_engine_output(line.rstrip("\n"))
            if sanitized is None:
                continue
            self._write_gui(sanitized)
        self.engine_ready.set()

    def _sanitize_engine_output(self, text: str) -> Optional[str]:
        stripped = text.strip()
        if not stripped:
            return ""

        lowered = stripped.lower()
        if lowered == "uciok":
            self.engine_ready.set()
            if self.spoof_handshake:
                return None
        if lowered.startswith("id ") and self.spoof_handshake:
            return None

        if "illegal san" in lowered:
            self.illegal_san_counter += 1
            if self.illegal_san_counter == 1:
                self._log_failure_context("Engine reported illegal SAN from backend")
            if self.illegal_san_counter >= self.illegal_san_threshold:
                if self._emit_fallback_move("illegal san threshold reached"):
                    return None

        if lowered.startswith("bestmove"):
            self.illegal_san_counter = 0
            self.last_go_command = None
            if lowered == "bestmove 0000":
                if self._emit_fallback_move("backend resigned (bestmove 0000)"):
                    return None

        passthrough_prefixes = (
            "bestmove",
            "info",
            "option",
            "readyok",
            "id ",
        )
        if lowered.startswith(passthrough_prefixes):
            return stripped

        return f"info string {stripped}"

    def _write_gui(self, text: str) -> None:
        if text is None:
            return
        line = text if text.endswith("\n") else f"{text}\n"
        with self.stdout_lock:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except BrokenPipeError:
                self.stopped.set()
                return
        logged_entry = log_line(self.log_path, "<<<", line)
        self._record_game_event(logged_entry)

    def shutdown(self) -> None:
        if self.stopped.is_set():
            return
        self.stopped.set()
        try:
            if self.engine.stdin and not self.engine.stdin.closed:
                self.engine.stdin.close()
        except Exception:
            pass
        try:
            self.engine.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.engine.kill()
        if self.engine.stdout and not self.engine.stdout.closed:
            self.engine.stdout.close()
        if self.stdout_thread.is_alive():
            self.stdout_thread.join(timeout=1)

    def _update_board_from_position(self, command: str) -> None:
        tokens = command.split()
        if len(tokens) < 2:
            return
        board: Optional[chess.Board]
        idx = 1
        kind = "unknown"
        if tokens[idx].lower() == "startpos":
            board = chess.Board()
            kind = "startpos"
            idx += 1
        elif tokens[idx].lower() == "fen":
            fen_fields = tokens[idx + 1 : idx + 7]
            if len(fen_fields) < 6:
                return
            fen = " ".join(fen_fields[:6])
            try:
                board = chess.Board(fen)
            except ValueError:
                return
            kind = "fen"
            idx += 7
        else:
            return

        base_board = board.copy(stack=False)
        applied_moves: List[str] = []
        san_moves: List[str] = []

        if idx < len(tokens) and tokens[idx].lower() == "moves":
            idx += 1
            for move in tokens[idx:]:
                if not move:
                    continue
                try:
                    uci_move = chess.Move.from_uci(move)
                    san = board.san(uci_move)
                    san_moves.append(san)
                    applied_moves.append(move)
                    board.push(uci_move)
                except Exception:
                    try:
                        board.push_san(move)
                        applied_moves.append(move)
                        san_moves.append(move)
                    except ValueError:
                        self._log_failure_context(
                            f"Failed to apply move '{move}' from position command"
                        )
                        break
        with self.board_lock:
            self.latest_board = board
            self.latest_position_cmd = command

        self._log_position_summary(kind, base_board, board, applied_moves, san_moves)

    def _prepare_for_search(self, command: str) -> None:
        with self.board_lock:
            self.board_at_go = self.latest_board.copy() if self.latest_board else None
        self.current_search_id += 1
        self.illegal_san_counter = 0
        self.last_go_command = command

    def _log_failure_context(self, reason: str) -> None:
        with self.board_lock:
            board_fen = self.board_at_go.fen() if self.board_at_go else "<unknown>"
            latest_fen = self.latest_board.fen() if self.latest_board else "<unknown>"
            position_cmd = self.latest_position_cmd or "<none>"
        details = (
            f"reason={reason}; "
            f"last_position={position_cmd}; "
            f"board_at_go={board_fen}; "
            f"latest_board={latest_fen}; "
            f"last_go={self.last_go_command or '<none>'}"
        )
        log_line(self.failure_log_path, "!!!", details)

    def _log_position_summary(
        self,
        kind: str,
        base_board: Optional[chess.Board],
        final_board: Optional[chess.Board],
        applied_moves: List[str],
        san_moves: List[str],
    ) -> None:
        if not self.position_log_path:
            return
        base_fen = base_board.fen() if base_board else "<unknown>"
        final_fen = final_board.fen() if final_board else "<unknown>"
        msg = (
            f"position_summary kind={kind} moves={len(applied_moves)} "
            f"base_fen={base_fen} final_fen={final_fen} "
            f"uci_moves={' '.join(applied_moves) if applied_moves else '<none>'} "
            f"san_moves={' '.join(san_moves) if san_moves else '<none>'}"
        )
        log_line(self.position_log_path, "###", msg)

    def _emit_fallback_move(self, reason: str) -> bool:
        with self.board_lock:
            board = self.board_at_go.copy() if self.board_at_go else None
        if board is None:
            self._log_failure_context(f"Fallback move unavailable: {reason}")
            self._mark_current_game_failed(f"fallback unavailable: {reason}")
            return False
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            self._log_failure_context(
                f"No legal moves available while attempting fallback ({reason})"
            )
            self._mark_current_game_failed(f"no legal moves for fallback: {reason}")
            return False
        move = random.choice(legal_moves)
        fallback_move = move.uci()
        self._log_failure_context(f"Injecting fallback move {fallback_move}: {reason}")
        self._write_gui(
            f"info string fallback move {fallback_move} injected because {reason}"
        )
        self._write_gui(f"bestmove {fallback_move}")
        self._mark_current_game_failed(f"fallback move {fallback_move}: {reason}")
        return True

    def _record_game_event(self, entry: Optional[str]) -> None:
        if not self.capture_active or not entry:
            return
        self.current_game_events.append(entry)

    def _start_new_game(self, initial_entry: Optional[str]) -> None:
        self.current_game_events = []
        self.current_game_failed = False
        self.capture_active = True
        if initial_entry:
            self.current_game_events.append(initial_entry)

    def _mark_current_game_failed(self, reason: str) -> None:
        if not self.capture_active or self.current_game_failed or not self.current_game_events:
            return
        self.current_game_failed = True
        self._persist_failed_game(reason)
        self.capture_active = False

    def _persist_failed_game(self, reason: str) -> None:
        os.makedirs(os.path.dirname(self.failed_game_log_path), exist_ok=True)
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        self.failed_game_counter += 1
        header = (
            f"### Failed game {self.failed_game_counter} (v{self.engine_version}) ###"
        )
        with open(self.failed_game_log_path, "a", encoding="utf-8") as failed_log:
            failed_log.write(f"{header}\n")
            failed_log.write(
                f"{timestamp} ### failure={reason} version={self.engine_version}\n"
            )
            for entry in self.current_game_events:
                failed_log.write(entry if entry.endswith("\n") else f"{entry}\n")
            failed_log.write("\n")

    def _load_failed_game_counter(self) -> int:
        try:
            with open(self.failed_game_log_path, "r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.startswith("### Failed game"))
        except FileNotFoundError:
            return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log and proxy a UCI engine")
    parser.add_argument(
        "--engine-cmd",
        nargs="+",
        default=DEFAULT_ENGINE_CMD,
        help="Command used to launch the backend engine",
    )
    parser.add_argument(
        "--log-path",
        default=DEFAULT_LOG_PATH,
        help="File used to store the transcript",
    )
    parser.add_argument(
        "--failure-log",
        default=DEFAULT_FAILURE_LOG_PATH,
        help="Supplemental log used for illegal move diagnostics",
    )
    parser.add_argument(
        "--position-log",
        default=DEFAULT_POSITION_LOG_PATH,
        help="Log used for position summaries (set empty to disable)",
    )
    parser.add_argument(
        "--failed-game-log",
        default=DEFAULT_FAILED_GAME_LOG_PATH,
        help="File that stores transcripts of failed games",
    )
    parser.add_argument(
        "--engine-version",
        default=DEFAULT_ENGINE_VERSION,
        help="Engine version stamp recorded in failed-game logs",
    )
    parser.add_argument(
        "--engine-name",
        default=DEFAULT_ENGINE_NAME,
        help="Name reported to the GUI during the UCI handshake",
    )
    parser.add_argument(
        "--engine-author",
        default=DEFAULT_ENGINE_AUTHOR,
        help="Author field reported to the GUI",
    )
    parser.add_argument(
        "--passthrough-handshake",
        action="store_true",
        help="Forward the backend's UCI handshake instead of spoofing it",
    )
    parser.add_argument(
        "--illegal-san-threshold",
        type=int,
        default=3,
        help="Emit a fallback move after this many illegal SAN reports",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proxy = UciLoggingProxy(
        engine_cmd=args.engine_cmd,
        log_path=args.log_path,
        failure_log_path=args.failure_log,
        position_log_path=args.position_log,
        failed_game_log_path=args.failed_game_log,
        engine_version=args.engine_version,
        engine_name=args.engine_name,
        engine_author=args.engine_author,
        spoof_handshake=not args.passthrough_handshake,
        illegal_san_threshold=args.illegal_san_threshold,
    )
    proxy.run()


if __name__ == "__main__":
    main()
