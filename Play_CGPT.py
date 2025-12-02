#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import chess

from nanogpt.nanogpt_module import NanoGptPlayer
from play_vs_chessgptCommandLine import GPTPlayer, LegalMoveResponse, get_legal_move

PROMPT_PATH = os.path.join(ROOT_DIR, "gpt_inputs", "prompt.txt")
DEFAULT_NANOGPT_MODEL = "lichess_200k_bins_16layers_ckpt_with_optimizer.pt"
FAILURE_LOG = os.path.join(ROOT_DIR, "logs", "uci_failures.log")
DEBUG_LOG = os.path.join(ROOT_DIR, "logs", "uci_debug.log")
PROMPT_LOG = os.path.join(ROOT_DIR, "logs", "nanogpt_prompts_uci.log")

# Ensure log directory exists before configuring logger (avoid import-time failures)
os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)
logging.basicConfig(
    filename=DEBUG_LOG,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def log_failure(
    reason: str,
    board: chess.Board,
    game_state: str,
    attempts: int,
    attempt_history: Optional[List[str]],
):
    os.makedirs(os.path.dirname(FAILURE_LOG), exist_ok=True)
    with open(FAILURE_LOG, "a", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write(f"Reason: {reason}\n")
        side = "white" if board.turn == chess.WHITE else "black"
        f.write(f"Attempts: {attempts}\n")
        if attempt_history:
            f.write("Attempt history:\n")
            for idx, mv in enumerate(attempt_history, 1):
                shown = mv if (mv is not None and str(mv).strip() != "") else "[EMPTY]"
                f.write(f"  {idx}: {shown}\n")
        f.write(f"Side to move: {side}\n")
        f.write(f"FEN: {board.fen()}\n")
        f.write(f"Transcript length (chars): {len(game_state)}\n")
        f.write(f"Transcript tokens: {len(game_state.split())}\n")
        f.write("Transcript:\n")
        f.write(game_state.strip() + "\n")
        f.write("=" * 40 + "\n\n")


def log_prompt_uci(stage: str, board: chess.Board, game_state: str, extra: Optional[str] = None, attempts: Optional[List[str]] = None):
    os.makedirs(os.path.dirname(PROMPT_LOG), exist_ok=True)
    side = "white" if board.turn == chess.WHITE else "black"
    with open(PROMPT_LOG, "a", encoding="utf-8") as f:
        f.write("#" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Stage: {stage}\n")
        f.write(f"Side to move: {side}\n")
        f.write(f"FEN: {board.fen()}\n")
        f.write(f"Transcript length: {len(game_state)}\n")
        if extra:
            f.write(f"Extra: {extra}\n")
        if attempts:
            f.write("Attempts:\n")
            for i, a in enumerate(attempts, 1):
                shown = a if (a is not None and str(a).strip() != "") else "[EMPTY]"
                f.write(f"  {i}: {shown}\n")
        f.write("Transcript:\n")
        f.write(game_state + "\n")


class ChessGptUciEngine:
    """Minimal UCI bridge that wraps an existing ChessGPT player."""

    def __init__(self, player, engine_name: str, temperature: float = 0.0):
        self.player = player
        self.engine_name = engine_name
        self.temperature = temperature
        self.base_prompt = self._load_prompt()
        self.board = chess.Board()
        self.history_path = os.path.join(ROOT_DIR, "logs", "active_game_uci.json")
        # Prompt prefix can include optional FEN headers when GUIs start from arbitrary positions
        self.prompt_prefix = self.base_prompt
        self.active_fen: Optional[str] = None
        self.history_base_fen: str = chess.STARTING_FEN
        self.uci_history: List[str] = []
        self.pgn_tokens: List[str] = []  # e.g., ["1. e4 d6", "2. d4 c5"]
        self.game_state = self.prompt_prefix
        self.fen_only_context = False  # True when we only have a FEN and partial moves, not full game history
        self._opening_cache: Optional[Dict[str, Tuple[List[str], List[str]]]] = None

    def _load_prompt(self) -> str:
        with open(PROMPT_PATH, "r") as f:
            return f.read()

    def reset(self):
        self.board.reset()
        self.active_fen = None
        self.prompt_prefix = self.base_prompt
        self.history_base_fen = chess.STARTING_FEN
        self.uci_history = []
        self.pgn_tokens = []
        self._refresh_game_state()
        # Don't delete history file here; some GUIs send 'ucinewgame' before replaying moves

    def _legal_moves_san(self) -> str:
        legal_san: List[str] = []
        for m in self.board.legal_moves:
            try:
                legal_san.append(self.board.san(m))
            except Exception:
                continue
        return " ".join(legal_san)

    def _refresh_game_state(self):
        moves_text = " " + " ".join(self.pgn_tokens) if self.pgn_tokens else ""
        base = self.prompt_prefix
        if self.fen_only_context:
            # Append a SAN move list hint using only tokens in the NanoGPT vocabulary
            hint = self._legal_moves_san()
            if hint:
                base += f"; {hint} ;\n\n"
        self.game_state = base + moves_text

    def _configure_prompt_for_fen(self, fen: Optional[str]):
        if fen:
            header = self.base_prompt.strip()
            self.prompt_prefix = f"{header}\n[SetUp \"1\"]\n[FEN \"{fen}\"]\n\n"
        else:
            self.prompt_prefix = self.base_prompt
        self.active_fen = fen
        self.fen_only_context = False
        if not self.pgn_tokens:
            self.game_state = self.prompt_prefix

    def _initialize_from_new_fen(self, fen: str):
        self.history_base_fen = fen
        self._configure_prompt_for_fen(fen)
        self.uci_history = []
        self.pgn_tokens = []
        self.board.set_fen(fen)
        self.fen_only_context = True
        self._refresh_game_state()

    def _prompt_fullmove_number(self, board: Optional[chess.Board] = None) -> int:
        ref = board or self.board
        return ref.fullmove_number

    def handle_position(self, tokens: List[str]):
        if not tokens:
            self.reset()
            return

        idx = 0
        current_fen: Optional[str] = None
        token = tokens[idx]
        start_from_fen = False
        if token == "startpos":
            self.reset()
            idx += 1
        elif token == "fen":
            fen_tokens = tokens[idx + 1 : idx + 7]
            fen = " ".join(fen_tokens)
            current_fen = fen
            self.board.set_fen(fen)
            idx += 7
            start_from_fen = True
        else:
            self.reset()

        if idx < len(tokens) and tokens[idx] == "moves":
            moves = tokens[idx + 1 :]
            logging.info("Rebuilding from moves: %s", moves)
            self._apply_external_moves(
                moves, start_from_fen=start_from_fen, fen_source=current_fen
            )
            self._persist_history()
        elif start_from_fen and not (idx < len(tokens) and tokens[idx] == "moves"):
            # FEN without moves – try to reconstruct just from saved history
            if self._resume_from_saved_history_if_possible():
                self._persist_history()
            elif current_fen and self._bootstrap_from_opening(current_fen):
                self._persist_history()
            elif current_fen:
                self._initialize_from_new_fen(current_fen)
                self._persist_history()

    def _apply_external_moves(
        self,
        moves: List[str],
        start_from_fen: bool = False,
        fen_source: Optional[str] = None,
    ):
        if start_from_fen:
            if not self._resume_from_saved_history_if_possible():
                source = fen_source or self.board.fen()
                logging.info(
                    "Initializing new FEN context for GUI supplied position: %s", source
                )
                if not (source and self._bootstrap_from_opening(source)):
                    self._initialize_from_new_fen(source)
            else:
                # Ensure prompt reflects any persisted fen context
                if self.active_fen:
                    self._configure_prompt_for_fen(self.active_fen)
                self.fen_only_context = bool(self.active_fen)
        else:
            # startpos case: rebuild from scratch
            self.board.reset()
            self.active_fen = None
            self.history_base_fen = chess.STARTING_FEN
            self.prompt_prefix = self.base_prompt
            self.uci_history = []
            self.pgn_tokens = []
            self.fen_only_context = False
            self._refresh_game_state()

        # At this point, either we've resumed history into self.board/self.pgn_tokens
        # or we're at startpos with empty tokens. Append incoming moves to both.
        current_pair: Optional[str] = None
        if self.pgn_tokens:
            # If last token has both moves? We detect based on side to move
            if self.board.turn == chess.WHITE:
                current_pair = None
            else:
                # Start appending to existing last token
                current_pair = self.pgn_tokens.pop()

        for move_str in moves:
            move = chess.Move.from_uci(move_str)
            san = self.board.san(move)
            if current_pair is None:
                current_pair = f"{self._prompt_fullmove_number()}. {san}"
            else:
                current_pair += f" {san}"
                self.pgn_tokens.append(current_pair)
                current_pair = None
            self.board.push(move)
            self.uci_history.append(move_str)

        if current_pair:
            self.pgn_tokens.append(current_pair)

        # Rebuild game_state from tokens
        self._refresh_game_state()
        self._persist_history()

    def _persist_history(self):
        try:
            os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
            import json
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "uci_history": self.uci_history,
                        "pgn_tokens": self.pgn_tokens,
                        "history_base_fen": self.history_base_fen,
                        "active_fen": self.active_fen,
                    },
                    f,
                )
        except Exception as e:
            logging.warning("Failed to persist history: %s", e)

    def _load_history(self) -> bool:
        try:
            import json
            with open(self.history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.uci_history = data.get("uci_history", [])
            self.pgn_tokens = data.get("pgn_tokens", [])
            self.history_base_fen = data.get("history_base_fen", chess.STARTING_FEN)
            self.active_fen = data.get("active_fen")
            if self.active_fen:
                self._configure_prompt_for_fen(self.active_fen)
            else:
                self.prompt_prefix = self.base_prompt
            self._refresh_game_state()
            return bool(self.uci_history)
        except Exception:
            return False

    def _resume_from_saved_history_if_possible(self) -> bool:
        # Attempt to rebuild from saved in-memory or persisted history so that
        # board.fen() before applying incoming moves matches the GUI-provided FEN
        if not self.uci_history and not self._load_history():
            return False
        saved_moves = list(self.uci_history)
        # Recreate board from stored base using saved moves
        temp = chess.Board(self.history_base_fen)
        try:
            for m in saved_moves:
                temp.push_uci(m)
        except Exception:
            return False
        # If FENs match current board FEN, accept and rebuild state from saved tokens
        if self._boards_equivalent(temp, self.board):
            self.board = temp
            # If no pgn_tokens loaded, rebuild from moves
            if not self.pgn_tokens:
                self._rebuild_tokens_from_moves(saved_moves, self.history_base_fen)
            self._refresh_game_state()
            return True
        return False

    @staticmethod
    def _boards_equivalent(a: chess.Board, b: chess.Board) -> bool:
        return (
            a.board_fen() == b.board_fen()
            and a.turn == b.turn
            and a.castling_xfen() == b.castling_xfen()
            and a.ep_square == b.ep_square
        )

    def _rebuild_tokens_from_moves(self, moves: List[str], start_fen: Optional[str] = None):
        self.pgn_tokens = []
        temp = chess.Board(start_fen) if start_fen else chess.Board()
        current_pair: Optional[str] = None
        for m in moves:
            move = chess.Move.from_uci(m)
            san = temp.san(move)
            if current_pair is None:
                move_no = temp.fullmove_number
                current_pair = f"{move_no}. {san}"
            else:
                current_pair += f" {san}"
                self.pgn_tokens.append(current_pair)
                current_pair = None
            temp.push(move)
        if current_pair:
            self.pgn_tokens.append(current_pair)

    def _append_move_number_if_needed(self):
        if self.board.turn == chess.WHITE:
            move_no = self._prompt_fullmove_number()
            if move_no != 1 or self.pgn_tokens:
                self.game_state += " "
            self.game_state += f"{move_no}."
        else:
            self.game_state += " "

    def _append_san_to_game_state(self, san: str):
        self.game_state += san

    def _get_best_move(self) -> Tuple[Optional[str], Optional[str]]:
        self._append_move_number_if_needed()
        # Log the exact transcript we are about to send to NanoGPT
        log_prompt_uci("PROMPT", self.board, self.game_state)
        result: LegalMoveResponse = get_legal_move(
            self.player,
            self.board,
            self.game_state,
            player_one=self.board.turn == chess.WHITE,
        )
        if result.move_uci is None:
            if result.is_resignation:
                message = "model resigned (returned game result token)"
            elif result.is_illegal_move:
                message = (
                    f"model produced illegal moves for {result.attempts} attempts"
                )
            else:
                message = "model failed to return a move"

            log_failure(
                message,
                self.board.copy(),
                self.game_state,
                result.attempts,
                result.attempt_history,
            )
            print(f"info string {message}")
            sys.stdout.flush()
            log_prompt_uci("FAILURE", self.board, self.game_state, extra=message, attempts=result.attempt_history)
            return None, None
        self._append_san_to_game_state(result.move_san)
        # Record JSON history before mutating the board so move numbers align
        try:
            self._record_engine_move(result.move_uci, result.move_san)
        except Exception as e:
            logging.warning("History record error: %s", e)
        self.board.push(result.move_uci)
        logging.info(
            "Engine move selected: %s (SAN %s) with transcript length %d",
            result.move_uci.uci(),
            result.move_san,
            len(self.game_state),
        )
        log_prompt_uci("RESULT", self.board, self.game_state, extra=f"bestmove {result.move_uci.uci()} SAN {result.move_san}")
        return result.move_uci.uci(), result.move_san

    def _record_engine_move(self, move_uci: chess.Move, move_san: str):
        """Append engine's move to persistent history and rebuild tokens."""
        # Track UCI history
        try:
            uci_text = move_uci.uci()
        except Exception:
            uci_text = None
        if uci_text:
            self.uci_history.append(uci_text)
        # Append to PGN tokens in training style
        san_clean = move_san.strip()
        if self.board.turn == chess.WHITE:
            # Engine is about to play White's move
            pair = f"{self._prompt_fullmove_number()}. {san_clean}"
            self.pgn_tokens.append(pair)
        else:
            # Completing the current pair with Black's SAN
            if self.pgn_tokens:
                self.pgn_tokens[-1] += f" {san_clean}"
            else:
                self.pgn_tokens.append(
                    f"{self._prompt_fullmove_number()}. {san_clean}"
                )
        # Rebuild game_state and persist
        self._refresh_game_state()
        self._persist_history()

    def _load_opening_cache(self) -> Dict[str, Tuple[List[str], List[str]]]:
        if self._opening_cache is not None:
            return self._opening_cache
        mapping: Dict[str, Tuple[List[str], List[str]]] = {}
        path = os.path.join(ROOT_DIR, "logs", "openings.csv")
        if not os.path.exists(path):
            self._opening_cache = mapping
            return mapping

        def snapshot(board: chess.Board, tokens: List[str], pending: Optional[str], history: List[str]):
            temp_tokens = list(tokens)
            if pending:
                temp_tokens.append(pending)
            mapping.setdefault(board.fen(), (temp_tokens, list(history)))

        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.lower().startswith("opening"):
                    continue
                san_tokens: List[str] = []
                for token in line.replace("...", " ").split():
                    token = token.strip()
                    if not token:
                        continue
                    if token.endswith("."):
                        continue
                    if "." in token:
                        token = token.split(".")[-1]
                    if not token:
                        continue
                    san_tokens.append(token)
                board = chess.Board()
                pgn_tokens: List[str] = []
                history: List[str] = []
                current_pair: Optional[str] = None
                snapshot(board, pgn_tokens, current_pair, history)
                for san in san_tokens:
                    try:
                        move = board.parse_san(san)
                    except Exception:
                        current_pair = None
                        break
                    if current_pair is None:
                        current_pair = f"{board.fullmove_number}. {san}"
                    else:
                        current_pair += f" {san}"
                        pgn_tokens.append(current_pair)
                        current_pair = None
                    board.push(move)
                    history.append(move.uci())
                    snapshot(board, pgn_tokens, current_pair, history)
                if current_pair:
                    temp_tokens = list(pgn_tokens)
                    temp_tokens.append(current_pair)
                    mapping.setdefault(board.fen(), (temp_tokens, list(history)))
        self._opening_cache = mapping
        logging.info("Loaded %d opening positions for FEN bootstrapping", len(mapping))
        return mapping

    def _lookup_opening_history(self, fen: str) -> Optional[Tuple[List[str], List[str]]]:
        cache = self._load_opening_cache()
        entry = cache.get(fen)
        if not entry:
            return None
        tokens, history = entry
        return list(tokens), list(history)

    def _bootstrap_from_opening(self, fen: str) -> bool:
        lookup = self._lookup_opening_history(fen)
        if not lookup:
            return False
        tokens, history = lookup
        board = chess.Board()
        for uci in history:
            try:
                board.push_uci(uci)
            except Exception:
                return False
        target = chess.Board(fen)
        if not self._boards_equivalent(board, target):
            return False
        self.board = board
        self.history_base_fen = chess.STARTING_FEN
        self.prompt_prefix = self.base_prompt
        self.active_fen = None
        self.uci_history = history
        self.pgn_tokens = tokens
        self._refresh_game_state()
        logging.info("Bootstrapped history from opening book for FEN %s", fen)
        return True

    def loop(self):
        for raw_line in sys.stdin:
            line = raw_line.strip()
            logging.info("RECV: %s", line)
            if not line:
                continue
            tokens = line.split()
            command = tokens[0]

            if command == "uci":
                print(f"id name {self.engine_name}")
                print("id author ChessGPT")
                print("uciok")
                sys.stdout.flush()
            elif command == "isready":
                print("readyok")
                sys.stdout.flush()
            elif command == "ucinewgame":
                self.reset()
            elif command == "position":
                self.handle_position(tokens[1:])
            elif command == "go":
                # Log who is to move and any time controls the GUI provided
                side = "white" if self.board.turn == chess.WHITE else "black"
                logging.info("GO: side_to_move=%s %s", side, " ".join(tokens[1:]))
                bestmove, _ = self._get_best_move()
                if bestmove is None:
                    print("bestmove 0000")
                else:
                    print(f"bestmove {bestmove}")
                sys.stdout.flush()
            elif command == "quit":
                logging.info("Received quit command")
                break
            # Ignoring other commands (stop, ponder, etc.) for this minimal implementation.


def create_player(opponent: str, gpt_model: str, nanogpt_checkpoint: str):
    if opponent == "gpt":
        player = GPTPlayer(model=gpt_model)
        label = gpt_model
    else:
        player = NanoGptPlayer(model_name=nanogpt_checkpoint)
        label = nanogpt_checkpoint
    return player, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expose ChessGPT as a minimal UCI engine for GUI play."
    )
    parser.add_argument(
        "--opponent",
        choices=["nanogpt", "gpt"],
        default="nanogpt",
        help="Which ChessGPT backend to use.",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo-instruct",
        help="OpenAI model name when --opponent=gpt.",
    )
    parser.add_argument(
        "--nanogpt-checkpoint",
        default=DEFAULT_NANOGPT_MODEL,
        help="Checkpoint filename under nanogpt/out for NanoGPT play.",
    )
    parser.add_argument(
        "--engine-name",
        default="ChessGPT",
        help="Name reported to the GUI via the UCI 'id name' command.",
    )
    args = parser.parse_args()

    player, label = create_player(args.opponent, args.model, args.nanogpt_checkpoint)
    engine = ChessGptUciEngine(player=player, engine_name=f"{args.engine_name} ({label})")
    engine.loop()
