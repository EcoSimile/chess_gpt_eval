#!/usr/bin/env python3

import argparse
import os
import sys
from typing import List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import chess

from nanogpt.nanogpt_module import NanoGptPlayer
from play_vs_chessgptCommandLine import GPTPlayer, LegalMoveResponse, get_legal_move

PROMPT_PATH = os.path.join(ROOT_DIR, "gpt_inputs", "prompt.txt")
DEFAULT_NANOGPT_MODEL = "lichess_200k_bins_16layers_ckpt_with_optimizer.pt"


class ChessGptUciEngine:
    """Minimal UCI bridge that wraps an existing ChessGPT player."""

    def __init__(self, player, engine_name: str, temperature: float = 0.0):
        self.player = player
        self.engine_name = engine_name
        self.temperature = temperature
        self.base_prompt = self._load_prompt()
        self.board = chess.Board()
        self.game_state = self.base_prompt

    def _load_prompt(self) -> str:
        with open(PROMPT_PATH, "r") as f:
            return f.read()

    def reset(self):
        self.board.reset()
        self.game_state = self.base_prompt

    def handle_position(self, tokens: List[str]):
        if not tokens:
            self.reset()
            return

        idx = 0
        token = tokens[idx]
        if token == "startpos":
            self.reset()
            idx += 1
        elif token == "fen":
            fen_tokens = tokens[idx + 1 : idx + 7]
            fen = " ".join(fen_tokens)
            self.board.set_fen(fen)
            # Prompt-based transcripts expect to start from the initial instructions,
            # so when a custom FEN is provided we reuse the base prompt even though
            # historical moves are unknown.
            self.game_state = self.base_prompt
            idx += 7
        else:
            self.reset()

        if idx < len(tokens) and tokens[idx] == "moves":
            self._apply_external_moves(tokens[idx + 1 :])

    def _apply_external_moves(self, moves: List[str]):
        for move_str in moves:
            move = chess.Move.from_uci(move_str)
            if move not in self.board.legal_moves:
                # If the GUI sends an illegal move, just push it to keep in sync
                # and skip prompt updates.
                self.board.push(move)
                continue
            self._append_move_number_if_needed()
            san = self.board.san(move)
            self._append_san_to_game_state(san)
            self.board.push(move)

    def _append_move_number_if_needed(self):
        if self.board.turn == chess.WHITE:
            if self.board.fullmove_number != 1:
                self.game_state += " "
            self.game_state += f"{self.board.fullmove_number}."

    def _append_san_to_game_state(self, san: str):
        if not san.startswith(" "):
            san = " " + san
        self.game_state += san

    def _get_best_move(self) -> Tuple[Optional[str], Optional[str]]:
        self._append_move_number_if_needed()
        result: LegalMoveResponse = get_legal_move(
            self.player,
            self.board,
            self.game_state,
            player_one=self.board.turn == chess.WHITE,
        )
        if result.is_resignation or result.is_illegal_move or result.move_uci is None:
            return None, None
        self._append_san_to_game_state(result.move_san)
        self.board.push(result.move_uci)
        return result.move_uci.uci(), result.move_san

    def loop(self):
        for raw_line in sys.stdin:
            line = raw_line.strip()
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
                bestmove, _ = self._get_best_move()
                if bestmove is None:
                    print("bestmove 0000")
                else:
                    print(f"bestmove {bestmove}")
                sys.stdout.flush()
            elif command == "quit":
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
