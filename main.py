import chess
import chess.engine
import os
import csv
import random
import time
import platform
import subprocess
import math
import fcntl
#added this to be able to access the stockfish i install on fedora
import os, shutil
import chess.engine

ENGINE = (shutil.which("stockfish")
          or os.environ.get("STOCKFISH_PATH")
          or "/usr/bin/stockfish")   # last-ditch default

# NOTE: Avoid eager Stockfish startup on import because python-chess + Py3.13 was hanging here.
# engine = chess.engine.SimpleEngine.popen_uci(ENGINE)



# NOTE: LLAMA AND NANOGPT ARE EXPERIMENTAL PLAYERS that most people won't need to use
# They are commented by default to avoid unnecessary dependencies such as pytorch.
# from llama_module import BaseLlamaPlayer, LocalLlamaPlayer, LocalLoraLlamaPlayer
from nanogpt.nanogpt_module import NanoGptPlayer
import gpt_query

from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class LegalMoveResponse:
    move_san: Optional[str] = None
    move_uci: Optional[chess.Move] = None
    attempts: int = 0
    is_resignation: bool = False
    is_illegal_move: bool = False
    is_timeout: bool = False


# Define base Player class
class Player:
    def get_move(self, board: chess.Board, game_state: str, temperature: float) -> str:
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError


class GPTPlayer(Player):
    def __init__(self, model: str):
        import openai

        with open("gpt_inputs/api_key.txt", "r") as f:
            openai.api_key = f.read().strip()
        self.model = model
        self._openai = openai  # keep a reference so lazy imports don't get GC'd

    def get_move(
        self, board: chess.Board, game_state: str, temperature: float
    ) -> Optional[str]:
        response = get_gpt_response(game_state, self.model, temperature)
        return get_move_from_gpt_response(response)

    def get_config(self) -> dict:
        return {"model": self.model}


class StockfishPlayer(Player):
    @staticmethod
    def get_stockfish_path() -> str:
        """
        Determines the operating system and returns the appropriate path for Stockfish.

        Returns:
            str: Path to the Stockfish executable based on the operating system.
        """
        if platform.system() == "Linux":
            return "/usr/games/stockfish"
        elif platform.system() == "Darwin":  # Darwin is the system name for macOS
            return "stockfish"
        elif platform.system() == "Windows":
            return (
                r"C:\Users\adamk\Documents\Stockfish\stockfish-windows-x86-64-avx2.exe"
            )
        else:
            raise OSError("Unsupported operating system")

    def __init__(self, skill_level: int, play_time: float):
        self._skill_level = skill_level
        self._play_time = play_time
        self._engine = None  # kept for reference to the original API usage
        # If getting started, you need to run brew install stockfish
        stockfish_path = StockfishPlayer.get_stockfish_path()
        # Original chess.engine.SimpleEngine approach (left commented because it hangs on this setup):
        # self._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self._proc = self._start_uci_engine(stockfish_path)

    def _start_uci_engine(self, stockfish_path: str) -> subprocess.Popen:
        proc = subprocess.Popen(
            [stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # UCI handshake
        self._send(proc, "uci")
        self._read_until(proc, "uciok", timeout=5)
        # Configure strength via Skill Level
        skill = max(0, min(20, self._skill_level))
        self._send(proc, f"setoption name Skill Level value {skill}")
        return proc

    def _send(self, proc: subprocess.Popen, line: str):
        if proc.stdin:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()

    def _read_until(self, proc: subprocess.Popen, token: str, timeout: float = 5.0):
        start = time.time()
        while True:
            if proc.stdout is None:
                raise RuntimeError("Stockfish stdout closed unexpectedly")
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("Stockfish terminated during read")
            if line.strip() == token:
                return
            if time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for {token}, last line: {line.strip()}")

    def _read_bestmove(self, proc: subprocess.Popen, timeout: float = 10.0) -> Optional[str]:
        start = time.time()
        while True:
            if proc.stdout is None:
                return None
            line = proc.stdout.readline()
            if not line:
                return None
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) >= 2 else None
            if time.time() - start > timeout:
                return None

    def get_move(
        self, board: chess.Board, game_state: str, temperature: float
    ) -> Optional[str]:
        if self._skill_level == -2:
            legal_moves = list(board.legal_moves)
            random_move = random.choice(legal_moves)
            return board.san(random_move)

        if self._proc is None:
            return None

        # Set position and request a move.
        try:
            self._send(self._proc, f"position fen {board.fen()}")
            if self._skill_level < 0:
                # Lowest strength equivalent: one-node search
                self._send(self._proc, "go nodes 1")
            else:
                movetime_ms = max(1, int(self._play_time * 1000))
                self._send(self._proc, f"go movetime {movetime_ms}")
            uci_move = self._read_bestmove(
                self._proc,
                timeout=max(2.0, self._play_time * 5 + 2),
            )
        except Exception as e:
            print(f"Stockfish error: {e}")
            return None

        if not uci_move or uci_move == "(none)":
            return None

        try:
            move_obj = chess.Move.from_uci(uci_move)
            if move_obj not in board.legal_moves:
                print(f"Stockfish returned illegal move {uci_move} for FEN {board.fen()}; skipping move.")
                return None
            return board.san(move_obj)
        except Exception as e:
            print(f"Failed to parse Stockfish move {uci_move}: {e}")
            return None

    def get_config(self) -> dict:
        return {"skill_level": self._skill_level, "play_time": self._play_time}

    def close(self):
        if self._engine is not None:
            self._engine.quit()
        if hasattr(self, "_proc") and self._proc:
            try:
                self._send(self._proc, "quit")
                self._proc.wait(timeout=2)
            except Exception:
                pass

    def reset_clock(self):
        pass


def describe_player(player: Player) -> Tuple[str, Optional[float]]:
    cfg = player.get_config()
    if "model" in cfg:  # GPT / NanoGPT side
        model = cfg.get("model", "ChessGPT")
        base = os.path.basename(str(model))
        if base.endswith(".pt"):
            base = base[:-3]
        return f"ChessGPT ({base})", None
    skill = cfg.get("skill_level", "?")
    return f"Stockfish level {skill}", cfg.get("play_time")


def get_gpt_response(game_state: str, model: str, temperature: float) -> Optional[str]:
    import gpt_query

    # trying to prevent what I believe to be rate limit issues
    if model == "gpt-4":
        time.sleep(0.4)
    return gpt_query.get_gpt_response(game_state, model, temperature)


def get_move_from_gpt_response(response: Optional[str]) -> Optional[str]:
    if response is None:
        return None

    # Parse the response to get only the first move
    moves = response.split()
    first_move = moves[0] if moves else None

    return first_move


def build_result_info(
    board: chess.Board,
    player_one: Player,
    player_two: Player,
    primary_is_white: bool,
    game_state: str,
    player_one_illegal_moves: int,
    player_two_illegal_moves: int,
    player_one_legal_moves: int,
    player_two_legal_moves: int,
    total_time: float,
    player_one_resignation: bool,
    player_two_resignation: bool,
    player_one_failed_to_find_legal_move: bool,
    player_two_failed_to_find_legal_move: bool,
    total_moves: int,
    illegal_moves: int,
    player_one_timeout: bool = False,
    player_two_timeout: bool = False,
):
    unique_game_id = generate_unique_game_id()

    (
        player_one_title,
        player_two_title,
        player_one_time,
        player_two_time,
    ) = get_player_titles_and_time(player_one, player_two)

    if player_one_resignation or player_one_failed_to_find_legal_move:
        result = "0-1"
        player_one_score = 0
        player_two_score = 1
    elif player_two_resignation or player_two_failed_to_find_legal_move:
        result = "1-0"
        player_one_score = 1
        player_two_score = 0
    else:
        result = board.result()
        # Hmmm.... debating this one. Annoying if I leave it running and it fails here for some reason, probably involving some
        # resignation / failed move situation I didn't think of
        # -1e10 at least ensures it doesn't fail silently
        if "-" in result:
            player_one_score = result.split("-")[0]
            player_two_score = result.split("-")[1]
        elif result == "*":  # Unresolved (e.g., max moves) -> leave result blank for clarity
            result = ""
            player_one_score = -1e10
            player_two_score = -1e10
        else:
            player_one_score = -1e10
            player_two_score = -1e10
    # Ensure numeric scores for downstream calculations, handling 1/2 strings
    p1 = parse_score(str(player_one_score))
    p2 = parse_score(str(player_two_score))
    player_one_score = p1 if p1 is not None else -1e10
    player_two_score = p2 if p2 is not None else -1e10

    # Map the score/flags to the primary (NanoGPT) player regardless of color.
    if primary_is_white:
        primary_player = player_one_title
        primary_score = player_one_score
        primary_illegal_moves = player_one_illegal_moves
        primary_legal_moves = player_one_legal_moves
        primary_timeout = player_one_timeout
        primary_resignation = player_one_resignation
        primary_failed = player_one_failed_to_find_legal_move
    else:
        primary_player = player_two_title
        primary_score = player_two_score
        primary_illegal_moves = player_two_illegal_moves
        primary_legal_moves = player_two_legal_moves
        primary_timeout = player_two_timeout
        primary_resignation = player_two_resignation
        primary_failed = player_two_failed_to_find_legal_move

    info_dict = {
        "game_id": unique_game_id,
        "transcript": game_state,
        "result": result,
        "player_one": player_one_title,
        "player_two": player_two_title,
        "player_one_time": player_one_time,
        "player_two_time": player_two_time,
        "player_one_score": player_one_score,
        "player_two_score": player_two_score,
        "primary_player": primary_player,
        "primary_player_score": primary_score,
        "primary_player_illegal_moves": primary_illegal_moves,
        "primary_player_legal_moves": primary_legal_moves,
        "primary_player_timeout": primary_timeout,
        "primary_player_resignation": primary_resignation,
        "primary_player_failed_to_find_legal_move": primary_failed,
        "player_one_illegal_moves": player_one_illegal_moves,
        "player_two_illegal_moves": player_two_illegal_moves,
        "player_one_legal_moves": player_one_legal_moves,
        "player_two_legal_moves": player_two_legal_moves,
        "player_one_resignation": player_one_resignation,
        "player_two_resignation": player_two_resignation,
        "player_one_failed_to_find_legal_move": player_one_failed_to_find_legal_move,
        "player_two_failed_to_find_legal_move": player_two_failed_to_find_legal_move,
        "game_title": f"{player_one_title} vs. {player_two_title}",
        "number_of_moves": board.fullmove_number,
        "time_taken": total_time,
        "total_moves": total_moves,
        "illegal_moves": illegal_moves,
    }
    return player_one_score, player_two_score, info_dict


def write_result_row(info_dict: dict):
    csv_file_path = get_recording_path()
    write_headers = not os.path.exists(csv_file_path)
    lock_path = csv_file_path + ".lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        with open(csv_file_path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=info_dict.keys())
            if write_headers:
                writer.writeheader()
            writer.writerow(info_dict)
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    # Write game transcript to a process-specific file to avoid collisions.
    game_txt = f"game.txt.{os.getpid()}"
    with open(game_txt, "w") as f:
        f.write(info_dict["transcript"])


def record_results(
    board: chess.Board,
    player_one: Player,
    player_two: Player,
    primary_is_white: bool,
    game_state: str,
    player_one_illegal_moves: int,
    player_two_illegal_moves: int,
    player_one_legal_moves: int,
    player_two_legal_moves: int,
    total_time: float,
    player_one_resignation: bool,
    player_two_resignation: bool,
    player_one_failed_to_find_legal_move: bool,
    player_two_failed_to_find_legal_move: bool,
    total_moves: int,
    illegal_moves: int,
    player_one_timeout: bool = False,
    player_two_timeout: bool = False,
):
    # Backward-compatible wrapper: build info and write immediately.
    p1, p2, info = build_result_info(
        board,
        player_one,
        player_two,
        primary_is_white,
        game_state,
        player_one_illegal_moves,
        player_two_illegal_moves,
        player_one_legal_moves,
        player_two_legal_moves,
        total_time,
        player_one_resignation,
        player_two_resignation,
        player_one_failed_to_find_legal_move,
        player_two_failed_to_find_legal_move,
        total_moves,
        illegal_moves,
        player_one_timeout,
        player_two_timeout,
    )
    write_result_row(info)
    return p1, p2

def generate_unique_game_id() -> str:
    timestamp = int(time.time())
    random_num = random.randint(1000, 9999)  # 4-digit random number
    return f"{timestamp}-{random_num}"


def get_recording_path() -> str:
    if RUN_FOR_ANALYSIS:
        csv_file_path = f"logs/{player_one_recording_name}_vs_{player_two_recording_name}"
        csv_file_path = csv_file_path.replace(
            ".", "_"
        )  # filenames can't have periods in them. Useful for e.g. gpt-3.5 models
        csv_file_path += ".csv"
    else:
        csv_file_path = recording_file
    return csv_file_path


def parse_score(val: str) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        cleaned = val.strip()
        if cleaned in {"1/2", ".5", "0.5"}:
            return 0.5
    return None


def get_player_titles_and_time(
    player_one: Player, player_two: Player
) -> Tuple[str, str, Optional[float], Optional[float]]:
    p1_title, p1_time = describe_player(player_one)
    p2_title, p2_time = describe_player(player_two)

    return (p1_title, p2_title, p1_time, p2_time)


def initialize_game_with_opening(
    game_state: str, board: chess.Board
) -> Tuple[str, chess.Board]:
    with open("openings.csv", "r") as file:
        lines = file.readlines()[1:]  # Skip header
    moves_string = random.choice(lines)
    game_state += moves_string
    # Splitting the moves string on spaces
    tokens = moves_string.split()

    for token in tokens:
        # If the token contains a period, it's a move number + move combination
        if "." in token:
            move = token.split(".")[-1]  # Take the move part after the period
        else:
            move = token

        board.push_san(move)
    return game_state, board


# Return is (move_san, move_uci, attempts, is_resignation, is_illegal_move)
def get_legal_move(
    player: Player,
    board: chess.Board,
    game_state: str,
    player_one: bool,
    max_attempts: int = 5,
    max_move_time: Optional[float] = None,
) -> LegalMoveResponse:
    """Request a move from the player and ensure it's legal."""
    move_san = None
    move_uci = None

    attempt_start_time = time.time()

    for attempt in range(max_attempts):
        move_san = player.get_move(
            board, game_state, min(((attempt / max_attempts) * 1) + 0.001, 0.5)
        )
        if max_move_time is not None and (time.time() - attempt_start_time) > max_move_time:
            print(f"Move timed out after {time.time() - attempt_start_time:.2f}s")
            return LegalMoveResponse(
                move_san=None,
                move_uci=None,
                attempts=1,  # count a timeout as a single failed attempt
                is_resignation=False,
                is_illegal_move=True,
                is_timeout=True,
            )

        # Sometimes when GPT thinks it's the end of the game, it will just output the result
        # Like "1-0". If so, this really isn't an illegal move, so we'll add a check for that.
        if move_san is not None:
            if move_san == "1-0" or move_san == "0-1" or move_san == "1/2-1/2":
                print(f"{move_san}, player has resigned")
                return LegalMoveResponse(
                    move_san=None,
                    move_uci=None,
                    attempts=attempt,
                    is_resignation=True,
                )

        if move_san is None:
            print("No move returned; retrying")
            continue

        try:
            move_uci = board.parse_san(move_san)
        except Exception as e:
            print(f"Error parsing move {move_san}: {e}")
            # check if player is gpt-3.5-turbo-instruct
            # only recording errors for gpt-3.5-turbo-instruct because it's errors are so rare
            cfg = player.get_config()
            if cfg.get("model") == "gpt-3.5-turbo-instruct":
                with open("gpt-3.5-turbo-instruct-illegal-moves.txt", "a") as f:
                    f.write(f"{game_state}\n{move_san}\n")
            continue

        if move_uci in board.legal_moves:
            if not move_san.startswith(" "):
                move_san = " " + move_san
            return LegalMoveResponse(move_san, move_uci, attempt)
        print(f"Illegal move: {move_san}")

    # If we reach here, the player has made illegal moves for all attempts.
    print(f"{player} provided illegal moves for {max_attempts} attempts.")
    return LegalMoveResponse(
        move_san=None, move_uci=None, attempts=max_attempts, is_illegal_move=True
    )


def play_turn(
    player: Player, board: chess.Board, game_state: str, player_one: bool
) -> Tuple[str, bool, bool, int, bool]:
    max_move_time = None
    if player_one and isinstance(player, NanoGptPlayer):
        max_move_time = 30.0  # fail fast if NanoGPT hangs
    result = get_legal_move(player, board, game_state, player_one, 5, max_move_time)
    illegal_moves = result.attempts
    move_san = result.move_san
    move_uci = result.move_uci
    resignation = result.is_resignation
    failed_to_find_legal_move = result.is_illegal_move
    timed_out = result.is_timeout

    if resignation:
        print(f"{player} resigned with result: {board.result()}")
    elif failed_to_find_legal_move:
        if timed_out:
            print(f"Game over: timeout from {player}")
        else:
            print(f"Game over: 5 consecutive illegal moves from {player}")
    elif move_san is None or move_uci is None:
        print(f"Game over: {player} failed to find a legal move")
    else:
        board.push(move_uci)
        game_state += move_san
        print(move_san, end=" ")

    return game_state, resignation, failed_to_find_legal_move, illegal_moves, timed_out


def initialize_game_with_random_moves(
    board: chess.Board, initial_game_state: str, randomize_opening_moves: int
) -> tuple[str, chess.Board]:
    # We loop for multiple attempts because sometimes the random moves will result in a game over
    MAX_INIT_ATTEMPTS = 5
    for attempt in range(MAX_INIT_ATTEMPTS):
        board.reset()  # Reset the board for a new attempt
        game_state = initial_game_state  # Reset the game state for a new attempt
        moves = []
        for moveIdx in range(1, randomize_opening_moves + 1):
            for player in range(2):
                moves = list(board.legal_moves)
                if not moves:
                    break  # Break if no legal moves are available

                move = random.choice(moves)
                moveString = board.san(move)
                if moveIdx > 1 or player == 1:
                    game_state += " "
                game_state += (
                    str(moveIdx) + ". " + moveString if player == 0 else moveString
                )
                board.push(move)

            if not moves:
                break  # Break if no legal moves are available

        if moves:
            # Successful generation of moves, break out of the attempt loop
            break
    else:
        # If the loop completes without a break, raise an error
        raise Exception("Failed to initialize the game after maximum attempts.")

    print(game_state)
    return game_state, board


def play_game(
    player_one: Player,
    player_two: Player,
    max_games: int = 10,
    randomize_opening_moves: Optional[int] = None,
    alternate_colors: bool = True,
):
    # NOTE: I'm being very particular with game_state formatting because I want to match the PGN notation exactly
    # It looks like this: 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 etc. HOWEVER, GPT prompts should not end with a trailing whitespace
    # due to tokenization issues. If you make changes, ensure it still matches the PGN notation exactly.
    run_start_time = time.time()
    cumulative_p1_score = 0.0
    anchor_rating = None
    if isinstance(player_two, StockfishPlayer):
        cfg = player_two.get_config()
        anchor_rating = STOCKFISH_ELO_TABLE.get(cfg["skill_level"])
    valid_games = 0

    # Resume from existing log if present.
    csv_path = get_recording_path()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Prefer the primary-player score if present; fall back to player_one_score for backward compatibility.
                key = "primary_player_score" if "primary_player_score" in row else "player_one_score"
                val = parse_score(row.get(key, ""))
                if val is None or not (0.0 <= val <= 1.0):
                    continue
                cumulative_p1_score += val
                valid_games += 1
        if valid_games > 0:
            running_p = cumulative_p1_score / valid_games
            if anchor_rating is not None:
                p_clamped = min(max(running_p, 1e-6), 1 - 1e-6)
                delta = -400 * math.log10(1 / p_clamped - 1)
                elo_estimate = anchor_rating + delta
                print(
                    f"Resuming Elo from {valid_games} prior game(s) in {csv_path}: score {running_p:.3f}, Elo {elo_estimate:.1f} vs anchor {anchor_rating}"
                )
            else:
                print(f"Resuming score from {valid_games} prior game(s) in {csv_path}: {running_p:.3f}")
    p1_cfg, p2_cfg, p1_time, p2_time = get_player_titles_and_time(player_one, player_two)
    print(
        f"Starting match: {p1_cfg} (t={p1_time}) vs {p2_cfg} (t={p2_time}) for {max_games} games; MAX_MOVES={MAX_MOVES}"
    )

    for game_idx in range(max_games):  # Play max_games games
        board = chess.Board()

        if alternate_colors and game_idx % 2 == 1:
            white_player, black_player = player_two, player_one
        else:
            white_player, black_player = player_one, player_two
        primary_is_white = white_player is player_one

        # Build PGN headers with the actual player titles for this game.
        white_title, _ = describe_player(white_player)
        black_title, _ = describe_player(black_player)
        game_state = f'[White "{white_title}"]\n[Black "{black_title}"]\n\n'

        if randomize_opening_moves is not None:
            game_state, board = initialize_game_with_random_moves(
                board, game_state, randomize_opening_moves
            )

        white_illegal_moves = 0
        black_illegal_moves = 0
        white_legal_moves = 0
        black_legal_moves = 0
        player_one_resignation = False
        player_two_resignation = False
        player_one_failed_to_find_legal_move = False
        player_two_failed_to_find_legal_move = False
        player_one_timeout = False
        player_two_timeout = False
        start_time = time.time()
        print(f"\n--- Game {game_idx + 1}/{max_games} ---")

        total_moves = 0
        illegal_moves = 0

        while not board.is_game_over():
            with open("game.txt", "w") as f:
                f.write(game_state)
            current_move_num = str(board.fullmove_number) + "."
            total_moves += 1
            # I increment legal moves here so player_two isn't penalized for the game ending before its turn
            white_legal_moves += 1
            black_legal_moves += 1

            # this if statement may be overkill, just trying to get format to exactly match PGN notation
            if board.fullmove_number != 1:
                game_state += " "
            game_state += current_move_num
            print(f"{current_move_num}", end="")

            (
                game_state,
                player_one_resignation,
                player_one_failed_to_find_legal_move,
                illegal_moves_one,
                timeout_one,
            ) = play_turn(white_player, board, game_state, player_one=True)
            white_illegal_moves += illegal_moves_one
            if illegal_moves_one != 0:
                white_legal_moves -= 1
                illegal_moves += illegal_moves_one
            if timeout_one:
                player_one_timeout = True
            if (
                board.is_game_over()
                or player_one_resignation
                or player_one_failed_to_find_legal_move
            ):
                break

            (
                game_state,
                player_two_resignation,
                player_two_failed_to_find_legal_move,
                illegal_moves_two,
                timeout_two,
            ) = play_turn(black_player, board, game_state, player_one=False)
            black_illegal_moves += illegal_moves_two
            if illegal_moves_two != 0:
                black_legal_moves -= 1
                illegal_moves += illegal_moves_two
            if timeout_two:
                player_two_timeout = True
            if (
                board.is_game_over()
                or player_two_resignation
                or player_two_failed_to_find_legal_move
            ):
                break

            print("\n", end="")

            if total_moves > MAX_MOVES:
                break

        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nGame over. Total time: {total_time:.2f} seconds")
        print(f"Result: {board.result()}")
        print(board)
        print()
        white_score, black_score, info = build_result_info(
            board,
            white_player,
            black_player,
            primary_is_white,
            game_state,
            white_illegal_moves,
            black_illegal_moves,
            white_legal_moves,
            black_legal_moves,
            total_time,
            player_one_resignation,
            player_two_resignation,
            player_one_failed_to_find_legal_move,
            player_two_failed_to_find_legal_move,
            total_moves,
            illegal_moves,
            player_one_timeout,
            player_two_timeout,
        )
        p1_score = white_score if primary_is_white else black_score
        if 0.0 <= p1_score <= 1.0:
            cumulative_p1_score += p1_score
            valid_games += 1
            running_p = cumulative_p1_score / valid_games
            if anchor_rating is not None:
                p_clamped = min(max(running_p, 1e-6), 1 - 1e-6)
                delta = -400 * math.log10(1 / p_clamped - 1)
                elo_estimate = anchor_rating + delta
                info["primary_player_elo"] = elo_estimate
                print(
                    f"Running score: {running_p:.3f} over {valid_games} valid game(s) | Elo vs Stockfish anchor {anchor_rating}: {elo_estimate:.1f}"
                )
            else:
                print(f"Running score: {running_p:.3f} over {valid_games} valid game(s)")
        else:
            print(f"Skipping Elo update for game {game_idx + 1}: invalid score {p1_score}")
        write_result_row(info)
        print(
            f"Finished game {game_idx + 1}/{max_games} | elapsed this game: {total_time:.2f}s | total run so far: {time.time() - run_start_time:.2f}s"
        )
    if isinstance(player_one, StockfishPlayer):
        player_one.close()
    if isinstance(player_two, StockfishPlayer):
        player_two.close()

    print(f"Match complete in {time.time() - run_start_time:.2f} seconds.")

        # print(game_state)


NANOGPT = True
RUN_FOR_ANALYSIS = True
MAX_MOVES = 1000
if NANOGPT:
    MAX_MOVES = 89  # Due to nanogpt max input length of 1024

# Stockfish approximate Elo anchors from README (skill level -> rating)
STOCKFISH_ELO_TABLE = {
    0: 1320.1,
    1: 1467.6,
    2: 1608.4,
    3: 1742.3,
    4: 1922.9,
    5: 2203.7,
    6: 2363.2,
    7: 2499.5,
    8: 2596.2,
    9: 2702.8,
    10: 2788.3,
    11: 2855.5,
    12: 2923.1,
    13: 2972.9,
    14: 3024.8,
    15: 3069.5,
    16: 3111.2,
    17: 3141.3,
    18: 3170.3,
    19: 3191.1,
}
recording_file = os.environ.get(
    "RECORDING_FILE",
    "logs/lichess_200k_bins_16layers_vs_stockfish_level_4_2s.csv",  # fallback when RUN_FOR_ANALYSIS is False
)
player_one_recording_name = os.environ.get(
    "PLAYER_ONE_RECORDING_NAME", "CGPT_run1"
)
player_two_recording_name = os.environ.get(
    "PLAYER_TWO_RECORDING_NAME", "stockfish_level_4_2s"
)
if __name__ == "__main__":
    num_games = 35
    player_one = NanoGptPlayer(
        model_name="lichess_200k_bins_16layers_ckpt_with_optimizer.pt"
    )
    # Fixed Stockfish level/time for this run (overridable via env)
    stockfish_skill = int(os.environ.get("STOCKFISH_SKILL", 4))
    stockfish_time = 2.0
    player_two = StockfishPlayer(skill_level=stockfish_skill, play_time=stockfish_time)

    play_game(player_one, player_two, num_games)

# Original sweep loop kept for reference:
# recording_file = "logs/determine.csv"  # default recording file. Because we are using list [player_ones], recording_file is overwritten
# player_ones = ["stockfish_16layers_ckpt_no_optimizer.pt"]
# player_ones = ["gpt-3.5-turbo-instruct"]
# player_two_recording_name = "stockfish_sweep"
# if __name__ == "__main__":
#     for player in player_ones:
#         player_one_recording_name = player
#         for i in range(11):
#             num_games = 3
#             # player_one = GPTPlayer(model=player)
#             # player_one = GPTPlayer(model="gpt-4")
#             # player_one = StockfishPlayer(skill_level=-1, play_time=0.1)
#             player_one = NanoGptPlayer(model_name="lichess_200k_bins_16layers_ckpt_with_optimizer.pt")
#             # NanoGptPlayer(model_name=player_ones) #(model_name=player_one_recording_name)
#             player_two = StockfishPlayer(skill_level=i, play_time=0.1)
#             # player_two = GPTPlayer(model="gpt-4")
#             # player_two = GPTPlayer(model="gpt-3.5-turbo-instruct")
#
#             play_game(player_one, player_two, num_games)
