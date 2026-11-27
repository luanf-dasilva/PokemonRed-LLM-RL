"""Emulator-only action simulation; never calls the RL environment's step()."""

from dataclasses import dataclass
from io import BytesIO

from pokemon_game.state_encoder import MAP_ID, X_COORD, Y_COORD

# The v2 environment's first four actions are down, left, right, up.
DIRECTIONS = ("down", "left", "right", "up")
BATTLE_FLAG = 0xD057


@dataclass(frozen=True)
class Position:
    map_id: int
    x: int
    y: int
    in_battle: bool


class EmulatorSimulator:
    """Own a PyBoy instance and keep committed and simulated costs separate.

    The instance must not receive input outside this class while snapshots are
    used. Every action releases its key and drains the input queue before save.
    No Python-side RedGymEnv state is touched.
    """
    def __init__(self, pyboy, press_frames=8, release_frames=16, transition_frames=0):
        if press_frames <= 0 or release_frames <= 0 or transition_frames < 0:
            raise ValueError("Invalid action timing")
        self.pyboy = pyboy
        self.press_frames = press_frames
        self.release_frames = release_frames
        self.transition_frames = transition_frames
        self.committed_actions = 0
        self.simulated_actions = 0
        self.committed_frames = 0
        self.simulated_frames = 0

    def observe(self):
        memory = self.pyboy.memory
        return Position(memory[MAP_ID], memory[X_COORD], memory[Y_COORD], bool(memory[BATTLE_FLAG]))

    def snapshot(self):
        buffer = BytesIO()
        self.pyboy.save_state(buffer)
        return buffer.getvalue()

    def restore(self, snapshot):
        self.pyboy.load_state(BytesIO(snapshot))

    def step(self, action, *, simulated=False):
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < len(DIRECTIONS):
            raise ValueError("Action must be a direction index from 0 to 3")
        before = self.observe()
        if before.in_battle:
            raise RuntimeError("Navigation simulator cannot act in battle")
        button = DIRECTIONS[action]
        self.pyboy.button_press(button)
        self.pyboy.tick(self.press_frames, False)
        self.pyboy.button_release(button)
        if self.release_frames > 1:
            self.pyboy.tick(self.release_frames - 1, False)
        # Match the v2 environment's final rendered frame and timing.
        self.pyboy.tick(1, True)
        frames = self.press_frames + self.release_frames
        after = self.observe()
        if after.map_id != before.map_id and self.transition_frames:
            self.pyboy.tick(self.transition_frames, False)
            frames += self.transition_frames
            after = self.observe()
        if simulated:
            self.simulated_actions += 1
            self.simulated_frames += frames
        else:
            self.committed_actions += 1
            self.committed_frames += frames
        return after
