"""Play and record Pokémon Red using discrete actions in the v2 Gym environment."""
import argparse
import hashlib
import json
from pathlib import Path

from pokemon_game.demonstrations.environment import ROOT, enter_v2, make_env
from pokemon_game.demonstrations.recorder import DemonstrationRecorder
from pokemon_game.state_encoder import load_lookups

KEYS = {'Down': 0, 'Left': 1, 'Right': 2, 'Up': 3, 'z': 4, 'Z': 4,
        'x': 5, 'X': 5, 'Return': 6}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_start(start_state, curriculum_archive):
    if curriculum_archive is None:
        return Path(start_state).resolve()
    directory = Path(curriculum_archive).resolve()
    manifest = json.loads((directory / 'current.json').read_text())
    entry = manifest['states']['worst']
    filename = entry['filename']
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError('Invalid curriculum snapshot filename')
    path = directory / filename
    if sha(path) != entry['sha256']:
        raise ValueError('Curriculum snapshot hash mismatch')
    return path


def run(rom, start_state, output, max_steps=100000, target_map=None, curriculum_archive=None):
    import tkinter as tk
    from PIL import Image, ImageTk

    rom, output = Path(rom).resolve(), Path(output).resolve()
    start = resolve_start(start_state, curriculum_archive)
    lookups = load_lookups(ROOT / 'pokemon_rag/data/pret-pokered/processed/lookups.json')
    enter_v2()
    env = make_env(rom, start, output.parent / (output.name + '-sessions'), max_steps, target_map)
    recorder = None
    try:
        recorder = DemonstrationRecorder(env, output, lookups, 'human',
                                         rom_sha256=sha(rom), start_sha256=sha(start))
        recorder.manifest['goal_map_id'] = target_map
        recorder.reset()
        window = tk.Tk()
        window.title('Pokémon Red demonstration recorder')
        canvas = tk.Label(window)
        canvas.pack()
        stats = tk.StringVar()
        tk.Label(window, textvariable=stats, justify='left', font=('monospace', 12)).pack()
        tk.Label(window, text='Arrows: move   Z: A   X: B   Enter: Start   S: snapshot   R: reset   Q: finish').pack()
        closed = False
        last_reward = 0.0

        def refresh():
            state = recorder._state()
            health = state['hp_fraction']
            hp = 'unknown' if health is None else f'{health:.0%}'
            stats.set(f"Location: {state['location']}  ({state['x']}, {state['y']})  Mode: {state['mode']}\n"
                      f"Party health: {hp}  Levels: {state['party_levels']}  Badges: {state['badges']}\n"
                      f"Session reward: {recorder.reward_total:.2f}  Last reward: {last_reward:.2f}  "
                      f"Transitions: {recorder.manifest['transitions']}\n"
                      + ('Episode ended; press R to start another.' if recorder.obs is None else ''))
            picture = env.unwrapped.pyboy.screen.image.convert('RGB')
            picture = picture.resize((640, 576), Image.Resampling.NEAREST)
            canvas.image = ImageTk.PhotoImage(picture)
            canvas.configure(image=canvas.image)

        def finish():
            nonlocal closed
            if not closed:
                closed = True
                recorder.finish('human_stopped')
                window.destroy()

        def key(event):
            nonlocal last_reward
            if event.keysym in ('q', 'Q', 'Escape'):
                finish()
            elif event.keysym in ('s', 'S'):
                recorder.snapshot('manual')
                refresh()
            elif event.keysym in ('r', 'R') and recorder.obs is None:
                recorder.reset()
                last_reward = 0.0
                refresh()
            elif event.keysym in KEYS and recorder.obs is not None:
                _, last_reward, _, _, _ = recorder.step(KEYS[event.keysym])
                refresh()

        window.bind('<KeyPress>', key)
        window.protocol('WM_DELETE_WINDOW', finish)
        refresh()
        window.mainloop()
    finally:
        if recorder is not None:
            recorder.close()
        else:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rom', type=Path, default=ROOT / 'PokemonRed.gb')
    parser.add_argument('--start-state', type=Path, default=ROOT / 'init.state')
    parser.add_argument('--curriculum-archive', type=Path,
                        help='Use the current checksummed worst snapshot from this archive')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=100000)
    parser.add_argument('--target-map', type=int)
    args = parser.parse_args()
    run(args.rom, args.start_state, args.output, args.max_steps, args.target_map, args.curriculum_archive)


if __name__ == '__main__':
    main()
