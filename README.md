# Fly-Brain Flappy Bird

A single simulated fruit fly learns to play Flappy Bird — for real, while you watch.

There's no genetic algorithm here, no population of 50 birds trying at once. It's one fly, living one life after another. Every time it crashes, it's dead for about a tenth of a second, then it respawns and tries again — and it remembers everything it learned from the crash before. Watch it for a couple of minutes and you'll see it go from bouncing around randomly to actually threading pipes on purpose.

The interesting part isn't the game — it's *why* the fly gets better. It's not "trained" in the usual deep-learning sense. It uses a simplified version of how real fruit flies are believed to learn: a brain region called the mushroom body, where a chemical signal called dopamine tells a handful of neurons "that was good, do more of that" or "that was bad, don't do that again." Passing a pipe releases a little jolt of the "good" signal (real flies have a neuron type called PAM for this). Crashing releases the "bad" signal (real flies call this PPL1). Over time, those signals reshape the connections in its brain, and the fly's behavior actually changes.

## What you'll see on screen

The window is split in two:

- **Left side — the game.** A normal-looking Flappy Bird. The fly is the yellow dot. A white ring flashes around it when it flaps; a red ring means its "emergency reflex" just kicked in to save it from a wall it was about to hit (more on that below).
- **Right side — the fly's brain, live.** A stylized rendering of its head (the two teal blobs are the optic lobes, the brownish blob in the middle is the mushroom body), with little particles flickering around to show neurons firing. Below that is a stack of readouts: a learning curve tracking its score over time, its current "gut feeling" about whether to flap, a small grid showing what it's learned to do in different situations, a list of the (simplified) neuron populations involved, and some stats on how much its brain has actually changed.

It's genuinely fun to just leave it running and watch the learning curve creep upward.

## How the fly "senses" the game

It doesn't see the screen the way you do. Every instant, it gets a handful of numbers:

- **Vision** — how far away the next pipe is, and how far off-center it is from the gap, plus its own falling/rising speed.
- **Scent** — the gap "smells" like food, and the smell gets stronger the closer the fly gets to it. This is a nod to how real flies actually navigate a lot of the world: by following odor gradients, not just by sight.
- **A "danger" reflex** — if it's about to slam into the ground, the ceiling, or a pipe, a hard-wired reflex (loosely inspired by the fly's real giant fiber neuron, which triggers its fastest escape reactions) can override its learned behavior and force the right emergency move. This is why it rarely dies in an obviously dumb way — the *learning* is mostly what determines whether it can actually thread the needle through a gap, not whether it face-plants into the floor.

## Requirements

- Python 3.9 or newer
- macOS, Linux, or Windows (it's just Pygame + PyTorch, nothing platform-specific)

## Setup

Clone the repo and set up a virtual environment so the dependencies stay tidy:

```bash
git clone https://github.com/EzHook/fly-brain-flappy-bird.git
cd fly-brain-flappy-bird
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

That installs Pygame (for the game/graphics) and PyTorch (for the little neural network that is the fly's brain).

## Running it

```bash
python main.py
```

A window opens and the fly starts flying (badly, at first) immediately. No setup screen, no menu — it just goes.

## Controls

| Key | What it does |
|---|---|
| `T` | Turbo mode — fast-forwards the simulation so you can watch hundreds of lifetimes fly by in seconds instead of waiting for it to learn in real time |
| `P` | Pause / resume |
| `S` | Save the fly's brain to disk (`fly_brain_save.pt`), so it doesn't forget everything when you close the window |
| `L` | Load a previously saved brain |
| `F` | Forget — wipe its learned synapses and start over from scratch, naive |
| `Esc` | Quit |

**Tip:** if you just want to *see* it get good fast, hit `T` for turbo the moment it opens and let it run for 20–30 seconds. Then hit `T` again to drop back to normal speed and watch it play for real.

## A note on realism

This is a toy model, not a research-grade simulation of an actual fly brain — there's no claim here that this is biologically precise. The neuron counts, the wiring, and the learning rule are all drastically simplified so that something recognizable as "a fly learning" can happen live, on screen, in a couple of minutes, instead of needing a supercomputer and a research paper. The parts that *are* deliberately modeled on the real biology are the shape of the idea: sparse random coding in the mushroom body, a small set of output neurons whose balance decides behavior, and dopamine neurons that reshape those specific connections based on reward and punishment — because that's genuinely how researchers believe real fruit flies learn.

## Project structure

```
main.py           entry point — the game loop, rendering, keyboard controls
fly_brain.py       the fly's brain: senses -> Kenyon cells -> valence -> flap, and the learning rule
bird_game.py        Flappy Bird physics: gravity, pipes, collisions, scent, the danger reflex
visualization.py     everything drawn on the right-hand side of the screen
```
