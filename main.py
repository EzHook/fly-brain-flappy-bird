"""
Fly-Brain Flappy Bird
======================
A Flappy Bird agent controlled by a simplified fruit-fly mushroom-body
learning circuit: sensory input (vision + scent) projects through a frozen
random wiring into sparse Kenyon cells, which drive two MBONs (approach /
avoid) through PLASTIC synapses. Dopamine signals -- PAM for reward (a pipe
cleared, or the gap's "smell" growing stronger), PPL1 for punishment (a
collision) -- reshape those synapses live, so the one fly on screen visibly
gets better over its (endless sequence of) lives. See fly_brain.py for the
learning rule itself.

Run:
    python main.py

Controls:
    T   - toggle turbo (fast-forward physics + learning)
    P   - pause / resume
    S   - save the learned brain to disk
    L   - load a previously saved brain from disk
    F   - forget (reset the plastic synapses to naive random)
    ESC - quit
"""
import os
import sys
from collections import deque

import numpy as np
import pygame
import torch

from bird_game import (
    BIRD_RADIUS, BIRD_X, GROUND_HEIGHT, GROUND_Y, HEIGHT, LEFT_WIDTH,
    PIPE_GAP, PIPE_WIDTH, Game,
)
from fly_brain import FLAP_VALENCE_THRESHOLD, GF_LOOM_THRESHOLD, FlyBrain
from visualization import PANE_WIDTH, BrainOverlay

WINDOW_WIDTH = LEFT_WIDTH + PANE_WIDTH
WINDOW_HEIGHT = HEIGHT
FPS = 60
TURBO_MULT = 12
EPISODE_HISTORY_LEN = 200
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fly_brain_save.pt")

SKY_COLOR = (110, 190, 220)
GROUND_COLOR = (222, 197, 120)
GROUND_LINE_COLOR = (160, 130, 70)
PIPE_COLOR = (60, 170, 80)
PIPE_EDGE_COLOR = (30, 110, 50)
BIRD_COLOR = (250, 210, 40)
BIRD_OUTLINE = (120, 90, 10)
GF_RING_COLOR = (255, 70, 70)
FLAP_RING_COLOR = (255, 255, 255)
HUD_BAR_COLOR = (12, 14, 22)
HUD_TEXT = (235, 235, 240)
HUD_BAR_HEIGHT = 26


class Simulation:
    """Owns the single fly's brain and game world, and advances both by one
    synchronized logic tick at a time. The brain persists across episodes;
    only the game world resets when the fly dies."""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.brain = FlyBrain(seed=1337)
        self.game = Game(rng=self.rng)
        self.episode_scores = deque(maxlen=EPISODE_HISTORY_LEN)
        self.episode_count = 0
        self.best_score = 0
        _, self.last_raw = self.game.get_features()
        self.last_flap = False
        self.last_gf = False

    def step_logic(self):
        features, raw = self.game.get_features()
        features_t = torch.tensor(features, dtype=torch.float32)

        kc, valence = self.brain.forward(features_t)
        gf_triggered = raw["loom"] > GF_LOOM_THRESHOLD
        flap = raw["reflex_action"] if gf_triggered else (valence > FLAP_VALENCE_THRESHOLD)
        self.brain.commit(kc, flap)

        self.game.apply_flap(flap)
        self.brain.continuous_update(raw["shaping_reward"], kc, flap)
        result = self.game.physics_step()

        if result.passed_pipe:
            self.brain.reward()
        if result.died:
            self.brain.punish()
            score = self.game.pipes_passed_this_episode
            self.episode_scores.append(score)
            self.episode_count += 1
            self.best_score = max(self.best_score, score)
            self.game.reset_episode()
            self.brain.reset_traces()

        self.last_raw = raw
        self.last_flap = flap
        self.last_gf = gf_triggered

    def save(self):
        torch.save({
            "brain": self.brain.state_dict(),
            "episode_count": self.episode_count,
            "episode_scores": list(self.episode_scores),
            "best_score": self.best_score,
        }, SAVE_PATH)
        return SAVE_PATH

    def load(self):
        if not os.path.exists(SAVE_PATH):
            return False
        data = torch.load(SAVE_PATH, weights_only=False)
        self.brain.load_state_dict(data["brain"])
        self.episode_count = data.get("episode_count", 0)
        self.episode_scores = deque(data.get("episode_scores", []), maxlen=EPISODE_HISTORY_LEN)
        self.best_score = data.get("best_score", 0)
        return True


def draw_game_pane(surface, sim, big_font, small_font):
    pygame.draw.rect(surface, SKY_COLOR, (0, 0, LEFT_WIDTH, HEIGHT))

    for pipe in sim.game.pipes:
        top_rect = pygame.Rect(int(pipe.x), 0, PIPE_WIDTH, max(0, int(pipe.gap_y)))
        bottom_y = int(pipe.gap_y + PIPE_GAP)
        bottom_rect = pygame.Rect(int(pipe.x), bottom_y, PIPE_WIDTH, max(0, HEIGHT - bottom_y))
        pygame.draw.rect(surface, PIPE_COLOR, top_rect)
        pygame.draw.rect(surface, PIPE_EDGE_COLOR, top_rect, 3)
        pygame.draw.rect(surface, PIPE_COLOR, bottom_rect)
        pygame.draw.rect(surface, PIPE_EDGE_COLOR, bottom_rect, 3)

    pygame.draw.rect(surface, GROUND_COLOR, (0, GROUND_Y, LEFT_WIDTH, GROUND_HEIGHT))
    pygame.draw.line(surface, GROUND_LINE_COLOR, (0, GROUND_Y), (LEFT_WIDTH, GROUND_Y), 3)

    score_text = big_font.render(str(sim.game.pipes_passed_this_episode), True, (255, 255, 255))
    surface.blit(score_text, (LEFT_WIDTH // 2 - score_text.get_width() // 2, HUD_BAR_HEIGHT + 20))
    ep_text = small_font.render(f"episode {sim.episode_count}", True, (255, 255, 255))
    surface.blit(ep_text, (LEFT_WIDTH // 2 - ep_text.get_width() // 2, HUD_BAR_HEIGHT + 66))

    x, y = BIRD_X, int(sim.game.y)
    if sim.last_gf:
        pygame.draw.circle(surface, GF_RING_COLOR, (x, y), BIRD_RADIUS + 6, 3)
    elif sim.last_flap:
        pygame.draw.circle(surface, FLAP_RING_COLOR, (x, y), BIRD_RADIUS + 4, 2)
    pygame.draw.circle(surface, BIRD_COLOR, (x, y), BIRD_RADIUS)
    pygame.draw.circle(surface, BIRD_OUTLINE, (x, y), BIRD_RADIUS, 2)


def draw_hud(surface, sim, font, turbo, paused, fps):
    pygame.draw.rect(surface, HUD_BAR_COLOR, (0, 0, WINDOW_WIDTH, HUD_BAR_HEIGHT))

    help_text = font.render("[T]urbo [P]ause [S]ave [L]oad [F]orget [ESC]", True, HUD_TEXT)
    help_x = WINDOW_WIDTH - help_text.get_width() - 8
    surface.blit(help_text, (help_x, 5))

    lines = [
        f"Episode {sim.episode_count}",
        f"Score {sim.game.pipes_passed_this_episode}",
        f"Best {sim.best_score}",
    ]
    if paused:
        lines.append("PAUSED")
    elif turbo:
        lines.append(f"TURBO x{TURBO_MULT}")
    lines.append(f"{fps:.0f} fps")

    x = 8
    max_x = help_x - 12
    for line in lines:
        text = font.render(line, True, HUD_TEXT)
        if x + text.get_width() > max_x:
            break
        surface.blit(text, (x, 5))
        x += text.get_width() + 14


def main():
    pygame.init()
    pygame.display.set_caption("Fly-Brain Flappy Bird - Dopamine-Learned Mushroom Body")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 13, bold=True)
    big_font = pygame.font.SysFont("consolas", 40, bold=True)
    small_font = pygame.font.SysFont("consolas", 14)

    sim = Simulation()
    overlay = BrainOverlay()

    turbo = False
    paused = False
    status_message = ""
    status_timer = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_t:
                    turbo = not turbo
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_s:
                    path = sim.save()
                    status_message = f"saved brain -> {os.path.basename(path)}"
                    status_timer = 120
                elif event.key == pygame.K_l:
                    status_message = "loaded saved brain" if sim.load() else "no saved brain found"
                    status_timer = 120
                elif event.key == pygame.K_f:
                    sim.brain.forget()
                    status_message = "forgot learned synapses"
                    status_timer = 120

        if status_timer > 0:
            status_timer -= 1
            if status_timer == 0:
                status_message = ""

        if not paused:
            for _ in range(TURBO_MULT if turbo else 1):
                sim.step_logic()

        screen.fill((0, 0, 0))
        draw_game_pane(screen, sim, big_font, small_font)

        gf_triggered = sim.last_gf
        overlay.draw(
            screen, LEFT_WIDTH, sim.brain, sim.last_raw, sim.last_flap, gf_triggered,
            episode_count=sim.episode_count, episode_scores=sim.episode_scores,
            best_score=sim.best_score, turbo_mult=(TURBO_MULT if turbo else 1), paused=paused,
        )

        draw_hud(screen, sim, font, turbo, paused, clock.get_fps())
        if status_message:
            msg = font.render(status_message, True, (255, 255, 255))
            bg = pygame.Surface((msg.get_width() + 16, msg.get_height() + 8), pygame.SRCALPHA)
            bg.fill((20, 20, 30, 210))
            bg.blit(msg, (8, 4))
            screen.blit(bg, (WINDOW_WIDTH // 2 - bg.get_width() // 2, HUD_BAR_HEIGHT + 6))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
