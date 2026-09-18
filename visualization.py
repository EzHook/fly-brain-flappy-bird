"""
Live view of the fly's brain: a stylized anatomical rendering (optic lobes +
central mushroom-body blob, with small "spike" particles standing in for
neural activity) plus a stack of panels describing what's actually
happening inside the model -- a learning curve across episodes, the current
sensory-to-decision readout, a live "learned policy" heatmap, named neuron
populations with live activity bars, and the plastic synapse statistics.

This is a stylized, presentation-friendly rendering (particle bursts, not a
literal spike raster of every simulated unit) built entirely from pygame
primitives -- no extra rendering dependencies.
"""
import math
import random

import torch
import pygame

from fly_brain import NUM_KC
from bird_game import HEIGHT as GAME_HEIGHT, SCENT_VERTICAL_SIGMA

PANE_WIDTH = 500
PANE_HEIGHT = 600

BG_COLOR = (8, 10, 18)
PANEL_BG = (10, 12, 22, 235)
PANEL_BORDER = (60, 65, 90)
TEXT_COLOR = (225, 228, 238)
DIM_TEXT_COLOR = (150, 155, 175)
TITLE_COLOR = (245, 247, 250)

OPTIC_LOBE_COLOR = (60, 190, 170)
MUSHROOM_BODY_COLOR = (185, 150, 90)
SPIKE_COLOR = (245, 240, 220)
REWARD_COLOR = (120, 220, 140)
PUNISH_COLOR = (230, 90, 100)
GF_FLASH_COLOR = (255, 70, 70)
APPROACH_COLOR = (250, 165, 60)
AVOID_COLOR = (70, 140, 230)

GRID_COLS = 8
GRID_ROWS = 3
GRID_VEL_BINS = (-6.0, 0.0, 6.0)  # rising, mid, falling (raw px/frame)
GRID_ROW_LABELS = ("rising", "mid", "falling")


def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[k] + (c2[k] - c1[k]) * t) for k in range(3))


def _squash(value, scale=5.0, k=10.0):
    """Smoothly saturating display transform: behaves ~linearly for small
    values and asymptotes to +-scale for large ones. The raw valence is an
    unbounded sum over up to KC_ACTIVE synapses (each clipped to +-4), so it
    can read like '-62.03' -- accurate, but unreadable as a presentation
    number. This keeps the SIGN and rough relative size while keeping the
    printed number in a sane range; it's display-only and never feeds back
    into any decision, which always uses the raw signed valence."""
    return scale * math.tanh(value / k)


class BrainOverlay:
    """Renders the fly's brain state + learning stats into the right pane."""

    def __init__(self, width=PANE_WIDTH, height=PANE_HEIGHT):
        self.width = width
        self.height = height

        self.header_h = 54
        self.brain_h = 160
        self.learning_h = 78
        self.decision_h = 56
        self.policy_h = 92
        self.pop_h = 110
        self.synapse_h = 40

        self.brain_surf = pygame.Surface((width, self.brain_h), pygame.SRCALPHA)
        self.left_lobe = (width * 0.30, self.brain_h * 0.5, 72)
        self.right_lobe = (width * 0.70, self.brain_h * 0.5, 72)
        self.central = (width * 0.5, self.brain_h * 0.66, 48)

        self.particles = []  # each: dict(x,y,age,max_age,color)

        self.font = pygame.font.SysFont("consolas", 14)
        self.font_small = pygame.font.SysFont("consolas", 12)
        self.font_tiny = pygame.font.SysFont("consolas", 11)
        self.title_font = pygame.font.SysFont("consolas", 17, bold=True)

    # ---- brain particle scene -------------------------------------------

    def _spawn_particles(self, kc_active, raw, gf_triggered):
        for _ in range(min(8, kc_active // 3)):
            self._spawn_in_disk(self.central, SPIKE_COLOR, max_age=18)

        vis_activity = raw["loom"] * 4 + abs(raw["delta_scent"]) * 40
        for _ in range(min(6, int(vis_activity))):
            lobe = self.left_lobe if random.random() < 0.5 else self.right_lobe
            self._spawn_in_disk(lobe, OPTIC_LOBE_COLOR, max_age=14)

        if gf_triggered:
            for lobe in (self.left_lobe, self.right_lobe, self.central):
                for _ in range(4):
                    self._spawn_in_disk(lobe, GF_FLASH_COLOR, max_age=16)

    def _spawn_in_disk(self, disk, color, max_age):
        cx, cy, r = disk
        angle = random.uniform(0, 2 * math.pi)
        radius = r * math.sqrt(random.random())
        self.particles.append({
            "x": cx + math.cos(angle) * radius,
            "y": cy + math.sin(angle) * radius,
            "age": 0, "max_age": max_age, "color": color,
        })

    def _update_and_draw_particles(self, surf):
        alive = []
        for p in self.particles:
            p["age"] += 1
            if p["age"] < p["max_age"]:
                alive.append(p)
                alpha = int(255 * (1.0 - p["age"] / p["max_age"]))
                pygame.draw.circle(surf, (*p["color"], alpha), (int(p["x"]), int(p["y"])), 2)
        self.particles = alive

    def _draw_brain_scene(self, kc_active, raw, gf_triggered):
        self.brain_surf.fill((0, 0, 0, 0))

        gf_glow = 60 if gf_triggered else 0
        for (cx, cy, r), color in ((self.left_lobe, OPTIC_LOBE_COLOR), (self.right_lobe, OPTIC_LOBE_COLOR)):
            pygame.draw.circle(self.brain_surf, (*color, 70 + gf_glow), (int(cx), int(cy)), int(r))
        cx, cy, r = self.central
        pygame.draw.circle(self.brain_surf, (*MUSHROOM_BODY_COLOR, 95 + gf_glow), (int(cx), int(cy)), int(r))
        if gf_triggered:
            pygame.draw.circle(self.brain_surf, (*GF_FLASH_COLOR, 140), (int(cx), int(cy)), int(r) + 6, 3)

        self._spawn_particles(kc_active, raw, gf_triggered)
        self._update_and_draw_particles(self.brain_surf)

    # ---- text/panel helpers ------------------------------------------------

    def _draw_panel(self, pane, rect, lines, title=None):
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        pygame.draw.rect(panel, PANEL_BORDER, panel.get_rect(), 1)
        y = 6
        if title:
            panel.blit(self.font.render(title, True, TEXT_COLOR), (10, y))
            y += 19
        for text, color in lines:
            panel.blit(self.font_small.render(text, True, color), (10, y))
            y += 16
        pane.blit(panel, rect.topleft)
        return panel, rect

    def _legend_row(self, surf, x, y, items):
        for label, color in items:
            pygame.draw.circle(surf, color, (x, y + 5), 4)
            text = self.font_tiny.render(label, True, DIM_TEXT_COLOR)
            surf.blit(text, (x + 9, y))
            x += 9 + text.get_width() + 14
        return x

    # ---- learned-policy grid ------------------------------------------------

    @staticmethod
    def _grid_features(row, col):
        dy_norm = -1.0 + 2.0 * col / (GRID_COLS - 1)
        vel = GRID_VEL_BINS[row]
        dy_px = dy_norm * (GAME_HEIGHT / 2.0)
        scent = math.exp(-(dy_px * dy_px) / (2 * SCENT_VERTICAL_SIGMA ** 2))
        return torch.tensor([dy_norm, 0.25, vel / 10.0, scent, 0.0, 0.0], dtype=torch.float32)

    def _current_cell(self, raw):
        dy_norm = max(-1.0, min(1.0, raw["dy"] / (GAME_HEIGHT / 2.0)))
        col = round((dy_norm + 1.0) / 2.0 * (GRID_COLS - 1))
        vel = raw["vel"]
        row = 0 if vel < -2.0 else (2 if vel > 2.0 else 1)
        return row, col

    def _draw_policy_panel(self, pane, rect, brain, raw):
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        pygame.draw.rect(panel, PANEL_BORDER, panel.get_rect(), 1)
        panel.blit(self.font.render("Learned policy: flap drive by situation", True, TEXT_COLOR), (10, 6))

        grid_x, grid_y = 10, 27
        cell_w = (rect.width - 20) / GRID_COLS
        cell_h = 18
        cur_row, cur_col = self._current_cell(raw)
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                valence = brain.probe_valence(self._grid_features(row, col))
                t = 0.5 + 0.5 * math.tanh(valence / 10.0)
                color = _lerp_color(AVOID_COLOR, APPROACH_COLOR, t)
                cell_rect = pygame.Rect(grid_x + col * cell_w, grid_y + row * cell_h, cell_w - 2, cell_h - 2)
                pygame.draw.rect(panel, color, cell_rect)
                if row == cur_row and col == cur_col:
                    pygame.draw.rect(panel, (255, 255, 255), cell_rect, 2)

        caption = self.font_tiny.render(
            "columns: gap high -> level -> low   rows: rising / mid / falling", True, DIM_TEXT_COLOR)
        panel.blit(caption, (10, grid_y + GRID_ROWS * cell_h + 4))
        pane.blit(panel, rect.topleft)

    # ---- main draw ---------------------------------------------------------

    def draw(self, target_surface, offset_x, brain, raw, flap_taken, gf_triggered,
             episode_count, episode_scores, best_score, turbo_mult, paused):
        pane = pygame.Surface((self.width, self.height))
        pane.fill(BG_COLOR)

        # --- header (own band, never touched by the brain scene) ---
        pane.blit(self.title_font.render("SIMULATED FLY CNS", True, TITLE_COLOR), (14, 6))
        recent = list(episode_scores)[-20:]
        avg20 = sum(recent) / len(recent) if recent else 0.0
        status = "paused" if paused else (f"turbo x{turbo_mult}" if turbo_mult > 1 else "1x")
        subtitle = f"{brain.last_kc_active}/{NUM_KC} KC active * {status} * ep {episode_count} * avg20 {avg20:.1f}"
        pane.blit(self.font_tiny.render(subtitle, True, DIM_TEXT_COLOR), (14, 28))
        self._legend_row(pane, 14, 40, [
            ("optic lobe", OPTIC_LOBE_COLOR), ("mushroom body", MUSHROOM_BODY_COLOR),
            ("reward (PAM)", REWARD_COLOR), ("punish (PPL1)", PUNISH_COLOR),
        ])

        # --- brain scene (own band, clipped to its own surface) ---
        self._draw_brain_scene(brain.last_kc_active, raw, gf_triggered)
        pane.blit(self.brain_surf, (0, self.header_h))

        # --- panels, stacked flush with zero gaps ---
        y = self.header_h + self.brain_h

        lc_rect = pygame.Rect(10, y, self.width - 20, self.learning_h)
        self._draw_learning_curve(pane, lc_rect, episode_scores, avg20, best_score)
        y += self.learning_h

        dec_rect = pygame.Rect(10, y, self.width - 20, self.decision_h)
        self._draw_decision_panel(pane, dec_rect, brain, raw, flap_taken, gf_triggered)
        y += self.decision_h

        pol_rect = pygame.Rect(10, y, self.width - 20, self.policy_h)
        self._draw_policy_panel(pane, pol_rect, brain, raw)
        y += self.policy_h

        pop_rect = pygame.Rect(10, y, self.width - 20, self.pop_h)
        self._draw_populations_panel(pane, pop_rect, brain, raw, flap_taken, gf_triggered)
        y += self.pop_h

        syn_rect = pygame.Rect(10, y, self.width - 20, self.synapse_h)
        self._draw_synapses_panel(pane, syn_rect, brain)

        target_surface.blit(pane, (offset_x, 0))

    def _draw_learning_curve(self, pane, rect, episode_scores, avg20, best_score):
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        pygame.draw.rect(panel, PANEL_BORDER, panel.get_rect(), 1)
        panel.blit(self.font.render(
            f"Learning curve - {len(episode_scores)} episodes - last 20 avg {avg20:.2f} - best {best_score}",
            True, TEXT_COLOR), (10, 6))

        chart = pygame.Rect(10, 26, rect.width - 20, rect.height - 34)
        scores = list(episode_scores)
        if len(scores) >= 2:
            max_val = max(max(scores), 1)
            n = len(scores)
            pts = [(chart.x + (i / (n - 1)) * chart.width, chart.bottom - (s / max_val) * chart.height)
                   for i, s in enumerate(scores)]
            for x, py in pts:
                pygame.draw.circle(panel, (150, 130, 210, 150), (int(x), int(py)), 2)

            window = 20
            avg_pts = []
            for i in range(n):
                lo = max(0, i - window + 1)
                seg = scores[lo:i + 1]
                avg_pts.append((pts[i][0], chart.bottom - (sum(seg) / len(seg) / max_val) * chart.height))
            pygame.draw.lines(panel, REWARD_COLOR, False, avg_pts, 2)
        pane.blit(panel, rect.topleft)

    def _draw_decision_panel(self, pane, rect, brain, raw, flap_taken, gf_triggered):
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        pygame.draw.rect(panel, PANEL_BORDER, panel.get_rect(), 1)
        panel.blit(self.font.render("Decision: MBON valence -> flap", True, TEXT_COLOR), (10, 6))
        line = (f"valence {_squash(brain.last_valence):+.2f} Hz   active KC {brain.last_kc_active}/{NUM_KC}   "
                f"cue dy {raw['dy']:+.0f}px dx {raw['dx']:.0f}px vy {raw['vel']:+.1f}")
        panel.blit(self.font_tiny.render(line, True, DIM_TEXT_COLOR), (10, 24))

        bar_rect = pygame.Rect(10, 40, rect.width - 20, 10)
        pygame.draw.rect(panel, (40, 42, 60), bar_rect)
        mid = bar_rect.x + bar_rect.width // 2
        pygame.draw.line(panel, DIM_TEXT_COLOR, (mid, bar_rect.y), (mid, bar_rect.bottom), 1)
        span = bar_rect.width / 2.0
        pos = max(-1.0, min(1.0, brain.last_valence / 2.5))
        fill_color = GF_FLASH_COLOR if gf_triggered else (APPROACH_COLOR if flap_taken else AVOID_COLOR)
        if pos >= 0:
            pygame.draw.rect(panel, fill_color, (mid, bar_rect.y, int(pos * span), bar_rect.height))
        else:
            pygame.draw.rect(panel, fill_color, (mid + int(pos * span), bar_rect.y, int(-pos * span), bar_rect.height))
        pane.blit(panel, rect.topleft)

    def _draw_populations_panel(self, pane, rect, brain, raw, flap_taken, gf_triggered):
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        pygame.draw.rect(panel, PANEL_BORDER, panel.get_rect(), 1)
        panel.blit(self.font.render("Populations (Hz, this tick)", True, TEXT_COLOR), (10, 6))

        rows = [
            ("Kenyon cells", brain.last_kc_active * 2.0, (200, 200, 210)),
            ("MBONs", 100.0 * math.tanh((abs(brain.last_approach) + abs(brain.last_avoid)) / 20.0), (200, 200, 210)),
            ("PAM (reward DA)", brain.pam_rate, REWARD_COLOR),
            ("PPL1 (punish DA)", brain.ppl1_rate, PUNISH_COLOR),
            ("L2 lamina (vision)", min(100.0, abs(raw["dy"]) * 0.15 + abs(raw["dx"]) * 0.1), OPTIC_LOBE_COLOR),
            ("LC4/LPLC2 (loom)", raw["loom"] * 90.0, OPTIC_LOBE_COLOR),
            ("Giant fiber", 90.0 if gf_triggered else 0.0, GF_FLASH_COLOR),
            ("Wing steering MNs", 65.0 if flap_taken else 6.0, APPROACH_COLOR),
        ]
        y = 24
        bar_x, bar_w = 140, rect.width - 210
        for label, value, color in rows:
            panel.blit(self.font_tiny.render(label, True, DIM_TEXT_COLOR), (10, y))
            pygame.draw.rect(panel, (40, 42, 60), (bar_x, y + 2, bar_w, 7))
            frac = max(0.0, min(1.0, value / 100.0))
            pygame.draw.rect(panel, color, (bar_x, y + 2, int(bar_w * frac), 7))
            val_text = self.font_tiny.render(f"{value:.0f}", True, DIM_TEXT_COLOR)
            panel.blit(val_text, (bar_x + bar_w + 8, y))
            y += 12
        pane.blit(panel, rect.topleft)

    def _draw_synapses_panel(self, pane, rect, brain):
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        pygame.draw.rect(panel, PANEL_BORDER, panel.get_rect(), 1)

        line = (f"KC->MBON synapses (plastic): |w| {brain.mean_weight_multiplier():.2f}  "
                f"rewards {brain.reward_events}  punish {brain.punish_events}")
        max_w = rect.width - 20
        text = self.font_small.render(line, True, TEXT_COLOR)
        if text.get_width() > max_w:  # very long runs can rack up big episode counts
            text = self.font_tiny.render(line, True, TEXT_COLOR)
        panel.blit(text, (10, 6))

        panel.blit(self.font_tiny.render(f"last: {brain.last_change_note}", True, DIM_TEXT_COLOR), (10, 22))
        pane.blit(panel, rect.topleft)
