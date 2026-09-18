"""
Single-fly Flappy Bird physics. Unlike a population-based approach, one fly
lives episode after episode, forever -- when it dies it immediately
respawns, and only the pipe layout resets. What it has learned (its
KC->MBON synapses, in fly_brain.py) persists across every episode.
"""
import math
from collections import namedtuple

import numpy as np

LEFT_WIDTH = 400
HEIGHT = 600

BIRD_X = 90
BIRD_RADIUS = 12
GROUND_HEIGHT = 20
GROUND_Y = HEIGHT - GROUND_HEIGHT

# A flap SETS velocity to FLAP_VELOCITY outright (not an additive impulse),
# and only GRAVITY decelerates it afterwards, so its natural stopping
# distance is FLAP_VELOCITY^2 / (2*GRAVITY). At -8.0 that's 64px -- almost
# exactly the width of the giant-fiber reflex's safety zone, so a fly that
# flaps right at the edge of "danger" cannot actually stop in time even
# with a perfect reflex. A gentler impulse gives the reflex real margin
# instead of a coin-flip.
GRAVITY = 0.5
FLAP_VELOCITY = -6.0
MAX_FALL_SPEED = 11.0
MAX_RISE_SPEED = -10.0

PIPE_WIDTH = 66
PIPE_GAP = 220        # generous gap -- an evolved/learned fly should clear this most of the time
PIPE_SPEED = 2.2      # slower approach gives more reaction time
PIPE_SPACING = 280    # more breathing room between pipes
GAP_MARGIN = 50  # keep gap center away from ceiling/ground

SCENT_VERTICAL_SIGMA = 90.0     # how tightly the "smell" concentrates around the gap center
SCENT_HORIZONTAL_SCALE = 380.0  # how far away the smell is still noticeable

LOOM_PIPE_RANGE = 110.0   # pipe closeness (px) at which an off-target approach starts to "loom"
LOOM_MARGIN = 40.0        # how far outside the gap counts as "off target"
# A single flap sets velocity to FLAP_VELOCITY (-8) and only GRAVITY (0.5)
# decelerates it, so a bird that has *just* flapped needs FLAP_VELOCITY^2 /
# (2*GRAVITY) = 64px of clearance to stop before the reflex's "don't flap"
# recommendation can actually take effect. A smaller range (55px, an earlier
# value) let the reflex fire too late to matter -- the fly kept sailing into
# the ceiling anyway.
LOOM_EDGE_RANGE = 90.0

GameResult = namedtuple("GameResult", ["died", "passed_pipe"])


class Pipe:
    __slots__ = ("x", "gap_y", "scored")

    def __init__(self, x, gap_y):
        self.x = x
        self.gap_y = gap_y  # y-coordinate of the TOP of the gap
        self.scored = False

    @property
    def gap_center(self):
        return self.gap_y + PIPE_GAP / 2.0


class Game:
    """A single fly's Flappy Bird world, reset every time it dies."""

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()
        self.pipes_passed_this_episode = 0
        self.frame_count_this_episode = 0
        self.prev_scent = 0.0
        self.prev_dy = 0.0
        self.reset_episode()

    def reset_episode(self):
        self.y = HEIGHT / 2.0
        self.vel = 0.0
        self.pipes = [Pipe(LEFT_WIDTH + 150 + i * PIPE_SPACING, self._random_gap_y()) for i in range(3)]
        self.pipes_passed_this_episode = 0
        self.frame_count_this_episode = 0
        pipe = self._next_pipe()
        self.prev_scent = self._scent_at(pipe, self.y)
        self.prev_dy = pipe.gap_center - self.y

    def _random_gap_y(self):
        lo = GAP_MARGIN
        hi = HEIGHT - GROUND_HEIGHT - GAP_MARGIN - PIPE_GAP
        return float(self.rng.uniform(lo, hi))

    def _next_pipe(self):
        for p in self.pipes:
            if p.x + PIPE_WIDTH >= BIRD_X:
                return p
        return self.pipes[-1]

    @staticmethod
    def _scent_at(pipe, y):
        dy = pipe.gap_center - y
        dx = max(0.0, pipe.x - BIRD_X)
        vertical = math.exp(-(dy * dy) / (2 * SCENT_VERTICAL_SIGMA ** 2))
        horizontal = math.exp(-dx / SCENT_HORIZONTAL_SCALE)
        return vertical * horizontal

    @staticmethod
    def _loom_and_reflex(pipe, y):
        """Returns (loom_magnitude, recommended_flap) -- the giant-fiber
        escape reflex needs to know not just THAT something looms, but which
        emergency action it calls for: forcing a flap near the ceiling would
        fly straight into it, so the hazard type has to pick the reflex."""
        dx = pipe.x - BIRD_X
        dy = pipe.gap_center - y

        pipe_loom, pipe_action = 0.0, None
        if 0 < dx < LOOM_PIPE_RANGE:
            closeness = 1.0 - dx / LOOM_PIPE_RANGE
            off_target = max(0.0, abs(dy) - (PIPE_GAP / 2 - LOOM_MARGIN)) / LOOM_MARGIN
            pipe_loom = closeness * min(1.0, off_target)
            pipe_action = dy < 0  # gap center is above the bird -> needs to climb

        if y < GROUND_Y - y:
            edge_loom = max(0.0, 1.0 - y / LOOM_EDGE_RANGE)
            edge_action = False  # close to the ceiling -> do NOT flap
        else:
            edge_loom = max(0.0, 1.0 - (GROUND_Y - y) / LOOM_EDGE_RANGE)
            edge_action = True  # close to the ground -> flap

        if pipe_loom >= edge_loom:
            return pipe_loom, pipe_action if pipe_action is not None else edge_action
        return edge_loom, edge_action

    def get_features(self):
        """Returns (feature_tensor_values as list[float], raw info dict) for
        the current tick -- vision (dy, dx, vel) + scent + its rate of
        change + a loom/collision-imminent signal."""
        pipe = self._next_pipe()
        dy = pipe.gap_center - self.y
        dx = pipe.x - BIRD_X

        scent = self._scent_at(pipe, self.y)
        delta_scent = scent - self.prev_scent
        self.prev_scent = scent

        # The SENSED scent (above) deliberately blends vertical alignment and
        # horizontal proximity -- a real odor plume gets stronger as you
        # approach it from any direction, and that's a fine INPUT feature.
        # But using that same blend to TEACH the fly is a trap: merely
        # closing the horizontal distance to a pipe reads as "reward" even
        # while badly misaligned vertically, which taught an earlier version
        # to hug the ceiling for an entire flight and coast toward pipes
        # sideways. The actual teaching signal below only credits genuine
        # vertical-alignment improvement, weighted by how much it currently
        # matters (i.e. how close the pipe already is).
        alignment_gain = abs(self.prev_dy) - abs(dy)
        self.prev_dy = dy
        proximity_weight = math.exp(-max(0.0, dx) / SCENT_HORIZONTAL_SCALE)
        shaping_reward = alignment_gain * proximity_weight

        loom, reflex_action = self._loom_and_reflex(pipe, self.y)

        features = [
            dy / (HEIGHT / 2.0),
            dx / LEFT_WIDTH,
            self.vel / 10.0,
            scent,
            delta_scent * 10.0,
            loom,
        ]
        raw = {
            "dy": dy, "dx": dx, "vel": self.vel,
            "scent": scent, "delta_scent": delta_scent, "shaping_reward": shaping_reward,
            "loom": loom, "reflex_action": reflex_action,
        }
        return features, raw

    def apply_flap(self, flap):
        if flap:
            self.vel = FLAP_VELOCITY

    def physics_step(self):
        self.frame_count_this_episode += 1
        self.vel = np.clip(self.vel + GRAVITY, MAX_RISE_SPEED, MAX_FALL_SPEED)
        self.y += self.vel

        for p in self.pipes:
            p.x -= PIPE_SPEED
        if self.pipes[0].x + PIPE_WIDTH < 0:
            self.pipes.pop(0)
            self.pipes.append(Pipe(self.pipes[-1].x + PIPE_SPACING, self._random_gap_y()))

        passed_pipe = False
        for p in self.pipes:
            if not p.scored and p.x + PIPE_WIDTH < BIRD_X:
                p.scored = True
                self.pipes_passed_this_episode += 1
                passed_pipe = True

        died = self._check_collision()
        return GameResult(died=died, passed_pipe=passed_pipe)

    def _check_collision(self):
        if self.y - BIRD_RADIUS < 0 or self.y + BIRD_RADIUS > GROUND_Y:
            return True
        for p in self.pipes:
            overlaps_x = (BIRD_X + BIRD_RADIUS > p.x) and (BIRD_X - BIRD_RADIUS < p.x + PIPE_WIDTH)
            if not overlaps_x:
                continue
            if self.y - BIRD_RADIUS < p.gap_y or self.y + BIRD_RADIUS > p.gap_y + PIPE_GAP:
                return True
        return False
