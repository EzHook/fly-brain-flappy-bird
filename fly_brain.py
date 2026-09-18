"""
A simplified, presentation-friendly model of the fly mushroom-body learning
circuit -- inspired by, not a literal reproduction of, real Drosophila
neuroscience:

    sensory (vision + scent)
        --frozen random projection-->  Kenyon cells (KC, sparse code)
        --PLASTIC synapses-->          MBONs (approach / avoid)
        -->                            valence  -->  flap / don't

The network is intentionally tiny (6 sensory channels, 200 KCs, 2 MBONs) so
it can visibly learn on-screen in a handful of episodes, rather than needing
millions of training steps. The one thing kept faithful to the biology is
the LEARNING RULE: KC->MBON synapses sit frozen at small random values until
a dopamine signal arrives, then they are nudged based on which Kenyon cells
were recently active and which action (flap / no-flap) the fly took --
coincident KC activity plus dopamine is believed to be how real mushroom-body
synapses change (Hige et al. 2015; Aso et al. 2014). PAM dopaminergic
neurons carry reward here (a pipe cleared, or the gap's "smell" getting
stronger), PPL1 carries punishment (a collision). The PN->KC projection
itself is never touched -- only the KC->MBON readout learns, mirroring the
fixed-wiring / plastic-readout structure real mushroom bodies are believed
to use.
"""
import torch

NUM_PN = 6          # projection neurons: the raw sensory feature channels
NUM_KC = 200         # Kenyon cells (sparse expansion layer)
KC_ACTIVE = 16       # ~8% sparse code, matching real Kenyon-cell sparseness
NUM_MBON = 2         # display-only split of the valence signal: approach / avoid

ELIGIBILITY_DECAY = 0.985      # ~a whole (short) episode's worth of credit-assignment memory
CONTINUOUS_LR = 0.05           # gentle, every-frame scent-following shaping
EVENT_LR = 0.6                 # big, sparse pipe-pass / collision updates
WEIGHT_CLIP = 4.0
# Slow forgetting so old, stale associations eventually fade -- but slow
# enough that it doesn't outrun sparse per-episode reward/punish events.
# 0.9997 (an early value) has a ~2300-tick half-life, under 30 episodes at
# this game's pace, which erased learning faster than it could accumulate.
WEIGHT_DECAY_PER_TICK = 0.999995

GF_LOOM_THRESHOLD = 0.3         # giant-fiber escape reflex trigger (bypasses learning)
FLAP_VALENCE_THRESHOLD = 0.0


class FlyBrain:
    """One fly's sensory-to-motor pipeline. PN->KC wiring is frozen at
    construction; KC->MBON weights are the only thing that learns, via
    dopamine-gated plasticity (see `reward` / `punish` / `continuous_update`)."""

    def __init__(self, seed=1337):
        g = torch.Generator().manual_seed(seed)

        # Each KC samples a handful of sensory channels at random -- real
        # PN->KC wiring in Drosophila is largely unstructured. FROZEN forever.
        proj = torch.zeros(NUM_KC, NUM_PN)
        for k in range(NUM_KC):
            taps = torch.randperm(NUM_PN, generator=g)[:3]
            proj[k, taps] = torch.randn(3, generator=g)
        self.pn_to_kc = proj

        # PLASTIC readout: a single signed valence weight per KC (positive ->
        # "approach"/flap, negative -> "avoid"/don't). Using one vector rather
        # than two separately-reinforced approach/avoid columns matters: an
        # earlier version used two columns that both only ever grew (reward
        # added to whichever was used, so did punishment, just cross-wired),
        # and after ~2500 episodes both columns converged to nearly equal
        # magnitudes, collapsing valence (their difference) to ~0 everywhere
        # -- the fly stopped being able to tell any two situations apart. A
        # single vector that reward pushes one way and punishment pushes the
        # other can't degenerate like that.
        self.kc_to_valence = torch.randn(NUM_KC, generator=g) * 0.15

        self.eligibility_flap = torch.zeros(NUM_KC)
        self.eligibility_noflap = torch.zeros(NUM_KC)

        self.reward_events = 0
        self.punish_events = 0
        self.pam_rate = 0.0     # display-only decaying "Hz" trace
        self.ppl1_rate = 0.0
        self.last_kc_active = 0
        self.last_valence = 0.0
        self.last_approach = 0.0
        self.last_avoid = 0.0
        self.last_change_note = "warming up..."

    # ---- forward pass -------------------------------------------------

    def encode(self, features):
        """features: (NUM_PN,) tensor -> sparse (NUM_KC,) binary KC code."""
        raw = torch.relu(self.pn_to_kc @ features)
        nonzero = int(torch.count_nonzero(raw).item())
        if nonzero == 0:
            return torch.zeros(NUM_KC)
        k = min(KC_ACTIVE, nonzero)
        _, top_idx = torch.topk(raw, k)
        kc = torch.zeros(NUM_KC)
        kc[top_idx] = 1.0
        return kc

    def decode(self, kc):
        """kc: (NUM_KC,) -> (approach, avoid, valence). approach/avoid are a
        display-only split of the single valence readout (their difference
        always exactly equals valence); only valence itself drives behavior."""
        valence = float(kc @ self.kc_to_valence)
        approach = float(kc @ torch.relu(self.kc_to_valence))
        avoid = float(kc @ torch.relu(-self.kc_to_valence))
        return approach, avoid, valence

    def probe_valence(self, features):
        """Pure forward pass for the 'learned policy' grid -- no side effects."""
        kc = self.encode(features)
        _, _, valence = self.decode(kc)
        return valence

    def forward(self, features):
        """One simulated tick's sense-and-decide step (no learning yet --
        call `commit` once the resulting action is known)."""
        self.pam_rate *= 0.9
        self.ppl1_rate *= 0.9

        kc = self.encode(features)
        approach, avoid, valence = self.decode(kc)
        self.last_kc_active = int(kc.sum().item())
        self.last_valence = valence
        self.last_approach = approach
        self.last_avoid = avoid
        return kc, valence

    def commit(self, kc, flap_taken):
        """Record which action was taken for this tick's KC pattern, and let
        old synaptic changes fade slightly (a mild forgetting/homeostasis term)."""
        if flap_taken:
            self.eligibility_flap = self.eligibility_flap * ELIGIBILITY_DECAY + kc * (1 - ELIGIBILITY_DECAY)
            self.eligibility_noflap = self.eligibility_noflap * ELIGIBILITY_DECAY
        else:
            self.eligibility_noflap = self.eligibility_noflap * ELIGIBILITY_DECAY + kc * (1 - ELIGIBILITY_DECAY)
            self.eligibility_flap = self.eligibility_flap * ELIGIBILITY_DECAY
        self.kc_to_valence *= WEIGHT_DECAY_PER_TICK

    # ---- dopamine-gated plasticity --------------------------------------
    # `signed_eligibility` is positive where flapping was the recent action,
    # negative where not-flapping was. Reward pushes valence further in the
    # direction of whatever was just done (reinforcing it); punishment pushes
    # valence the OPPOSITE way (favoring the other action next time this
    # situation recurs). Because both only ever act on the same single
    # vector, in opposite signed directions, this can't collapse toward zero
    # the way two independently-growing approach/avoid columns did.

    def _signed_eligibility(self):
        return self.eligibility_flap - self.eligibility_noflap

    def continuous_update(self, shaping_reward, kc, flap_taken):
        """Every-frame gentle shaping: getting more vertically aligned with
        the gap reinforces the action just taken; drifting out of alignment
        discourages it. (Not the same as the raw scent input -- see
        Game.get_features's `shaping_reward` for why that distinction matters.)

        Uses the INSTANTANEOUS KC pattern for THIS tick's action, not the
        slower eligibility trace `reward`/`punish` use -- fine-grained,
        moment-to-moment steering corrections need immediate credit, not one
        blurred across the last ~46 ticks."""
        magnitude = min(abs(shaping_reward) / 4.0, 1.0)
        if magnitude < 1e-4:
            return
        lr = CONTINUOUS_LR * magnitude
        signed_instant = kc if flap_taken else -kc
        if shaping_reward > 0:
            self.pam_rate = max(self.pam_rate, magnitude * 20.0)
            self._apply_delta(lr * signed_instant)
        else:
            self.ppl1_rate = max(self.ppl1_rate, magnitude * 20.0)
            self._apply_delta(-lr * signed_instant)

    def reward(self):
        """PAM dopamine pulse: a pipe was cleared."""
        self.reward_events += 1
        self.pam_rate = 45.0
        signed = self._signed_eligibility()
        n = int((signed.abs() > 0.05).sum().item())
        self._apply_delta(EVENT_LR * signed)
        self.last_change_note = f"PAM reward -> {n} synapses strengthened"

    def punish(self):
        """PPL1 dopamine pulse: a collision just happened."""
        self.punish_events += 1
        self.ppl1_rate = 45.0
        signed = self._signed_eligibility()
        n = int((signed.abs() > 0.05).sum().item())
        self._apply_delta(-EVENT_LR * signed)
        self.last_change_note = f"PPL1 punishment -> {n} synapses changed"

    def _apply_delta(self, delta):
        self.kc_to_valence += delta
        self.kc_to_valence.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

    def mean_weight_multiplier(self):
        return float(self.kc_to_valence.abs().mean().item())

    # ---- persistence -----------------------------------------------------

    def state_dict(self):
        return {
            "kc_to_valence": self.kc_to_valence.clone(),
            "reward_events": self.reward_events,
            "punish_events": self.punish_events,
        }

    def load_state_dict(self, state):
        self.kc_to_valence = state["kc_to_valence"].clone()
        self.reward_events = state.get("reward_events", 0)
        self.punish_events = state.get("punish_events", 0)

    def reset_traces(self):
        """Clear the eligibility traces at the start of a fresh episode --
        with a decay slow enough to span a whole (short) episode, letting it
        bleed into the next one would blame the new episode's death on the
        previous episode's actions."""
        self.eligibility_flap.zero_()
        self.eligibility_noflap.zero_()

    def forget(self, seed=None):
        """Reset the plastic KC->valence synapses back to naive small random
        values (the frozen PN->KC wiring and everything else is untouched)."""
        g = torch.Generator() if seed is None else torch.Generator().manual_seed(seed)
        self.kc_to_valence = torch.randn(NUM_KC, generator=g) * 0.15
        self.eligibility_flap.zero_()
        self.eligibility_noflap.zero_()
        self.reward_events = 0
        self.punish_events = 0
        self.last_change_note = "forgotten -- starting naive again"
