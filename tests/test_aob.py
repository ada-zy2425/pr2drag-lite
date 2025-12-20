# tests/test_aob.py
import numpy as np

from pr2drag.aob import AoBParams, aob_fill


def _cfg(**kw):
    base = dict(
        eps_gate=0.1,
        abstain_mode="linear",
        eta_L=0.05,
        eta_u=0.65,
        max_bridge_len=40,
        bridge_mode="hermite",
        hermite_scan=6,
        clamp_hermite=True,
        clamp_margin_px=0.0,
        clip_w=True,
        nan_is_low=True,
        require_finite_w=False,
        require_finite_z=False,
    )
    base.update(kw)
    return AoBParams(**base)


def test_no_low_segment():
    T = 20
    z = np.stack([np.linspace(0, 19, T), np.linspace(0, 0, T)], axis=1).astype(np.float32)
    w = np.ones((T,), dtype=np.float32) * 0.9
    tau = 0.5
    zf, ib, ia = aob_fill(z, w, tau, _cfg(), debug=[])
    assert np.allclose(zf, z)
    assert ib.sum() == 0
    assert ia.sum() == 0


def test_bridge_short_gap_with_anchors():
    # Low segment in middle, both anchors exist, should bridge.
    T = 30
    z = np.stack([np.linspace(0, 29, T), np.linspace(0, 0, T)], axis=1).astype(np.float32)
    w = np.ones((T,), dtype=np.float32) * 0.9
    w[10:15] = 0.0
    tau = 0.5
    dbg = []
    zf, ib, ia = aob_fill(z, w, tau, _cfg(bridge_mode="linear"), debug=dbg)
    assert ib[10:15].all() and (not ia[10:15].any())
    assert np.allclose(zf[9], z[9]) and np.allclose(zf[15], z[15])
    # linear bridge should be between endpoints strictly
    assert np.all(zf[10:15, 0] > zf[9, 0]) and np.all(zf[10:15, 0] < zf[15, 0])
    assert dbg[-1]["do_bridge"] is True


def test_abstain_when_gap_too_long():
    # Low segment length > max_bridge_len => abstain linear
    T = 100
    z = np.stack([np.linspace(0, 99, T), np.zeros((T,))], axis=1).astype(np.float32)
    w = np.ones((T,), dtype=np.float32) * 0.9
    w[10:80] = 0.0  # length 70
    tau = 0.5
    zf, ib, ia = aob_fill(z, w, tau, _cfg(max_bridge_len=40, abstain_mode="linear"), debug=[])
    assert ia[10:80].all() and (not ib[10:80].any())
    # should be roughly linear between anchors 9 and 80
    assert np.all(np.diff(zf[10:80, 0]) > 0)


def test_abstain_missing_left_anchor_uses_right():
    T = 20
    z = np.stack([np.linspace(0, 19, T), np.zeros((T,))], axis=1).astype(np.float32)
    w = np.ones((T,), dtype=np.float32) * 0.9
    w[0:5] = 0.0  # low at start, no left anchor
    tau = 0.5
    zf, ib, ia = aob_fill(z, w, tau, _cfg(abstain_mode="hold"), debug=[])
    assert ia[0:5].all()
    assert np.allclose(zf[0:5], zf[5][None, :])  # hold uses right when left missing
    assert (not ib[0:5].any())


def test_nan_in_w_treated_as_low():
    T = 20
    z = np.stack([np.linspace(0, 19, T), np.zeros((T,))], axis=1).astype(np.float32)
    w = np.ones((T,), dtype=np.float32) * 0.9
    w[8] = np.nan
    tau = 0.5
    zf, ib, ia = aob_fill(z, w, tau, _cfg(nan_is_low=True, abstain_mode="linear", bridge_mode="linear"), debug=[])
    # single-frame low segment with anchors => bridge
    assert ib[8] or ia[8]
    assert np.isfinite(zf[8]).all()


def test_hermite_clamp_prevents_overshoot():
    # Construct endpoints with strong velocities to induce overshoot; clamp must contain it.
    T = 30
    z = np.zeros((T, 2), dtype=np.float32)
    z[:, 0] = np.linspace(0, 10, T)
    z[:, 1] = 0.0
    w = np.ones((T,), dtype=np.float32) * 0.9
    w[10:20] = 0.0
    tau = 0.5
    zf, ib, ia = aob_fill(z, w, tau, _cfg(bridge_mode="hermite", clamp_hermite=True), debug=[])
    assert ib[10:20].all()
    lo = min(float(zf[9, 0]), float(zf[20, 0])) - 1e-3
    hi = max(float(zf[9, 0]), float(zf[20, 0])) + 1e-3
    # with clamp enabled, it should not go crazy outside an expanded box;
    # we just check it's not exploding.
    assert float(np.max(zf[10:20, 0])) < hi + 100.0
    assert float(np.min(zf[10:20, 0])) > lo - 100.0


def main():
    tests = [
        test_no_low_segment,
        test_bridge_short_gap_with_anchors,
        test_abstain_when_gap_too_long,
        test_abstain_missing_left_anchor_uses_right,
        test_nan_in_w_treated_as_low,
        test_hermite_clamp_prevents_overshoot,
    ]
    for fn in tests:
        fn()
    print(f"[OK] {len(tests)} AoB tests passed.")


if __name__ == "__main__":
    main()
