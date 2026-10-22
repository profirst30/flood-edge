import numpy as np

from src import simulate as sim


def ramp_flood(n_hours: int = 48, start_h: int = 20, rise_h: float = 2.0) -> np.ndarray:
    """Flat at 1 m, then a straight rise to 3 m over `rise_h` hours, then flat: on the 5-minute grid."""
    t = np.arange(n_hours * sim.PER_HOUR) / sim.PER_HOUR
    return np.clip(1 + 2 * (t - start_h) / rise_h, 1, 3).astype(np.float32)


def test_grid_interpolates_and_marks_long_gaps():
    t = np.array([0, 3600, 3600 + 13 * 3600])
    t0, x = sim.to_grid(t, np.array([0.0, 1.0, 1.0], np.float32))
    assert t0 == 0 and x[6] == np.float32(0.5)       # halfway between two source points
    assert np.isnan(x[sim.PER_HOUR + 1])                # inside a 13 h gap: missing, not invented


def test_measured_counts():
    flags = np.array([False, True, False])
    assert list(sim.measured(flags)) == [0, 12, 15, 18, 21, 24]


def test_events_merge_and_need_history():
    x = np.zeros(10 * 24 * sim.PER_HOUR, np.float32)
    x[100 * sim.PER_HOUR] = x[110 * sim.PER_HOUR] = 2   # 10 h apart: one flood
    x[200 * sim.PER_HOUR] = 2                            # 90 h later: a second one
    assert [s for s, _ in sim.events(x, 1.0)] == [100 * sim.PER_HOUR, 200 * sim.PER_HOUR]


def test_delay_hourly_versus_quarter_hourly():
    x = ramp_flood()
    thr = 2.4  # reached at 21 h 24, first seen on the grid at 21 h 25
    floods = sim.events(x, thr)
    hourly = sim.score(x, thr, floods, sim.fixed(sim.hourly(x), False))["floods"][0]
    quarter = sim.score(x, thr, floods, sim.fixed(sim.hourly(x), True))["floods"][0]
    assert hourly["delay_min"] == 35 and quarter["delay_min"] == 5  # measured at 22 h 00 versus 21 h 30


def test_rise_rule_switches_before_the_threshold():
    x = ramp_flood(rise_h=4.0)
    h1 = sim.hourly(x)
    flags = sim.rise(h1, 1.0, 2.9, 0.2)
    assert flags[21] and not flags[19] and not sim.level(h1, 1.0, 2.9, 1.0)[21]


def test_target_looks_three_hours_ahead():
    x = ramp_flood()
    y = sim.target(x, 2.4, len(sim.hourly(x)))
    assert y[19] == 1 and y[18] == 0  # from 19 h, 21 h 25 is less than 3 h ahead; from 18 h it is not
