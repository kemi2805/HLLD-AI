import pstats, sys, io
st = pstats.Stats(sys.argv[1]); s = io.StringIO(); st.stream = s
st.sort_stats("cumulative").print_stats(120)
want = ("planar4_b.py:113(slow_shock_p)", "slow_shock_b.py:598(retry)", "wave_speeds_b.py:299(xi_all)", "fan_njit.py:518(_run)", "planar4_b.py:302(structure)", "planar5_b.py:179(structure)", "_ufunc_config.py:33(seterr)", "shock_b.py:417(bisect)", "slow_shock_njit.py:304(solve_lanes)", "shock_njit.py:273(solve_lanes)", "api.py", "fullcontact_b.py:296(fast_wave)")
for l in s.getvalue().splitlines():
    if any(w in l for w in want):
        p = l.split()
        print("   %-50s calls %8s  cum %6.2f s" % (p[-1].split("/")[-1], p[0], float(p[3])))
