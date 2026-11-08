import pstats, sys, io
st = pstats.Stats(sys.argv[1]); s = io.StringIO(); st.stream = s
st.sort_stats("tottime").print_stats(16)
for l in s.getvalue().splitlines():
    p = l.split()
    if len(p) >= 6 and p[0].replace("/", "").isdigit():
        try:
            print("   %-62s calls %8s tot %6.2f cum %6.2f" % (p[-1].split("/")[-1][:62], p[0], float(p[1]), float(p[3])))
        except ValueError:
            pass
s = io.StringIO(); st.stream = s
st.sort_stats("cumulative").print_stats(70)
want = ("fast_edge_scan", "slow_shock_b.py:598(retry)", "xi_all", "fan_njit.py:518(_run)", "planar5_b.py:179(structure)", "api.py:144(solve_batch)", "fullcontact_b.py:296(fast_wave)", "_ufunc_config.py:33(seterr)", "planar4_b.py:302(structure)", "slow_shock_b.py:657(solve)", "fullcontact_b.py:552(fullfuncv6)")
print("   --- cumulative:")
for l in s.getvalue().splitlines():
    if any(w in l for w in want):
        p = l.split()
        try:
            print("   %-62s calls %8s cum %6.2f" % (p[-1].split("/")[-1][:62], p[0], float(p[3])))
        except ValueError:
            pass
