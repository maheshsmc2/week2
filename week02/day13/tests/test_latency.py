import time
def test_latency_guard():
    t0 = time.time()
    time.sleep(0.01)  # simulate work
    assert (time.time()-t0) * 1000 < 200  # 200ms guard for placeholder
