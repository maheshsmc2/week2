def test_rrf_merge_runs():
    from utils.retrieval import rrf_merge
    run = [[('A',1.0),('B',0.5)], [('B',0.9),('C',0.4)]]
    out = rrf_merge(run, k=2)
    assert len(out) == 2
