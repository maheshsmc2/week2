import csv

def run_grid():
    rows = []
    for size in [200, 300, 400]:
        for overlap in [30, 60, 100]:
            # TODO: run retrieval with (size, overlap)
            rows.append([size, overlap, 0.00, 0.00, "TODO"])
    with open("day11/grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chunk_size","overlap","ndcg10","recall10","notes"])
        w.writerows(rows)
    print("Wrote day11/grid.csv")

if __name__ == "__main__":
    run_grid()
