# Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""按本机数据重建 HIST 所需的概念图文件。

官方 qlib_csi300_stock2concept.npy / qlib_csi300_stock_index.npy 按官方
2008~2020 宇宙构建（735 股 × 4635 概念），与本机自建宇宙不匹配会导致
`IndexError: arrays used as indices must be of integer (or boolean) type`。
本脚本从本机日线库的概念成分文件（concept_tzt_*.txt）重建同构的两个 npy。

用法：python build_local_concept_npy.py
"""
from pathlib import Path

import numpy as np

INSTR_DIR = Path("~/.qlib/qlib_data/cn_data/instruments").expanduser()
UNIVERSE_FILE = INSTR_DIR / "csi300.txt"
OUT_DIR = Path(__file__).absolute().resolve().parent


def main():
    universe = [line.split("\t")[0] for line in UNIVERSE_FILE.read_text().splitlines() if line.strip()]
    stock_index = {code: i for i, code in enumerate(universe)}

    concept_files = sorted(INSTR_DIR.glob("concept_tzt_*.txt"))
    stock2concept = np.zeros((len(universe), len(concept_files)), dtype=np.float64)
    for j, cf in enumerate(concept_files):
        for line in cf.read_text().splitlines():
            parts = line.split("\t")
            if not parts or not parts[0]:
                continue
            i = stock_index.get(parts[0])
            if i is not None:
                stock2concept[i, j] = 1.0

    np.save(OUT_DIR / "qlib_local_stock2concept.npy", stock2concept)
    np.save(OUT_DIR / "qlib_local_stock_index.npy", np.array(stock_index, dtype=object), allow_pickle=True)
    per_stock = stock2concept.sum(1)
    print(
        f"宇宙 {len(universe)} 只 × 概念 {len(concept_files)} 个；"
        f"每股概念数 均值 {per_stock.mean():.1f} / 最大 {int(per_stock.max())} / 零概念股票 {(per_stock == 0).sum()} 只"
    )


if __name__ == "__main__":
    main()
