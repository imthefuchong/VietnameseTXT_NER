"""
Baseline NER tiếng Việt v2: CRF + đặc trưng mở rộng, có ablation, tuning c1/c2 trên dev
và phân tích lỗi. Dữ liệu: VLSP 2016 (CoNLL: token [POS] [chunk] NER, câu cách nhau dòng trống).

Cài đặt: pip install sklearn-crfsuite seqeval
Chạy:    python ner_crf_baseline_v2.py train.txt dev.txt test.txt
Tuỳ chọn:
  --gaz FILE   file gazetteer, mỗi dòng một tên (ví dụ: Hà Nội, Bộ Giáo dục và Đào tạo)
  --no-tune    bỏ qua bước thử c1/c2 (chạy nhanh hơn)
  --iters N    số vòng lặp khi ablation (mặc định 100)
Quy trình: ablation (chọn trên DEV) -> tune c1/c2 (DEV) -> chạy TEST đúng một lần.
"""
import argparse
import pickle
from collections import Counter

import sklearn_crfsuite
from seqeval.metrics import classification_report, f1_score

# ---------------------------------------------------------------- từ khoá gợi ý (âm tiết đơn)
ORG_TRIG = {
    "công", "ty", "ngân", "hàng", "bộ", "sở", "trường", "đại", "học", "tập", "đoàn",
    "ubnd", "hội", "đảng", "viện", "tổng", "cục", "ban", "liên", "đài", "tòa", "án",
    "quốc", "chính", "phủ", "uỷ", "ủy", "hiệp", "câu", "lạc", "bộ", "tờ", "báo",
    "cty", "tcty", "nhà", "máy", "hãng", "ủy", "ban",
}
LOC_TRIG = {
    "thành", "phố", "tỉnh", "huyện", "quận", "xã", "phường", "tp", "sông", "núi",
    "biển", "đảo", "đường", "nước", "thị", "trấn", "làng", "ấp", "khu", "vùng", "châu",
}
PER_TRIG = {
    "ông", "bà", "anh", "chị", "ngài", "ts", "gs", "pgs", "ths", "bs", "đồng", "chí",
    "chủ", "tịch", "giám", "đốc", "trưởng", "thủ", "tướng", "nhà", "ca", "sĩ", "nghệ",
}


# ---------------------------------------------------------------- đọc dữ liệu
def read_conll(path):
    sents, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            cols = line.split()
            pos = cols[1] if len(cols) > 2 else "_"
            cur.append((cols[0], pos, cols[-1]))
    if cur:
        sents.append(cur)
    return sents


# ---------------------------------------------------------------- hàm hỗ trợ đặc trưng
def shape(w):
    """Hình dạng từ: 'Hà' -> 'Xx', 'UBND' -> 'X', '2016' -> 'd'."""
    out = []
    for ch in w:
        if ch.isdigit():
            c = "d"
        elif ch.isalpha():
            c = "X" if ch.isupper() else "x"
        else:
            c = ch
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def cap_runs(sent):
    """Với mỗi token: (độ dài chuỗi âm tiết viết hoa liên tiếp chứa nó, vị trí trong chuỗi)."""
    n = len(sent)
    is_cap = [t[0][:1].isupper() for t in sent]
    info = [(0, "none")] * n
    i = 0
    while i < n:
        if not is_cap[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and is_cap[j + 1]:
            j += 1
        length = j - i + 1
        for k in range(i, j + 1):
            if length == 1:
                pos = "single"
            elif k == i:
                pos = "start"
            elif k == j:
                pos = "end"
            else:
                pos = "mid"
            info[k] = (min(length, 6), pos)
        i = j + 1
    return info


def load_gaz(path):
    gaz, maxlen = set(), 1
    with open(path, encoding="utf-8") as f:
        for line in f:
            toks = tuple(line.strip().lower().replace("_", " ").split())
            if toks:
                gaz.add(toks)
                maxlen = max(maxlen, len(toks))
    return gaz, maxlen


def gaz_tags(sent, gaz, maxlen):
    """Khớp cụm dài nhất theo gazetteer, trả về B/I/O cho từng token."""
    n = len(sent)
    words = [t[0].lower() for t in sent]
    tags = ["O"] * n
    i = 0
    while i < n:
        m = 0
        for L in range(min(maxlen, n - i), 0, -1):
            if tuple(words[i:i + L]) in gaz:
                m = L
                break
        if m:
            tags[i] = "B"
            for k in range(i + 1, i + m):
                tags[k] = "I"
            i += m
        else:
            i += 1
    return tags


def tok_feats(sent, i, pre, full):
    w, pos, _ = sent[i]
    lw = w.lower()
    f = {pre + "lower": lw, pre + "title": w.istitle(), pre + "upper": w.isupper()}
    if full:
        f[pre + "digit"] = w.isdigit()
        f[pre + "hyphen"] = "-" in w
        f[pre + "pre2"] = lw[:2]
        f[pre + "suf2"] = lw[-2:]
        f[pre + "pos"] = pos
    return f


# ---------------------------------------------------------------- trích đặc trưng
def sent2features(sent, cfg, gaz=None):
    n = len(sent)
    runs = cap_runs(sent) if cfg["caprun"] else None
    gt = gaz_tags(sent, *gaz) if (cfg["gaz"] and gaz) else None
    feats = []
    for i in range(n):
        w = sent[i][0]
        f = {"bias": 1.0, "is_first": i == 0, "is_last": i == n - 1}
        f.update(tok_feats(sent, i, "", True))
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < n:
                f.update(tok_feats(sent, j, f"{off:+d}:", True))
        if i > 0:
            f["bigram_prev"] = sent[i - 1][0].lower() + "_" + w.lower()
        if i < n - 1:
            f["bigram_next"] = w.lower() + "_" + sent[i + 1][0].lower()

        if cfg["wide"]:  # cửa sổ ±3 (đặc trưng nhẹ) + trigram
            for off in (-3, 3):
                j = i + off
                if 0 <= j < n:
                    f.update(tok_feats(sent, j, f"{off:+d}:", False))
            if i > 1:
                f["trigram_prev"] = "_".join(t[0].lower() for t in sent[i - 2:i + 1])
            if i < n - 2:
                f["trigram_next"] = "_".join(t[0].lower() for t in sent[i:i + 3])

        if cfg["trigger"]:  # từ khoá gợi ý trong ±3
            for off in range(-3, 4):
                j = i + off
                if 0 <= j < n:
                    t = sent[j][0].lower()
                    if t in ORG_TRIG:
                        f[f"trg_org:{off:+d}"] = True
                    if t in LOC_TRIG:
                        f[f"trg_loc:{off:+d}"] = True
                    if t in PER_TRIG:
                        f[f"trg_per:{off:+d}"] = True

        if cfg["shape"]:
            f["shape"] = shape(w)
            f["is_abbrev"] = w.isalpha() and w.isupper() and len(w) >= 2
            f["alnum_mix"] = any(c.isdigit() for c in w) and any(c.isalpha() for c in w)
            if i > 0:
                f["-1:shape"] = shape(sent[i - 1][0])
            if i < n - 1:
                f["+1:shape"] = shape(sent[i + 1][0])

        if cfg["caprun"]:
            length, pos = runs[i]
            f["run_len"] = length
            f["run_pos"] = pos
            f["run_len_pos"] = f"{length}_{pos}"

        if gt:
            f["gaz"] = gt[i]
            if i > 0:
                f["-1:gaz"] = gt[i - 1]
            if i < n - 1:
                f["+1:gaz"] = gt[i + 1]
        feats.append(f)
    return feats


def featurize(data, cfg, gaz):
    return [sent2features(s, cfg, gaz) for s in data], [[t[2] for t in s] for s in data]


# ---------------------------------------------------------------- huấn luyện / đánh giá
def train_crf(X, y, c1, c2, iters):
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=c1, c2=c2, max_iterations=iters, all_possible_transitions=True
    )
    crf.fit(X, y)
    return crf


def eval_f1(crf, X, y):
    pred = crf.predict(X)
    rep = classification_report(y, pred, output_dict=True, digits=4)
    return f1_score(y, pred), {k: rep[k]["f1-score"] for k in ("LOC", "ORG", "PER", "MISC") if k in rep}


def error_analysis(crf, data, X, top=10):
    """Đếm token thực thể bị đoán sai theo (loại gốc -> loại dự đoán)."""
    pred = crf.predict(X)
    c = Counter()
    for sent, p in zip(data, pred):
        for (_, _, gold), guess in zip(sent, p):
            if gold == "O" or gold == guess:
                continue
            gt = gold[2:]
            pt = guess[2:] if guess != "O" else "O"
            c[(gt, "sai ranh giới B/I" if gt == pt else pt)] += 1
    print("\nLỗi phổ biến trên DEV (token thực thể bị đoán sai):")
    for (gt, pt), k in c.most_common(top):
        print(f"  {gt:5s} -> {pt:18s}: {k}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("dev")
    ap.add_argument("test")
    ap.add_argument("--gaz", default=None)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()

    train, dev, test = read_conll(a.train), read_conll(a.dev), read_conll(a.test)
    gaz = load_gaz(a.gaz) if a.gaz else None

    steps = [("base", {}), ("+trigger", {"trigger": True}), ("+shape", {"shape": True}),
             ("+caprun", {"caprun": True}), ("+wide", {"wide": True})]
    if gaz:
        steps.append(("+gaz", {"gaz": True}))

    # ---- 1) Ablation tích luỹ, chọn theo DEV
    cfg = {"trigger": False, "shape": False, "caprun": False, "wide": False, "gaz": False}
    results = []
    print("ABLATION (đánh giá trên DEV, c1=c2=0.1)")
    print(f"{'cấu hình':10s} {'F1':>7s} {'LOC':>7s} {'ORG':>7s} {'PER':>7s} {'MISC':>7s}")
    for name, upd in steps:
        cfg.update(upd)
        X_tr, y_tr = featurize(train, cfg, gaz)
        X_dv, y_dv = featurize(dev, cfg, gaz)
        crf = train_crf(X_tr, y_tr, 0.1, 0.1, a.iters)
        f1, per = eval_f1(crf, X_dv, y_dv)
        results.append((f1, name, dict(cfg)))
        print(f"{name:10s} {f1:7.4f} " + " ".join(f"{per.get(k, 0):7.4f}" for k in ("LOC", "ORG", "PER", "MISC")))
    best_f1, best_name, best_cfg = max(results, key=lambda r: r[0])
    print(f"\n=> Cấu hình tốt nhất trên DEV: {best_name} (F1={best_f1:.4f})")

    # ---- 2) Tune c1/c2 trên DEV
    X_tr, y_tr = featurize(train, best_cfg, gaz)
    X_dv, y_dv = featurize(dev, best_cfg, gaz)
    grid = [(0.1, 0.1)] if a.no_tune else [(0.05, 0.05), (0.1, 0.1), (0.2, 0.05), (0.1, 0.01)]
    iters = a.iters if a.no_tune else max(a.iters, 200)
    best = (-1, None, None)
    print("\nTUNE c1/c2 trên DEV")
    for c1, c2 in grid:
        crf = train_crf(X_tr, y_tr, c1, c2, iters)
        f1, _ = eval_f1(crf, X_dv, y_dv)
        print(f"  c1={c1:<5} c2={c2:<5} F1={f1:.4f}")
        if f1 > best[0]:
            best = (f1, (c1, c2), crf)
    final = best[2]
    print(f"=> Chọn c1,c2 = {best[1]} (DEV F1={best[0]:.4f})")

    # ---- 3) Phân tích lỗi (DEV) và đánh giá TEST một lần
    error_analysis(final, dev, X_dv)
    X_te, y_te = featurize(test, best_cfg, gaz)
    pred = final.predict(X_te)
    print("\n===== TEST (kết quả báo cáo) =====")
    print("Entity-level F1:", round(f1_score(y_te, pred), 4))
    print(classification_report(y_te, pred, digits=4))

    with open("crf_ner_v2.pkl", "wb") as f:
        pickle.dump({"model": final, "cfg": best_cfg}, f)


if __name__ == "__main__":
    main()
