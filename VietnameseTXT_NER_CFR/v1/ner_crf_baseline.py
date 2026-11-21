"""
Baseline NER tiếng Việt: CRF + đặc trưng thủ công.
Dữ liệu: VLSP 2016 (định dạng CoNLL: mỗi dòng "token [POS] [chunk] NER", câu cách nhau dòng trống).
Cài đặt: pip install sklearn-crfsuite seqeval
Chạy:    python ner_crf_baseline.py train.txt dev.txt test.txt
"""
import sys
import pickle
import sklearn_crfsuite
from seqeval.metrics import classification_report, f1_score


def read_conll(path):
    """Trả về list câu; mỗi câu là list (token, pos, ner). Cột đầu = token, cột cuối = nhãn NER."""
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
            token, ner = cols[0], cols[-1]
            pos = cols[1] if len(cols) > 2 else "_"
            cur.append((token, pos, ner))
    if cur:
        sents.append(cur)
    return sents


def word_feats(sent, i, prefix=""):
    w, pos, _ = sent[i]
    return {
        prefix + "w.lower": w.lower(),
        prefix + "w.istitle": w.istitle(),
        prefix + "w.isupper": w.isupper(),
        prefix + "w.isdigit": w.isdigit(),
        prefix + "w.has_hyphen": "-" in w,
        prefix + "w.pre2": w[:2].lower(),
        prefix + "w.suf2": w[-2:].lower(),
        prefix + "pos": pos,
    }


def sent2features(sent):
    feats = []
    for i in range(len(sent)):
        f = {"bias": 1.0}
        f.update(word_feats(sent, i))
        # ngữ cảnh cửa sổ ±2
        for off in (-2, -1, 1, 2):
            j = i + off
            if 0 <= j < len(sent):
                f.update(word_feats(sent, j, prefix=f"{off:+d}:"))
            else:
                f[f"{off:+d}:BOS_EOS"] = True
        # bigram: đặc biệt hữu ích vì tiếng Việt là ngôn ngữ đơn lập (tên riêng gồm nhiều âm tiết)
        if i > 0:
            f["bigram_prev"] = sent[i - 1][0].lower() + "_" + sent[i][0].lower()
        if i < len(sent) - 1:
            f["bigram_next"] = sent[i][0].lower() + "_" + sent[i + 1][0].lower()
        feats.append(f)
    return feats


def sent2labels(sent):
    return [x[2] for x in sent]


def main(train_p, dev_p, test_p):
    train, dev, test = read_conll(train_p), read_conll(dev_p), read_conll(test_p)
    X_tr = [sent2features(s) for s in train]
    y_tr = [sent2labels(s) for s in train]

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(X_tr, y_tr)

    for name, data in (("DEV", dev), ("TEST", test)):
        X = [sent2features(s) for s in data]
        y = [sent2labels(s) for s in data]
        pred = crf.predict(X)
        print(f"\n===== {name} =====")
        print("Entity-level F1:", round(f1_score(y, pred), 4))
        print(classification_report(y, pred, digits=4))

    with open("crf_ner.pkl", "wb") as f:
        pickle.dump(crf, f)


if __name__ == "__main__":
    main(*sys.argv[1:4])
