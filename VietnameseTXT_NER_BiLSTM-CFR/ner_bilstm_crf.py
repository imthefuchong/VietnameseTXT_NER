"""
BiLSTM-CRF cho NER tiếng Việt (VLSP 2016), PyTorch. Phiên bản 2.
Kiến trúc: word embedding [+ char-CNN] [+ case embedding] -> BiLSTM -> Linear -> CRF.

Cài đặt: pip install torch pytorch-crf seqeval
Chạy:    python ner_bilstm_crf.py train.txt dev.txt test.txt

Tuỳ chọn:
  --emb FILE        embedding tiền huấn luyện (text, dạng word2vec/.vec), nên dùng bản mức âm tiết
  --normalize       khi tra embedding, thử cả hai cách đặt dấu thanh (hòa/hoà, thủy/thuỷ)
  --freeze-emb      không tinh chỉnh word embedding (chống overfit khi dùng embedding tiền huấn luyện)
  --no-char         tắt char-CNN
  --case            thêm embedding kiểu viết hoa (thường / Title / TOÀN HOA / số / khác)
  --seeds 1 2 3     chạy nhiều seed và in trung bình ± độ lệch chuẩn
  --epochs N        số epoch tối đa (mặc định 50); --patience N cho early stopping theo DEV
Quy trình mỗi seed: huấn luyện, chọn epoch tốt nhất theo DEV, đánh giá TEST đúng một lần.
"""
import argparse
import copy
import random
import statistics
import unicodedata
from collections import Counter

import torch
import torch.nn as nn
from torchcrf import CRF
from seqeval.metrics import classification_report, f1_score

PAD, UNK = 0, 1
MAX_CHARS = 20
CLASSES = ("LOC", "ORG", "PER", "MISC")


# ---------------------------------------------------------------- chuẩn hoá chữ / dấu thanh
TONES = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}  # huyền, sắc, ngã, hỏi, nặng


def norm(w):
    return unicodedata.normalize("NFC", w).lower()


def split_tone(ch):
    d = unicodedata.normalize("NFD", ch)
    tone = next((c for c in d if c in TONES), None)
    base = unicodedata.normalize("NFC", "".join(c for c in d if c not in TONES))
    return base, tone


def add_tone(base, tone):
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", base) + tone)


def tone_variants(w):
    """Với âm tiết mở kết thúc bằng oa/oe/uy, trả về cả hai kiểu đặt dấu (hòa/hoà)."""
    w = norm(w)
    if len(w) >= 2:
        (b1, t1), (b2, t2) = split_tone(w[-2]), split_tone(w[-1])
        if b1 + b2 in ("oa", "oe", "uy") and (t1 is None) != (t2 is None):
            tone = t1 or t2
            return {w, w[:-2] + add_tone(b1, tone) + b2, w[:-2] + b1 + add_tone(b2, tone)}
    return {w}


def case_id(w):
    if w.isdigit():
        return 4
    if w[:1].isupper():
        return 3 if (w.isupper() and len(w) > 1) else 2
    if w.islower():
        return 1
    return 5


# ---------------------------------------------------------------- dữ liệu
def read_conll(path):
    """Trả về (list câu gồm token, list câu gồm nhãn NER). Cột đầu = token, cột cuối = NER."""
    sents, tags, cs, ct = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cs:
                    sents.append(cs)
                    tags.append(ct)
                    cs, ct = [], []
                continue
            cols = line.split()
            cs.append(unicodedata.normalize("NFC", cols[0]))
            ct.append(cols[-1])
    if cs:
        sents.append(cs)
        tags.append(ct)
    return sents, tags


def build_vocabs(sents, tags):
    wcount = Counter(w.lower() for s in sents for w in s)
    wv = {"<pad>": PAD, "<unk>": UNK}
    for w in wcount:
        wv[w] = len(wv)
    cv = {"<pad>": PAD, "<unk>": UNK}
    for ch in sorted({ch for s in sents for w in s for ch in w}):
        cv[ch] = len(cv)
    tv = {t: i for i, t in enumerate(sorted({t for ts in tags for t in ts}))}
    return wv, cv, tv, wcount


def batchify(sents, tags, wv, cv, tv, wcount=None, train=False):
    B = len(sents)
    T = max(len(s) for s in sents)
    L = min(max(len(w) for s in sents for w in s), MAX_CHARS)
    W = torch.zeros(B, T, dtype=torch.long)
    C = torch.zeros(B, T, L, dtype=torch.long)
    K = torch.zeros(B, T, dtype=torch.long)
    Y = torch.zeros(B, T, dtype=torch.long)
    M = torch.zeros(B, T, dtype=torch.bool)
    lengths = torch.tensor([len(s) for s in sents])
    for i, s in enumerate(sents):
        for j, w in enumerate(s):
            lw = w.lower()
            idx = wv.get(lw, UNK)
            # word dropout: thay từ hiếm (xuất hiện 1 lần) bằng <unk> để mô hình quen với từ lạ
            if train and wcount.get(lw, 0) == 1 and random.random() < 0.5:
                idx = UNK
            W[i, j] = idx
            K[i, j] = case_id(w)
            for k, ch in enumerate(w[:L]):
                C[i, j, k] = cv.get(ch, UNK)
            M[i, j] = True
            if tags is not None:
                Y[i, j] = tv[tags[i][j]]
    return W, C, K, Y, M, lengths


# ---------------------------------------------------------------- mô hình
class BiLSTMCRF(nn.Module):
    def __init__(self, nw, nc, ntag, wdim=100, cdim=30, cfilters=50, kdim=10, hidden=200,
                 dropout=0.5, use_char=True, use_case=False, pretrained=None):
        super().__init__()
        self.use_char, self.use_case = use_char, use_case
        self.wemb = nn.Embedding(nw, wdim, padding_idx=PAD)
        if pretrained is not None:
            self.wemb.weight.data.copy_(pretrained)
        in_dim = wdim
        if use_char:
            self.cemb = nn.Embedding(nc, cdim, padding_idx=PAD)
            self.cconv = nn.Conv1d(cdim, cfilters, kernel_size=3, padding=1)
            in_dim += cfilters
        if use_case:
            self.kemb = nn.Embedding(6, kdim, padding_idx=PAD)
            in_dim += kdim
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(in_dim, hidden // 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden, ntag)
        self.crf = CRF(ntag, batch_first=True)

    def emissions(self, W, C, K, lengths):
        T = W.size(1)
        x = [self.wemb(W)]
        if self.use_char:
            B, T_, L = C.shape
            ce = self.cemb(C.view(B * T_, L)).transpose(1, 2)           # (B*T, cdim, L)
            x.append(torch.relu(self.cconv(ce)).max(dim=2)[0].view(B, T_, -1))
        if self.use_case:
            x.append(self.kemb(K))
        x = self.drop(torch.cat(x, dim=-1))
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        return self.fc(self.drop(out))

    def loss(self, W, C, K, Y, M, lengths):
        return -self.crf(self.emissions(W, C, K, lengths), Y, mask=M, reduction="mean")

    @torch.no_grad()
    def decode(self, W, C, K, M, lengths):
        return self.crf.decode(self.emissions(W, C, K, lengths), mask=M)


# ---------------------------------------------------------------- embedding tiền huấn luyện
def load_pretrained(path, wv, wcount, use_variants):
    """Trả về ma trận (len(wv), dim). In độ phủ và các từ phổ biến nhất không có vector."""
    wanted = set()
    for w in wv:
        wanted |= tone_variants(w) if use_variants else {w}
    found, dim = {}, None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) == 2 and parts[0].isdigit():  # dòng header của .vec
                continue
            if dim is None:
                dim = len(parts) - 1
            if len(parts) - 1 != dim:
                continue
            key = norm(parts[0])
            if key in wanted and key not in found:
                found[key] = torch.tensor([float(x) for x in parts[1:]])

    vecs = {}
    for w in wv:
        if w in found:
            vecs[w] = found[w]
        elif use_variants:
            for v in tone_variants(w):
                if v in found:
                    vecs[w] = found[v]
                    break
    std = torch.stack(list(vecs.values())).std().item() if vecs else 0.1
    mat = torch.randn(len(wv), dim) * std  # từ thiếu vector: khởi tạo cùng thang giá trị với vector thật
    mat[PAD] = 0
    for w, v in vecs.items():
        mat[wv[w]] = v

    total = sum(wcount.values())
    covered = sum(c for w, c in wcount.items() if w in vecs)
    print(f"Embedding: {len(vecs)}/{len(wcount)} từ vựng có vector ({len(vecs) / len(wcount):.1%}), "
          f"phủ {covered / total:.1%} token trong train (dim={dim})")
    miss = [(w, c) for w, c in wcount.most_common() if w not in vecs][:30]
    print("Từ phổ biến nhất KHÔNG có vector:", ", ".join(f"{w}({c})" for w, c in miss))
    return mat


# ---------------------------------------------------------------- huấn luyện / đánh giá
def predict(model, sents, wv, cv, tv, device, bs=64):
    itv = {i: t for t, i in tv.items()}
    model.eval()
    out = []
    for i in range(0, len(sents), bs):
        W, C, K, _, M, lengths = batchify(sents[i:i + bs], None, wv, cv, tv)
        W, C, K, M = W.to(device), C.to(device), K.to(device), M.to(device)
        for seq in model.decode(W, C, K, M, lengths):
            out.append([itv[k] for k in seq])
    return out


def run(a, seed, data, device):
    (tr_s, tr_t), (dv_s, dv_t), (te_s, te_t) = data
    random.seed(seed)
    torch.manual_seed(seed)
    wv, cv, tv, wcount = build_vocabs(tr_s, tr_t)
    pre = load_pretrained(a.emb, wv, wcount, a.normalize) if a.emb else None
    wdim = pre.size(1) if pre is not None else 100

    model = BiLSTMCRF(len(wv), len(cv), len(tv), wdim=wdim, use_char=not a.no_char,
                      use_case=a.case, pretrained=pre).to(device)
    if a.freeze_emb:
        model.wemb.weight.requires_grad = False
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    best_f1, best_state, bad = -1.0, None, 0
    idx = list(range(len(tr_s)))
    for ep in range(1, a.epochs + 1):
        model.train()
        random.shuffle(idx)
        total = 0.0
        for i in range(0, len(idx), a.batch):
            b = idx[i:i + a.batch]
            W, C, K, Y, M, lengths = batchify([tr_s[k] for k in b], [tr_t[k] for k in b],
                                              wv, cv, tv, wcount, train=True)
            W, C, K, Y, M = W.to(device), C.to(device), K.to(device), Y.to(device), M.to(device)
            opt.zero_grad()
            loss = model.loss(W, C, K, Y, M, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(b)
        dev_f1 = f1_score(dv_t, predict(model, dv_s, wv, cv, tv, device))
        print(f"[seed {seed}] Epoch {ep:2d} | loss {total / len(idx):8.4f} | DEV F1 {dev_f1:.4f}")
        if dev_f1 > best_f1:
            best_f1, best_state, bad = dev_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Early stopping (không cải thiện sau {a.patience} epoch)")
                break

    model.load_state_dict(best_state)
    pred = predict(model, te_s, wv, cv, tv, device)
    test_f1 = f1_score(te_t, pred)
    print(f"\n=> [seed {seed}] DEV F1 tốt nhất = {best_f1:.4f}")
    print(f"===== TEST (seed {seed}) =====")
    print("Entity-level F1:", round(test_f1, 4))
    print(classification_report(te_t, pred, digits=4))
    rep = classification_report(te_t, pred, output_dict=True)
    torch.save({"state": best_state, "wv": wv, "cv": cv, "tv": tv}, f"bilstm_crf_seed{seed}.pt")
    return {"dev": best_f1, "test": test_f1, **{c: rep[c]["f1-score"] for c in CLASSES if c in rep}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("dev")
    ap.add_argument("test")
    ap.add_argument("--emb", default=None)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--freeze-emb", action="store_true")
    ap.add_argument("--no-char", action="store_true")
    ap.add_argument("--case", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Thiết bị:", device)
    data = (read_conll(a.train), read_conll(a.dev), read_conll(a.test))

    results = [run(a, s, data, device) for s in a.seeds]

    if len(results) > 1:
        print(f"\n===== TỔNG HỢP {len(results)} seed {a.seeds} =====")
        for key in ("dev", "test") + CLASSES:
            vals = [r[key] for r in results if key in r]
            print(f"{key:5s}: {statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}")


if __name__ == "__main__":
    main()
