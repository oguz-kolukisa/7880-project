import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_hw(x: torch.Tensor) -> torch.Tensor:
    """[B,C,H,W] -> [B,HW,C]."""
    return x.flatten(2).transpose(1, 2).contiguous()


def _resize_mask_to(mask: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize mask [B,1,H,W] to size_hw using nearest-neighbor."""
    if mask.dtype != torch.float32:
        mask = mask.float()
    return F.interpolate(mask, size=size_hw, mode="nearest")


def gather_tokens_from_mask_fixed(
    x: torch.Tensor,
    mask: torch.Tensor,
    max_tokens: int,
    ensure_nonempty: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather masked tokens up to max_tokens, pad to fixed length."""
    if mask.dim() == 4:
        mask = mask[:, 0]
    B, C, H, W = x.shape
    x_flat = _flatten_hw(x)
    m_flat = mask.flatten(1).bool()

    T = x.new_zeros((B, max_tokens, C))
    Tpad = torch.ones((B, max_tokens), device=x.device, dtype=torch.bool)
    for b in range(B):
        idx = torch.where(m_flat[b])[0]
        if idx.numel() == 0 and ensure_nonempty:
            idx = torch.arange(H * W, device=x.device)
        if idx.numel() > max_tokens:
            idx = idx[torch.randperm(idx.numel(), device=x.device)[:max_tokens]]
        n = idx.numel()
        if n > 0:
            T[b, :n] = x_flat[b, idx]
            Tpad[b, :n] = False
    return T, Tpad


def gather_fg_variable_tokens(
    x: torch.Tensor,
    fg_mask: torch.Tensor,
    max_tokens: Optional[int] = None,
    ensure_nonempty: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather variable-length foreground tokens, then pad to batch max length."""
    if fg_mask.dim() == 4:
        fg_mask = fg_mask[:, 0]
    B, C, H, W = x.shape
    x_flat = _flatten_hw(x)
    m_flat = fg_mask.flatten(1).bool()

    toks: List[torch.Tensor] = []
    lens: List[int] = []
    for b in range(B):
        idx = torch.where(m_flat[b])[0]
        if idx.numel() == 0 and ensure_nonempty:
            idx = torch.arange(H * W, device=x.device)
        if max_tokens is not None and idx.numel() > max_tokens:
            idx = idx[torch.randperm(idx.numel(), device=x.device)[:max_tokens]]
        tok = x_flat[b, idx]
        toks.append(tok)
        lens.append(tok.shape[0])

    Lmax = max(lens) if lens else 1
    T = x.new_zeros((B, Lmax, C))
    Tpad = torch.ones((B, Lmax), device=x.device, dtype=torch.bool)
    for b, tok in enumerate(toks):
        n = tok.shape[0]
        if n > 0:
            T[b, :n] = tok
            Tpad[b, :n] = False
    return T, Tpad


def build_S_QP_from_support(
    f_s: torch.Tensor,
    y_s: torch.Tensor,
    max_support_tokens: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect support foreground tokens across K shots."""
    B, K, C, Hf, Wf = f_s.shape
    y_s_rs = _resize_mask_to(y_s.view(B * K, 1, *y_s.shape[-2:]), (Hf, Wf)).view(B, K, 1, Hf, Wf)

    all_tokens: List[torch.Tensor] = []
    lens: List[int] = []
    f_s_flat = f_s.flatten(3).transpose(2, 3).contiguous()
    y_flat = y_s_rs[:, :, 0].flatten(2).bool()

    for b in range(B):
        toks_b = []
        for k in range(K):
            idx = torch.where(y_flat[b, k])[0]
            if idx.numel() == 0:
                idx = torch.arange(Hf * Wf, device=f_s.device)
            tok = f_s_flat[b, k, idx]
            toks_b.append(tok)
        tok_cat = torch.cat(toks_b, dim=0) if toks_b else f_s.new_zeros((1, C))
        if tok_cat.shape[0] > max_support_tokens:
            sel = torch.randperm(tok_cat.shape[0], device=f_s.device)[:max_support_tokens]
            tok_cat = tok_cat[sel]
        all_tokens.append(tok_cat)
        lens.append(tok_cat.shape[0])

    Ls = max(lens) if lens else 1
    S_QP = f_s.new_zeros((B, Ls, C))
    S_QP_pad = torch.ones((B, Ls), device=f_s.device, dtype=torch.bool)
    for b, tok in enumerate(all_tokens):
        n = tok.shape[0]
        S_QP[b, :n] = tok
        S_QP_pad[b, :n] = False
    return S_QP, S_QP_pad


class QueryPrediction(nn.Module):
    """Prototype-based query decoder."""

    def __init__(self, C: int, num_classes: int = 2):
        super().__init__()
        self.phi_p = nn.Sequential(
            nn.Conv2d(2 * C, C, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, num_classes, 1),
        )

    def forward(
        self,
        S_QP: torch.Tensor,
        f_q: torch.Tensor,
        S_QP_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if S_QP_pad is None:
            s_qp = S_QP.mean(dim=1)
        else:
            keep = (~S_QP_pad).float().unsqueeze(-1)
            denom = keep.sum(dim=1).clamp_min(1.0)
            s_qp = (S_QP * keep).sum(dim=1) / denom

        B, C, H, W = f_q.shape
        s_qp = s_qp.view(B, C, 1, 1).expand(B, C, H, W)
        x = torch.cat([s_qp, f_q], dim=1)
        return self.phi_p(x)


class SoftHistogram(nn.Module):
    """Triangular soft binning over [0,1]."""

    def __init__(self, L: int = 16):
        super().__init__()
        edges = torch.linspace(0.0, 1.0, L + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.register_buffer("centers", centers)
        self.width = 1.0 / L

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        a = a.clamp(0.0, 1.0).unsqueeze(-1)
        c = self.centers.unsqueeze(0)
        w = self.width
        wts = (1.0 - (a - c).abs() / w).clamp_min(0.0)
        return wts.sum(dim=0)


class EvolutionFeature(nn.Module):
    """Compute pixel-wise and structure-wise evolution descriptors."""

    def __init__(self, C: int, L: int = 16, max_fg_tokens: int = 512, max_pairs: int = 8192):
        super().__init__()
        self.C = C
        self.L = L
        self.max_fg_tokens = max_fg_tokens
        self.max_pairs = max_pairs

        self.conv_I = nn.Conv2d(3, C, 1)
        self.conv_O = nn.Conv2d(C, C, 1)

        self.mlp_Ep = nn.Sequential(
            nn.Linear(2 * C, C),
            nn.ReLU(inplace=True),
            nn.Linear(C, C),
        )

        self.hist = SoftHistogram(L)
        self.mlp_Es = nn.Sequential(
            nn.Linear(2 * L, C),
            nn.ReLU(inplace=True),
            nn.Linear(C, C),
        )

    def forward(
        self,
        I_q: torch.Tensor,
        f_q: torch.Tensor,
        P: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = f_q.shape
        pred = P.argmax(dim=1, keepdim=True)
        fg_mask = (pred == 1).float()

        I_q_rs = F.interpolate(I_q, size=(H, W), mode="bilinear", align_corners=False)

        F_I = self.conv_I(I_q_rs)
        F_O = self.conv_O(f_q)

        F_I_tok, E_pad = gather_fg_variable_tokens(F_I, fg_mask, max_tokens=self.max_fg_tokens)
        F_O_tok, _ = gather_fg_variable_tokens(F_O, fg_mask, max_tokens=self.max_fg_tokens)

        E_p = self.mlp_Ep(torch.cat([F_I_tok, F_O_tok], dim=-1))

        T_I, TI_pad = gather_fg_variable_tokens(I_q_rs, fg_mask, max_tokens=self.max_fg_tokens)
        T_O, _ = gather_fg_variable_tokens(f_q, fg_mask, max_tokens=self.max_fg_tokens)

        Es_list: List[torch.Tensor] = []
        for b in range(B):
            keep = ~TI_pad[b]
            ti = T_I[b, keep]
            to = T_O[b, keep]
            if ti.numel() == 0 or to.numel() == 0:
                Es_list.append(f_q.new_zeros((C,)))
                continue

            ti_n = F.normalize(ti, dim=1)
            to_n = F.normalize(to, dim=1)
            A_I = (ti_n @ ti_n.t()).add(1.0).mul(0.5).clamp(0.0, 1.0)
            A_O = (to_n @ to_n.t()).add(1.0).mul(0.5).clamp(0.0, 1.0)

            a_i = A_I.flatten()
            a_o = A_O.flatten()
            if a_i.numel() > self.max_pairs:
                sel = torch.randperm(a_i.numel(), device=f_q.device)[:self.max_pairs]
                a_i = a_i[sel]
                a_o = a_o[sel]

            H_I = self.hist(a_i)
            H_O = self.hist(a_o)
            H_I = H_I / H_I.sum().clamp_min(1.0)
            H_O = H_O / H_O.sum().clamp_min(1.0)

            E_s = self.mlp_Es(torch.cat([H_I, H_O], dim=0))
            Es_list.append(E_s)

        E_s = torch.stack(Es_list, dim=0)
        E = E_p + E_s.unsqueeze(1)
        return E, E_pad


class SupportModulation(nn.Module):
    """Align support tokens with query context via dual attention."""

    def __init__(self, C: int, evo: EvolutionFeature, num_heads: int = 4, max_bg_tokens: int = 512):
        super().__init__()
        self.evo = evo
        self.max_bg_tokens = max_bg_tokens
        self.ATT_BE = nn.MultiheadAttention(C, num_heads=num_heads, batch_first=True)
        self.ATT_SC = nn.MultiheadAttention(C, num_heads=num_heads, batch_first=True)

    def forward(
        self,
        S_QP: torch.Tensor,
        f_q: torch.Tensor,
        P: torch.Tensor,
        I_q: torch.Tensor,
        S_QP_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        E, E_pad = self.evo(I_q, f_q, P)

        pred = P.argmax(dim=1, keepdim=True)
        bg_mask = (pred == 0).float()
        B_tok, B_pad = gather_tokens_from_mask_fixed(f_q, bg_mask, max_tokens=self.max_bg_tokens)

        C_ctx, _ = self.ATT_BE(query=B_tok, key=E, value=E, key_padding_mask=E_pad)

        delta, _ = self.ATT_SC(query=S_QP, key=C_ctx, value=C_ctx, key_padding_mask=B_pad)
        if S_QP_pad is not None:
            delta = delta.masked_fill(S_QP_pad.unsqueeze(-1), 0.0)
        return S_QP + delta


def entropy_confidence(P: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Entropy-based uncertainty map."""
    p = torch.softmax(P, dim=1).clamp_min(eps)
    return -(p * p.log()).sum(dim=1, keepdim=True)


class ConfidenceBiasedCrossAttention(nn.Module):
    """Cross-attn with per-key bias added to logits."""

    def __init__(self, C: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert C % num_heads == 0
        self.C = C
        self.H = num_heads
        self.D = C // num_heads
        self.Wq = nn.Linear(C, C)
        self.Wk = nn.Linear(C, C)
        self.Wv = nn.Linear(C, C)
        self.Wo = nn.Linear(C, C)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K_in: torch.Tensor,
        V_in: torch.Tensor,
        V_bias: torch.Tensor,
    ) -> torch.Tensor:
        B, Lq, C = Q.shape
        _, Lk, _ = K_in.shape

        q = self.Wq(Q).view(B, Lq, self.H, self.D).transpose(1, 2)
        k = self.Wk(K_in).view(B, Lk, self.H, self.D).transpose(1, 2)
        v = self.Wv(V_in).view(B, Lk, self.H, self.D).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.D)
        logits = logits + V_bias[:, None, None, :]

        attn = torch.softmax(logits, dim=-1)
        attn = self.drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, Lq, C)
        return self.Wo(out)


class InformationCleansing(nn.Module):
    """Use confidence drop to suppress noisy support tokens."""

    def __init__(self, C: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.phi_Q = nn.Sequential(
            nn.Linear(2 * C, C),
            nn.ReLU(inplace=True),
            nn.Linear(C, C),
        )
        self.attn = ConfidenceBiasedCrossAttention(C, num_heads=num_heads, dropout=dropout)
        self.phi = nn.Linear(C, C)

    def forward(
        self,
        QP: QueryPrediction,
        S_QP: torch.Tensor,
        S_SM: torch.Tensor,
        f_q: torch.Tensor,
        P: torch.Tensor,
        S_QP_pad: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        P_hat = QP(S_SM, f_q, S_QP_pad)

        C_map = entropy_confidence(P)
        C_hat = entropy_confidence(P_hat)
        V = C_hat - C_map

        f_q_flat = _flatten_hw(f_q)
        V_flat = V.flatten(2).squeeze(1).contiguous()

        Q = self.phi_Q(torch.cat([S_QP, S_SM], dim=-1))
        N = self.attn(Q=Q, K_in=f_q_flat, V_in=f_q_flat, V_bias=V_flat)
        S_IC = S_SM - self.phi(N)
        return S_IC, P_hat, V


class ResNet50Backbone(nn.Module):
    """Torchvision ResNet-50 truncated before the classification head."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet50

        try:
            from torchvision.models import ResNet50_Weights

            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            m = resnet50(weights=weights)
        except Exception:
            m = resnet50(pretrained=pretrained)

        self.backbone = nn.Sequential(*list(m.children())[:-2])
        self.out_channels = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ABCB(nn.Module):
    """Full iterative ABCB segmentation model."""

    def __init__(
        self,
        dim: int = 256,
        T: int = 3,
        pretrained_backbone: bool = True,
        num_heads: int = 4,
        L_hist: int = 16,
        max_support_tokens: int = 1024,
        max_fg_tokens: int = 512,
        max_bg_tokens: int = 512,
        normalize_imagenet: bool = True,
    ):
        super().__init__()
        self.T = T
        self.max_support_tokens = max_support_tokens
        self.normalize_imagenet = normalize_imagenet

        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        self.proj = nn.Conv2d(self.backbone.out_channels, dim, kernel_size=1)

        self.QP = QueryPrediction(C=dim, num_classes=2)
        self.evo = EvolutionFeature(C=dim, L=L_hist, max_fg_tokens=max_fg_tokens)
        self.SM = SupportModulation(C=dim, evo=self.evo, num_heads=num_heads, max_bg_tokens=max_bg_tokens)
        self.IC = InformationCleansing(C=dim, num_heads=num_heads)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean, persistent=False)
        self.register_buffer("img_std", std, persistent=False)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.img_mean) / self.img_std

    def forward(
        self,
        query_img: torch.Tensor,
        support_img: torch.Tensor,
        support_mask: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, Any]:
        B, K = support_img.shape[:2]
        Hq, Wq = query_img.shape[-2:]

        if self.normalize_imagenet:
            query_in = self._norm(query_img)
            support_in = self._norm(support_img.view(B * K, 3, Hq, Wq)).view(B, K, 3, Hq, Wq)
        else:
            query_in = query_img
            support_in = support_img

        f_q = self.proj(self.backbone(query_in))
        f_s = self.proj(self.backbone(support_in.view(B * K, 3, Hq, Wq))).view(
            B, K, -1, f_q.shape[-2], f_q.shape[-1]
        )

        S_QP, S_QP_pad = build_S_QP_from_support(f_s, support_mask, max_support_tokens=self.max_support_tokens)

        P_list: List[torch.Tensor] = []
        Phat_list: List[torch.Tensor] = []
        extras_SM: List[torch.Tensor] = []
        extras_V: List[torch.Tensor] = []
        for _ in range(self.T):
            P = self.QP(S_QP, f_q, S_QP_pad)
            P_list.append(P)

            S_SM = self.SM(S_QP=S_QP, f_q=f_q, P=P, I_q=query_img, S_QP_pad=S_QP_pad)

            S_IC, P_hat, V = self.IC(QP=self.QP, S_QP=S_QP, S_SM=S_SM, f_q=f_q, P=P, S_QP_pad=S_QP_pad)
            Phat_list.append(P_hat)
            extras_SM.append(S_SM)
            extras_V.append(V)

            S_QP = S_IC

        P_final = P_list[-1]
        logits = F.interpolate(P_final, size=(Hq, Wq), mode="bilinear", align_corners=False)

        out: Dict[str, Any] = {"logits": logits}
        if not return_all:
            return out

        out.update(
            {
                "P_list": P_list,
                "Phat_list": Phat_list,
                "extras": {
                    "P_final_feat": P_final,
                    "P_all_feat": torch.stack(P_list, dim=1),
                    "S_QP_final": S_QP,
                    "S_QP_pad": S_QP_pad,
                    "f_q": f_q,
                    "f_s": f_s,
                    "S_SM_list": extras_SM,
                    "V_list": extras_V,
                },
            }
        )
        return out

