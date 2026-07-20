"""
merge_results.py — combine SITS (AI) + NDWI + S1 (bi-temporal) into a per-event flood area.

Inputs:
  data/sits_scores/{event}.npz   (Track B: SITS score + NDWI per patch)  -- from sits_score.py
  data/flood_extent.csv          (Track A: area_s1_km2, area_s2_km2, s2_post_images) -- re-run Track A first

Per event:
  1. SITS is patch-level (a 0.4096 km2 tile is flagged whole even if only a sliver is water),
     so (#flagged * PATCH_KM2) is a DETECTION extent, not a water area (measured: tiles are
     ~0.8% water at the median -> using it as area over-counts ~100x).
     => SITS LOCATES flood tiles; a quality gate (Youden's J = SITS-NDWI agreement, not
        external validation) decides whether the fusion is used:
        (a) ndwi-calib  -- J>=0.15: SITS+NDWI fusion (SITS locates, NDWI quantifies)
        (b) otsu/low-conf -- SITS flags unreliable -> whole-district NDWI instead
  2. Flood AREA always comes from pixel-level NDWI (real water):
        SITS+NDWI  -> NDWI water *inside* SITS-flagged tiles (gated)
        NDWI       -> NDWI water over the whole district
     Restore rule: if gating cut the water by >half AND S1 (radar) independently shows the
     larger extent is real, restore full-district NDWI (guards against SITS missing tiles).
  3. Cloud fusion (replaces the old all-or-nothing cloud routing):
     The flood-date S2 composite splits the district into a part optical SAW and a part it
     did not. s1_cloudy.py measures the Track-A S1 flood area separately over each part, so
     the two halves are complementary and simply add:
         combined = optical (over the clear part) + s1_cloudy_km2 (over the cloudy part)
     Previously a >60%-cloud event threw the optical number away and used whole-district S1;
     now each sensor contributes exactly the ground it actually observed.
     Events with 0 SITS patches (no usable optical) fall back to whole-district S1.

Output: data/flood_combined.csv
Run:    python src/merge_results.py     (numpy + pandas only; no model / GEE)
"""
import os
import glob
import numpy as np
import pandas as pd

SCORES_DIR = 'data/sits_scores'
TRACK_A_CSV = 'data/flood_extent.csv'
OUT_CSV = 'data/flood_combined.csv'

PATCH_KM2 = (64 * 10 / 1000) ** 2   # 0.4096 km2 per patch
PX_KM2 = (10 / 1000) ** 2           # 1e-4 km2 per pixel
FLOOD_MIN_PX = 205                  # a patch is an "NDWI flood" patch if >= 5% (205/4096) is new water
MIN_POS = 20                        # need this many NDWI-flood (and non-flood) patches to calibrate
J_MIN = 0.15                        # min Youden's J (SITS-NDWI agreement) to use the SITS+NDWI fusion
NDWI_FALLBACK_MAX_CLOUD_PCT = 60    # no SITS and no S1: only trust Track A NDWI below this cloud
POST_CLOUD_CSV = 'data/post_cloud.csv'    # event_id,cloud_pct  (from post_cloud.py)
DISTRICT_CSV = 'data/district_area.csv'   # event_id,district_km2  (from district_area.py)
S1_CLOUDY_CSV = 'data/s1_cloudy.csv'      # event_id,s1_clear_km2,s1_cloudy_km2  (from s1_cloudy.py)


def _otsu(x):
    """Otsu threshold on scores in [0,1]."""
    hist, edges = np.histogram(x, bins=64, range=(0.0, 1.0))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.5
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = total - w0
    csum = np.cumsum(hist * centers)
    mu_total = csum[-1] / total
    with np.errstate(divide='ignore', invalid='ignore'):
        mu0 = csum / w0
        mu1 = (csum[-1] - csum) / w1
        var_between = w0 * w1 * (mu0 - mu1) ** 2
    var_between = np.nan_to_num(var_between)
    return float(centers[int(np.argmax(var_between))])


def _youden(scores, labels):
    """Threshold that maximises TPR - FPR against NDWI-flood labels; returns (t, J).
    This is a *balanced* boundary (not 'capture 85% of positives'), so it does not
    over-flag the way a low percentile does. J also measures how separable the two are."""
    P = int(labels.sum())
    N = int((~labels).sum())
    if P == 0 or N == 0:
        return None, 0.0
    ts = np.quantile(scores, np.linspace(0.02, 0.98, 60))
    best_t, best_j = float(np.median(scores)), -1.0
    for t in ts:
        pred = scores > t
        tpr = (pred & labels).sum() / P
        fpr = (pred & ~labels).sum() / N
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t, best_j


def sits_threshold(scores, ndwi_flood):
    """Return (threshold, method): NDWI-calibrated (Youden's J) -> Otsu -> low-confidence."""
    flood = ndwi_flood >= FLOOD_MIN_PX
    if int(flood.sum()) >= MIN_POS and int((~flood).sum()) >= MIN_POS:
        t, j = _youden(scores, flood)
        if t is not None and j >= J_MIN:            # good separation -> trust SITS
            return t, 'ndwi-calib'
        return _otsu(scores), 'low-conf'            # NDWI floods don't separate -> SITS unreliable
    return _otsu(scores), 'otsu'                    # too few NDWI positives -> plain Otsu


def main():
    # Track A (S1 + optical availability); may be missing/stale -> handle gracefully
    ta = {}
    if os.path.exists(TRACK_A_CSV):
        dfa = pd.read_csv(TRACK_A_CSV)
        for _, r in dfa.iterrows():
            ta[r['event_id']] = r
    else:
        print(f"WARN: {TRACK_A_CSV} missing -> no S1 / cloud routing (re-run Track A)")

    # flood-date cloud fraction over the district (from post_cloud.py); optional
    pcloud = {}
    if os.path.exists(POST_CLOUD_CSV):
        dfc = pd.read_csv(POST_CLOUD_CSV)
        pcloud = dict(zip(dfc['event_id'], dfc['cloud_pct']))
    else:
        print(f"WARN: {POST_CLOUD_CSV} missing -> no cloud-fraction routing (run post_cloud.py)")

    # S1 flood area split by what the flood-date S2 composite saw (from s1_cloudy.py).
    # s1_clear_km2 + s1_cloudy_km2 == Track A area_s1_km2 (verified 126/129 within 1%).
    s1split = {}
    if os.path.exists(S1_CLOUDY_CSV):
        dfs = pd.read_csv(S1_CLOUDY_CSV)
        for _, r in dfs.iterrows():
            if pd.notna(r.get('s1_cloudy_km2')):
                s1split[r['event_id']] = (float(r['s1_clear_km2']),
                                          float(r['s1_cloudy_km2']))
        print(f"s1_cloudy split loaded for {len(s1split)} events")
    else:
        print(f"WARN: {S1_CLOUDY_CSV} missing -> no cloud fusion (run s1_cloudy.py)")

    # district area (for flood_ratio); optional
    da = {}
    if os.path.exists(DISTRICT_CSV):
        dfd = pd.read_csv(DISTRICT_CSV)
        da = dict(zip(dfd['event_id'], dfd['district_km2']))
    else:
        print(f"WARN: {DISTRICT_CSV} missing -> no flood_ratio (run district_area.py)")

    npz = {os.path.basename(f)[:-4]: f
           for f in glob.glob(os.path.join(SCORES_DIR, '*.npz'))}
    # union: every event with a SITS score OR a Track A row (cloud-blind 0-patch events
    # have no .npz but still have S1/NDWI in Track A -> route them to S1, don't drop them)
    all_events = sorted(set(npz) | set(ta))

    rows = []
    for ev in all_events:
        a = ta.get(ev, {})
        s1_area = a.get('area_s1_km2', None)
        s2_area = a.get('area_s2_km2', None)                # Track A optical (NDWI over district)
        optical_ok = pd.notna(s2_area) and float(a.get('s2_post_images', 0) or 0) > 0

        d = np.load(npz[ev]) if ev in npz else None
        has_sits = d is not None and len(d['scores']) > 0   # empty npz = 0 patches = cloud-blind

        if has_sits:
            # SITS kept clear patches (KEEP_VALID>=70%) -> optical DID see the flood.
            scores, ndwi_flood = d['scores'], d['ndwi_flood']
            thr, method = sits_threshold(scores, ndwi_flood)
            flagged = scores > thr
            # SITS tile extent is NOT a water area (tiles are ~0.8% water at the median);
            # SITS locates, pixel-level NDWI quantifies (gated). Kept for reference only.
            sits_detect = float(flagged.sum()) * PATCH_KM2          # flagged-tile extent
            ndwi_full = float(ndwi_flood.sum()) * PX_KM2            # all new-flood water in district
            ndwi_gated = float(ndwi_flood[flagged].sum()) * PX_KM2  # water inside SITS-flagged tiles
            sits_area, ndwi_area = sits_detect, ndwi_full
            optical = True
            s1_clear, s1_cloud = s1split.get(ev, (None, None))
            if method == 'ndwi-calib':      # quality gate passed -> SITS+NDWI fusion
                optical_clear, source = ndwi_gated, 'SITS+NDWI'
                # SITS can also MISS flood tiles -> gating then under-counts. If gating cut the
                # water by >half AND radar independently confirms the larger extent (so the
                # trimmed water is real, not noise), restore the full NDWI. Compare against
                # s1_clear, not whole-district S1: both then cover the same clear ground.
                s1v = s1_clear if s1_clear is not None else (
                    float(s1_area) if (s1_area is not None and pd.notna(s1_area)) else None)
                if (ndwi_full > 0 and ndwi_gated < 0.5 * ndwi_full
                        and s1v is not None and s1v >= ndwi_full):
                    optical_clear, source = ndwi_full, 'SITS+NDWI(restored)'
            else:                           # gate failed -> SITS flags unreliable, use plain NDWI
                optical_clear, source = ndwi_full, 'NDWI'
            # Cloud fusion: optical measured the clear part, S1 measures the cloudy part.
            # The two masks are complementary by construction (s1_cloudy.py splits on the
            # same flood-date S2 valid mask), so the areas add without double-counting.
            if s1_cloud is not None:
                combined, source = optical_clear + s1_cloud, source + '+S1cloud'
            else:
                # no S1 imagery -> the clouded part stays unmeasured; combined is a
                # LOWER BOUND. cloud_pct in the output says how much ground is missing.
                combined = optical_clear
                if (pcloud.get(ev) or 0) > 0:
                    source += '(no-S1)'
        else:
            # 0 SITS patches = flood date too clouded for optical -> trust radar first.
            # (Track A's cloud-median can still report an NDWI, but with SITS blind it is
            #  unreliable, so S1 wins; NDWI is only a last resort when there is no S1.)
            thr, method, sits_area = None, 'no-sits', None
            ndwi_area = float(s2_area) if pd.notna(s2_area) else None   # Track A NDWI, if any
            optical = False
            s1_clear, s1_cloud = s1split.get(ev, (None, None))
            cl = pcloud.get(ev)
            too_cloudy = cl is not None and cl >= NDWI_FALLBACK_MAX_CLOUD_PCT
            if s1_area is not None and pd.notna(s1_area):    # cloud-blind -> radar
                # no optical half to fuse with -> whole-district S1 (= s1_clear + s1_cloudy)
                combined, source = float(s1_area), 'S1(cloud)'
            elif ndwi_area is not None and not too_cloudy:   # no S1, but optical mostly saw it
                combined, source = ndwi_area, 'NDWI(no-S1)'
                optical = optical_ok
            else:
                combined, source = None, 'none'             # no usable measurement at all

        dist_km2 = da.get(ev)
        ratio = (round(combined / dist_km2, 4)
                 if (combined is not None and dist_km2 and dist_km2 > 0) else None)
        rows.append({
            'event_id': ev,
            'sits_detect_km2': round(sits_area, 2) if sits_area is not None else None,  # SITS tile extent (NOT area)
            'sits_threshold': round(thr, 3) if thr is not None else None,
            'sits_method': method,
            'ndwi_flood_km2': round(ndwi_area, 2) if ndwi_area is not None else None,   # full-district NDWI
            's1_flood_km2': round(float(s1_area), 2) if (s1_area is not None and pd.notna(s1_area)) else None,
            's1_clear_km2': round(s1_clear, 2) if s1_clear is not None else None,    # S1 where optical saw
            's1_cloudy_km2': round(s1_cloud, 2) if s1_cloud is not None else None,   # S1 where it didn't
            'optical_clear_km2': (round(optical_clear, 2)                            # optical half of the sum
                                  if has_sits and optical_clear is not None else None),
            'optical_available': optical,
            'cloud_pct': pcloud.get(ev),                    # flood-date cloud over district (QA)
            'combined_km2': round(combined, 2) if combined is not None else None,
            'district_km2': round(float(dist_km2), 1) if dist_km2 else None,
            'flood_ratio': ratio,
            'combined_source': source,
        })
        _s = f"{sits_area:6.1f}" if sits_area is not None else "   -  "
        _c = f"{combined:6.1f}" if combined is not None else "   -  "
        print(f"  {ev}: sits_detect={_s} ({method:9s}) ndwi_full={ndwi_area}  "
              f"s1={s1_area}  -> combined={_c} ({source})")

    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"\nSaved -> {OUT_CSV}  ({len(rows)} events)")


if __name__ == '__main__':
    main()
