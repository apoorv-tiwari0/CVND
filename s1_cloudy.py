"""s1_cloudy.py — S1 flood area measured ONLY where the flood-date S2 was cloudy.

For each event, splits the district into:
  clear part  = where the flood-date S2 composite has valid data (optical saw it)
  cloudy part = where it doesn't (optical blind)
and measures the Track-A S1 flood area separately over each part, so merge can fuse:
  combined = optical (SITS+NDWI over the clear part) + s1_cloudy_km2

Exactness: reuses Track A's own Otsu threshold (flood_extent.otsu_threshold_db) and the
identical flood mask / reduceRegion(scale=30, NO bestEffort -- bestEffort silently
coarsens composites and inflated areas 2-3x). So s1_clear + s1_cloudy must equal
Track A's area_s1_km2; the script prints that check per event.

Speed: 1 GEE call per event (single two-band reduceRegion; threshold and cloud fraction
are reused from flood_extent.csv / post_cloud.csv), 4 events in parallel.

Output: data/s1_cloudy.csv
  (event_id, otsu_db, clear_pct, s1_clear_km2, s1_cloudy_km2, s1_total_km2, tracka_s1_km2)

Run: python s1_cloudy.py            # all events (resumes)
     python s1_cloudy.py E01 E08    # subset
"""
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import ee

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import satellite as sat   # importing runs initialize_gee()

OUT = 'data/s1_cloudy.csv'
WORKERS = 4

events = pd.read_csv('data/events.csv')
ta = pd.read_csv('data/flood_extent.csv').set_index('event_id')
pc = pd.read_csv('data/post_cloud.csv').set_index('event_id')
if sys.argv[1:]:
    events = events[events['event_id'].isin(sys.argv[1:])]

done = {}
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    done = {r['event_id']: dict(r) for _, r in prev.iterrows()}
    print(f"resume: {len(done)} events already in {OUT}")

lock = threading.Lock()
rows = list(done.values())


def save():
    pd.DataFrame(rows).to_csv(OUT, index=False)


def measure(row):
    ev = row['event_id']
    t = ta.loc[ev] if ev in ta.index else None
    # no S1 imagery (known from Track A) -> nothing to measure, no GEE call needed
    if t is None or t['baseline_status'] == 'SKIPPED_NO_S1_IMAGERY' or pd.isna(t['otsu_threshold_db']):
        return {'event_id': ev, 'otsu_db': None, 'clear_pct': None,
                's1_clear_km2': None, 's1_cloudy_km2': None,
                's1_total_km2': None, 'tracka_s1_km2': None}

    thr = float(t['otsu_threshold_db'])          # Track A's own threshold -> exact consistency
    region = sat.get_region(row)
    start = ee.Date(row['start_date'])

    # flood-date S2 valid mask (same composite as post_cloud.py / Track B t5);
    # post_images from post_cloud.csv avoids an extra size() round-trip
    if ev in pc.index and int(pc.loc[ev, 'post_images']) > 0:
        s2 = sat._get_s2_sits(region)
        valid = (s2.filterDate(start, start.advance(14, 'day'))
                 .median().mask().reduce(ee.Reducer.min()).unmask(0))
    else:
        valid = ee.Image.constant(0)             # no optical at all -> everything "cloudy"
    clear = valid.gte(0.5)

    # Track-A S1 flood detection, byte-for-byte the same steps
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
          .filterBounds(region)
          .filterDate(start, start.advance(7, 'day')))
    post_f = s1.select('VV').median().focal_median(1, 'square')
    water = post_f.lt(thr)
    permanent, flat, india = sat.get_masks(region)
    flood = (water.And(permanent.Not()).And(flat)
             .clipToCollection(india).rename('flood'))
    px = flood.multiply(ee.Image.pixelArea())

    # ONE reduceRegion for both parts (two bands, same reducer/scale as Track A).
    # Split by MULTIPLYING with the 0/1 indicator, NOT updateMask: updateMask REPLACES
    # fractional edge weights with 1.0 and inflated sums 2-3x (verified on E01/E08);
    # multiply keeps the original mask/weights, so clear+cloudy == Track A total exactly.
    both = (px.multiply(clear).rename('clear')
            .addBands(px.multiply(clear.Not()).rename('cloudy')))
    v = both.reduceRegion(reducer=ee.Reducer.sum(), geometry=region,
                          scale=30, maxPixels=1e9).getInfo()
    a_clear = round((v.get('clear') or 0) / 1e6, 2)
    a_cloudy = round((v.get('cloudy') or 0) / 1e6, 2)

    return {'event_id': ev, 'otsu_db': round(thr, 2),
            'clear_pct': round(100 - float(pc.loc[ev, 'cloud_pct']), 1) if ev in pc.index else None,
            's1_clear_km2': a_clear, 's1_cloudy_km2': a_cloudy,
            's1_total_km2': round(a_clear + a_cloudy, 2),
            'tracka_s1_km2': float(t['area_s1_km2']) if pd.notna(t['area_s1_km2']) else None}


todo = [row for _, row in events.iterrows() if row['event_id'] not in done]
print(f"todo: {len(todo)} events, {WORKERS} workers")

with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = {ex.submit(measure, row): row['event_id'] for row in todo}
    for fut in as_completed(futs):
        ev = futs[fut]
        try:
            rec = fut.result()
        except Exception as e:
            print(f"{ev}: ERROR {e}")
            continue
        with lock:
            rows.append(rec)
            save()
        if rec['s1_total_km2'] is None:
            print(f"{ev}: no S1 imagery -> skip")
        else:
            ref = rec['tracka_s1_km2']
            ratio = (rec['s1_total_km2'] / ref) if ref else None
            chk = f"check={ratio:.2f}x" if ratio else "check=n/a"
            print(f"{ev}: clear={rec['clear_pct']}%  S1 clear/cloudy="
                  f"{rec['s1_clear_km2']}/{rec['s1_cloudy_km2']}  "
                  f"total={rec['s1_total_km2']} vs trackA={ref}  {chk}")

save()
print(f"\nSaved -> {OUT} ({len(rows)} events)")
print("check=1.00x everywhere means the split exactly reproduces Track A totals.")
