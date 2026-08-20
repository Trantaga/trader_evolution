"""
generate_report.py -- builds a self-contained, offline-openable per-run HTML report by
injecting this run's data (compact telemetry + checkpoint + held-out dump + full-history
dump, whichever of the latter three exist yet) into tools/dashboard_template.html's
embedded-data hydration point (see that file's `tryEmbedded()` -- this script writes
exactly the JSON shape it expects: {run_id, compact, checkpoint, heldout, fullhistory}).
The template itself is read verbatim and never modified -- this script only injects one
<script> block into a copy of it. NOTE: as of this writing the template's own JS doesn't
yet render a "Full history" tab from the `fullhistory` field -- it's embedded here (null
until the sealed held-out step produces it) so the data is already present in every
report the moment that UI is added, without needing every existing report regenerated.

Pure output: reads runs/<run_id>/{compact.jsonl,checkpoint.json,heldout_dump.json,
fullhistory_dump.json} (the LATTER THREE optional -- checkpoint.json exists from gen 0
on, heldout_dump.json/fullhistory_dump.json only after the run has terminated and the
sealed held-out step has run) plus the unmodified template, and writes
runs/<run_id>/report.html. Never reads its own output back, never imports
evolution.py/fitness.py/champion_selection.py -- cannot feed back into any of them.

CRITICAL escaping: the embedded JSON sits inside a <script type="application/json">
block. A literal "</script>" anywhere in the DATA (e.g. inside a genome label or a stop-
reason string) would prematurely close that block and corrupt the page -- the remainder
renders as visible text and the data never loads. json.dumps()'s default output can
legally contain "<" (it is not JSON-escaped by default), so every "<" is replaced with
its unicode escape "\\u003c" AFTER dumping -- with no literal "<" left in the payload, no
"</script>" substring can appear either, regardless of what's in the data. This exact bug
happened during template development -- it is a real, not hypothetical, risk.
"""
import argparse
import json
import os

# The template's SINGLE bare main <script> tag -- verified unique (grep -c '<script'
# tools/dashboard_template.html == 1) at the time this was written. If the template ever
# grows a second bare <script> tag, the count check in build_report_html() below refuses
# to guess and raises instead of silently injecting at the wrong spot.
EMBED_ANCHOR = "<script>"
EMBED_ID_OPEN = '<script id="embedded-data" type="application/json">'


def load_compact_rows(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json_or_none(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_report_html(run_id, template_text, compact_rows, checkpoint_obj, heldout_obj,
                       fullhistory_obj=None):
    embedded = {
        "run_id": run_id,
        "compact": compact_rows,
        "checkpoint": checkpoint_obj,
        "heldout": heldout_obj,
        "fullhistory": fullhistory_obj,
    }
    # See module docstring: json.dumps does NOT escape "<" by default -- do it explicitly.
    payload = json.dumps(embedded).replace("<", "\\u003c")
    embed_block = f"{EMBED_ID_OPEN}{payload}</script>\n"

    n_anchor = template_text.count(EMBED_ANCHOR)
    if n_anchor != 1:
        raise RuntimeError(
            f"expected exactly one bare {EMBED_ANCHOR!r} in the template, found "
            f"{n_anchor} -- refusing to guess which one is the main script tag (the "
            f"template may have changed; this script must be kept in sync with it).")
    html = template_text.replace(EMBED_ANCHOR, embed_block + EMBED_ANCHOR, 1)

    _sanity_check(html)
    return html


def _sanity_check(html):
    """Fail LOUDLY (not silently) if the injected report is malformed -- see module
    docstring's CRITICAL escaping note for the exact failure mode this guards against."""
    n_open = html.count("<script")
    n_close = html.count("</script>")
    if n_open != n_close:
        raise RuntimeError(
            f"report.html sanity check failed: {n_open} '<script' tag(s) vs {n_close} "
            f"'</script>' tag(s) -- the embedded JSON almost certainly contains an "
            f"unescaped '<' that broke out of its <script> block. Refusing to write a "
            f"broken report.")
    if EMBED_ID_OPEN not in html:
        raise RuntimeError("report.html sanity check failed: embedded-data <script> tag "
                            "not found in the output -- injection did not happen.")
    start = html.index(EMBED_ID_OPEN) + len(EMBED_ID_OPEN)
    end = html.index("</script>", start)
    try:
        json.loads(html[start:end])
    except json.JSONDecodeError as e:
        raise RuntimeError(f"report.html sanity check failed: embedded JSON does not "
                            f"re-parse ({e}). Refusing to write a broken report.") from e


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_id")
    p.add_argument("runs_root",
                    help="directory containing <run_id>/{compact.jsonl,checkpoint.json,"
                         "heldout_dump.json,fullhistory_dump.json}")
    p.add_argument("template", help="path to tools/dashboard_template.html")
    args = p.parse_args()

    run_dir = os.path.join(args.runs_root, args.run_id)
    compact_rows = load_compact_rows(os.path.join(run_dir, "compact.jsonl"))
    checkpoint_obj = load_json_or_none(os.path.join(run_dir, "checkpoint.json"))
    heldout_obj = load_json_or_none(os.path.join(run_dir, "heldout_dump.json"))
    fullhistory_obj = load_json_or_none(os.path.join(run_dir, "fullhistory_dump.json"))

    with open(args.template) as f:
        template_text = f.read()

    html = build_report_html(args.run_id, template_text, compact_rows, checkpoint_obj,
                              heldout_obj, fullhistory_obj)

    out_path = os.path.join(run_dir, "report.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote {out_path} ({len(html)} bytes; {len(compact_rows)} compact rows, "
          f"checkpoint={'yes' if checkpoint_obj is not None else 'no'}, "
          f"heldout={'yes' if heldout_obj is not None else 'no'}, "
          f"fullhistory={'yes' if fullhistory_obj is not None else 'no'})")


if __name__ == "__main__":
    main()
