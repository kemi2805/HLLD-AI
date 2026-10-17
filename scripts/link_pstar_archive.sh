#!/bin/bash
# Put the ARCHIVED p*-surrogate files back where the restored stack looks for
# them -- as SYMLINKS.  Nothing is copied or moved: the archive
# (data/ARCHIVE_pstar_surrogate, see its README) stays the only copy of the
# 1.2 GB dataset and of the checkpoints, none of which were ever in git.
#
#   bash scripts/link_pstar_archive.sh                 # this checkout
#   ARCHIVE=/Users/miler/Codes/HLLD/data/ARCHIVE_pstar_surrogate \
#       bash scripts/link_pstar_archive.sh             # e.g. from a worktree
#
# Links made (paths relative to the checkout root; data/ and checkpoints/ are
# gitignored, so none of this can be committed by accident):
#   checkpoints/default-mignone*.pt
#   data/raw/dataset_mignone_rankine.npz, data/raw/dataset_mignone.npz
#   data/splits_mignone_rankine/            (the rankine split, a directory link)
#   data/splits/default-mignone/            (the older 19-feature split)
#   data/processed/{splits_mignone_rankine,default-mignone}/norm_stats.npz
#   data/rotor_envelope_256.npz, data/ot_envelope_128.npz   (when ENVDIR has them)
# An existing regular file at a link's place is left alone and reported.
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
ARCHIVE=${ARCHIVE:-$REPO/data/ARCHIVE_pstar_surrogate}
ENVDIR=${ENVDIR:-$(dirname "$ARCHIVE")}
[ -d "$ARCHIVE/checkpoints" ] && [ -d "$ARCHIVE/data" ] ||
    { echo "no archive at $ARCHIVE (set ARCHIVE=...)"; exit 2; }

link() {   # link <target> <link path>
    local tgt=$1 dst=$REPO/$2
    [ -e "$tgt" ] || { echo "  missing in archive: $tgt"; return; }
    mkdir -p "$(dirname "$dst")"
    if [ -L "$dst" ]; then
        ln -sfn "$tgt" "$dst"
    elif [ -e "$dst" ]; then
        echo "  NOT linked, a real file is in the way: $dst"; return
    else
        ln -s "$tgt" "$dst"
    fi
    echo "  $2 -> $tgt"
}

echo "== linking $ARCHIVE into $REPO"
for f in "$ARCHIVE"/checkpoints/*.pt; do
    link "$f" "checkpoints/$(basename "$f")"
done
link "$ARCHIVE/data/raw/dataset_mignone_rankine.npz" data/raw/dataset_mignone_rankine.npz
link "$ARCHIVE/data/raw/dataset_mignone.npz" data/raw/dataset_mignone.npz
link "$ARCHIVE/data/splits_mignone_rankine" data/splits_mignone_rankine
link "$ARCHIVE/data/splits/default-mignone" data/splits/default-mignone
link "$ARCHIVE/data/processed/splits_mignone_rankine/norm_stats.npz" \
     data/processed/splits_mignone_rankine/norm_stats.npz
link "$ARCHIVE/data/processed/default-mignone/norm_stats.npz" \
     data/processed/default-mignone/norm_stats.npz
for f in rotor_envelope_256.npz ot_envelope_128.npz; do
    if [ -e "$ENVDIR/$f" ] && [ "$ENVDIR/$f" != "$REPO/data/$f" ]; then
        link "$ENVDIR/$f" "data/$f"
    fi
done
