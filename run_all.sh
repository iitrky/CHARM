#!/usr/bin/env bash
# Full experimental pipeline for the CHARM manuscript.
# Usage: bash run_all.sh /path/ptb-xl-1.0.3 /path/chapman_shaoxing /path/cpsc_2018
set -e
PTB=$1; CHAP=$2; CPSC=$3; R=results
for s in 0 1 2 3 4; do
  # Table 3-6: proposed model
  python train_charm.py --root $PTB --seed $s --out $R
  # Table 3-6: baselines
  for m in xresnet inception cnn_lstm_att transformer ft_transformer early_fusion late_fusion medfuse mc_dropout dagat hyperfusion; do
    python baselines.py --root $PTB --model $m --seed $s --out $R
  done
  python baselines.py --root $PTB --model late_fusion --temp_scale --seed $s --out $R
  # Table 7: ablations (one component removed at a time)
  python train_charm.py --root $PTB --seed $s --out $R --no_film      --tag abl_nofilm
  python train_charm.py --root $PTB --seed $s --out $R --static_film  --tag abl_staticfilm
  python train_charm.py --root $PTB --seed $s --out $R --no_prior     --tag abl_noprior
  python train_charm.py --root $PTB --seed $s --out $R --lam_imp 0    --tag abl_noimp
  python train_charm.py --root $PTB --seed $s --out $R --lam_cons 0   --tag abl_nocons
  python train_charm.py --root $PTB --seed $s --out $R --omega 1 --lam_proto 0 --tag abl_noproto
  python train_charm.py --root $PTB --seed $s --out $R --bce_loss     --tag abl_bce
  python train_charm.py --root $PTB --seed $s --out $R --no_aug       --tag abl_noaug
  # Table 6: external validation
  if [ -n "$CHAP" ]; then python external_eval.py --ext_dir $CHAP --name chapman --ckpt $R/charm_seed${s}_final.pt --out $R; fi
  if [ -n "$CPSC" ]; then python external_eval.py --ext_dir $CPSC --name cpsc    --ckpt $R/charm_seed${s}_final.pt --out $R; fi
  for m in xresnet late_fusion dagat hyperfusion; do
    if [ -n "$CHAP" ]; then python external_eval.py --ext_dir $CHAP --name chapman --model $m --ckpt $R/${m}_seed${s}_final.pt --out $R; fi
    if [ -n "$CPSC" ]; then python external_eval.py --ext_dir $CPSC --name cpsc    --model $m --ckpt $R/${m}_seed${s}_final.pt --out $R; fi
  done
done
python aggregate.py --results $R --ensemble late_fusion     # -> results/ALL_TABLES.md + CSVs
python figures.py --results $R --root $PTB --ckpt $R/charm_seed0_final.pt
echo "Done. Copy results/ALL_TABLES.md values into the [fill] cells of the manuscript."
