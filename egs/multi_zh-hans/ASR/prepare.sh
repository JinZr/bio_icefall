#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=16
stage=-1
stop_stage=100
num_splits=100

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

vocab_sizes=(
  2000
)


# multidataset list.
# LibriSpeech and musan are required.
# The others are optional.
multidataset=(
  "gigaspeech",
  "commonvoice",
  "librilight",
)

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

log "Dataset: musan"
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Soft link fbank of musan"
  mkdir -p data/fbank
  if [ -e ../../librispeech/ASR/data/fbank/.musan.done ]; then
    cd data/fbank
    ln -svf $(realpath ../../../../librispeech/ASR/data/fbank/musan_feats) .
    ln -svf $(realpath ../../../../librispeech/ASR/data/fbank/musan_cuts.jsonl.gz) .
    cd ../..
  else
    log "Abort! Please run ../../librispeech/ASR/prepare.sh --stage 4 --stop-stage 4"
    exit 1
  fi
fi

log "Dataset: THCHS-30"
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare THCHS-30"
  if [ ! -d $dl_dir/thchs30 ]; then
    log "Downloading THCHS-30"
    lhotse download thchs30 $dl_dir/thchs30
  fi

  if [ ! -f data/manifests/.thchs30.done ]; then
    mkdir -p data/manifests
    lhotse prepare thchs-30 $dl_dir/thchs30 data/manifests/thchs30
    touch data/manifests/.thchs30.done
  fi

  if [ ! -f data/fbank/.thchs30.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_thchs30.py
    touch data/fbank/.thchs30.done
  fi
fi

log "Dataset: AISHELL-1"
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare AISHELL-1"
  if [ -e ../../aishell/ASR/data/fbank/.aishell.done ]; then
    cd data/fbank
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_feats_train) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_feats_dev) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_feats_test) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_cuts_train.jsonl.gz) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_cuts_dev.jsonl.gz) .
    ln -svf $(realpath ../../../../aishell/ASR/data/fbank/aishell_cuts_test.jsonl.gz) .
    cd ../..
  else
    log "Abort! Please run ../../aishell/ASR/prepare.sh --stage 3 --stop-stage 3"
    exit 1
  fi
fi

log "Dataset: AISHELL-2"
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Prepare AISHELL-2"
  if [ -e ../../aishell/ASR/data/fbank/.aishell2.done ]; then
    cd data/fbank
    ln -svf $(realpath ../../../../aishell2/ASR/data/fbank/aishell2_feats_train) .
    ln -svf $(realpath ../../../../aishell2/ASR/data/fbank/aishell2_feats_dev) .
    ln -svf $(realpath ../../../../aishell2/ASR/data/fbank/aishell2_feats_test) .
    ln -svf $(realpath ../../../../aishell2/ASR/data/fbank/aishell2_cuts_train.jsonl.gz) .
    ln -svf $(realpath ../../../../aishell2/ASR/data/fbank/aishell2_cuts_dev.jsonl.gz) .
    ln -svf $(realpath ../../../../aishell2/ASR/data/fbank/aishell2_cuts_test.jsonl.gz) .
    cd ../..
  else 
    log "Abort! Please run ../../aishell2/ASR/prepare.sh --stage 3 --stop-stage 3"
    exit 1
  fi 
fi

log "Dataset: AISHELL-4"
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare AISHELL-4"
  if [ -e ../../aishell/ASR/data/fbank/.aishell4.done ]; then
    cd data/fbank
    ln -svf $(realpath ../../../../aishell4/ASR/data/fbank/aishell4_feats_train) .
    ln -svf $(realpath ../../../../aishell4/ASR/data/fbank/aishell4_feats_dev) .
    ln -svf $(realpath ../../../../aishell4/ASR/data/fbank/aishell4_feats_test) .
    ln -svf $(realpath ../../../../aishell4/ASR/data/fbank/aishell4_cuts_train.jsonl.gz) .
    ln -svf $(realpath ../../../../aishell4/ASR/data/fbank/aishell4_cuts_dev.jsonl.gz) .
    ln -svf $(realpath ../../../../aishell4/ASR/data/fbank/aishell4_cuts_test.jsonl.gz) .
    cd ../..
  else 
    log "Abort! Please run ../../aishell4/ASR/prepare.sh --stage 3 --stop-stage 3"
    exit 1
  fi 
fi

log "Dataset: ST-CMDS"
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare ST-CMDS"
  if [ ! -f $dl_dir/stcmds/ST-CMDS-20170001_1-OS.tar.gz ]; then
    log "Downloading ST-CMDS"
    lhotse download stcmds $dl_dir/stcmds
  fi

  if [ ! -f data/manifests/.stcmds.done ]; then
    mkdir -p data/manifests
    lhotse prepare stcmds $dl_dir/stcmds data/manifests/stcmds
    touch data/manifests/.stcmds.done
  fi

  if [ ! -f data/fbank/.stcmds.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_stcmds.py
    touch data/fbank/.stcmds.done
  fi
fi


log "Dataset: Primewords"
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Prepare Primewords"
  if [ ! -f $dl_dir/primewords/primewords_md_2018_set1.tar.gz ]; then
    log "Downloading Primewords"
    lhotse download primewords $dl_dir/primewords
  fi

  if [ ! -f data/manifests/.stcmds.done ]; then
    mkdir -p data/manifests
    lhotse prepare stcmds $dl_dir/primewords data/manifests/primewords
    touch data/manifests/.primewords.done
  fi

  if [ ! -f data/fbank/.primewords.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_primewords.py
    touch data/fbank/.primewords.done
  fi
fi

log "Dataset: MagicData"
if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Prepare MagicData"
  if [ ! -f $dl_dir/magicdata/train_set.tar.gz ]; then
    log "Downloading MagicData"
    lhotse download magicdata $dl_dir/magicdata
  fi

  if [ ! -f data/manifests/.magicdata.done ]; then
    mkdir -p data/manifests
    lhotse prepare magicdata $dl_dir/magicdata data/manifests/magicdata
    touch data/manifests/.magicdata.done
  fi

    if [ ! -f data/fbank/.magicdata.done ]; then
    mkdir -p data/fbank
    ./local/compute_fbank_magicdata.py
    touch data/fbank/.magicdata.done
  fi
fi

log "Dataset: aidatatang_200zh"
if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Prepare aidatatang_200zh"
  if [ -e ../../aidatatang_200zh/ASR/data/fbank/.aidatatang_200zh.done ]; then
    cd data/fbank
    ln -svf $(realpath ../../../../aidatatang_200zh/ASR/data/fbank/aidatatang_feats_train) .
    ln -svf $(realpath ../../../../aidatatang_200zh/ASR/data/fbank/aidatatang_feats_dev) .
    ln -svf $(realpath ../../../../aidatatang_200zh/ASR/data/fbank/aidatatang_feats_test) .
    ln -svf $(realpath ../../../../aidatatang_200zh/ASR/data/fbank/aidatatang_cuts_train.jsonl.gz) .
    ln -svf $(realpath ../../../../aidatatang_200zh/ASR/data/fbank/aidatatang_cuts_dev.jsonl.gz) .
    ln -svf $(realpath ../../../../aidatatang_200zh/ASR/data/fbank/aidatatang_cuts_test.jsonl.gz) .
    cd ../..
  else
    log "Abort! Please run ../../aidatatang_200zh/ASR/prepare.sh --stage 4 --stop-stage 4"
    exit 1
  fi
fi

log "Dataset: Ali-Meeting"
if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  log "Stage 10: Prepare Ali-Meeting"
  if [ -e ../../alimeeting/ASR/data/fbank/.fbank.done ]; then
    cd data/fbank
    ln -svf $(realpath ../../../../alimeeting/ASR/data/fbank/alimeeting-far_feats_train) .
    ln -svf $(realpath ../../../../alimeeting/ASR/data/fbank/alimeeting-far_feats_eval) .
    ln -svf $(realpath ../../../../alimeeting/ASR/data/fbank/alimeeting-far_feats_test) .
    ln -svf $(realpath ../../../../alimeeting/ASR/data/fbank/alimeeting-far_cuts_train.jsonl.gz) .
    ln -svf $(realpath ../../../../alimeeting/ASR/data/fbank/alimeeting-far_cuts_eval.jsonl.gz) .
    ln -svf $(realpath ../../../../alimeeting/ASR/data/fbank/alimeeting-far_cuts_test.jsonl.gz) .
    cd ../..
  else
    log "Abort! Please run ../../alimeeting/ASR/prepare.sh --stage 5 --stop-stage 5"
    exit 1
  fi
fi

log "Dataset: WenetSpeech"
if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
  log "Stage 11: Prepare WenetSpeech"
  if [ -e ../../wenetspeech/ASR/data/fbank/.preprocess_complete ]; then
    cd data/fbank
    ln -svf $(realpath ../../../../wenetspeech/ASR/data/fbank/cuts_DEV.jsonl.gz) .
    ln -svf $(realpath ../../../../wenetspeech/ASR/data/fbank/cuts_DEV_raw.jsonl.gz) .
    ln -svf $(realpath ../../../../wenetspeech/ASR/data/fbank/cuts_L_raw.jsonl.gz) .
    ln -svf $(realpath ../../../../wenetspeech/ASR/data/fbank/cuts_M_raw.jsonl.gz) .
    ln -svf $(realpath ../../../../wenetspeech/ASR/data/fbank/cuts_S_raw.jsonl.gz) .
    ln -svf $(realpath ../../../../wenetspeech/ASR/data/fbank/cuts_TEST_MEETING_raw.jsonl.gz) .
    ln -svf $(realpath ../../../../wenetspeech/ASR/data/fbank/cuts_TEST_NET_raw.jsonl.gz) .
    cd ../..
  else
    log "Abort! Please run ../../wenetspeech/ASR/prepare.sh"
    exit 1
  fi

  if [ -d ../../wenetspeech/ASR/data/lang_char/ ]; then
    cd data
    cp -r ../../../../wenetspeech/ASR/data/lang_char .
    cd ..
  else
    log "Abort! Please run ../../wenetspeech/ASR/prepare.sh"
    exit 1
  fi
fi

log "Dataset: KeSpeech"
if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
  log "Stage 12: Prepare KeSpeech"
  if [ ! -d $dl_dir/KeSpeech ]; then
    log "Abort! Please download KeSpeech first."
  fi

  if [ ! -f data/manifests/.kespeech.done ]; then
    mkdir -p data/manifests
    lhotse prepare kespeech -j $nj $dl_dir/KeSpeech data/manifests/kespeech 
    touch data/manifests/.kespeech.done
  fi

  if [ ! -f data/fbank/.kespeech.done ]; then
    mkdir -p data/fbank

    log "Preprocess KeSpeech manifest"
    if [ ! -f data/fbank/.kespeech_preprocess_complete ]; then
      python3 ./local/preprocess_kespeech.py
      touch data/fbank/.kespeech_preprocess_complete
    fi  
    
    if [ -f data/fbank/.kespeech.train_phase1.split.${num_splits}.done ]; then
      log "Spliting KeSpeech train_phase1"
      lhotse split ${num_splits} \
        data/fbank/kespeech/kespeech-asr_cuts_train_phase1_raw.jsonl.gz \
        data/fbank/kespeech/train_phase1_split_${num_splits}
      touch data/fbank/.kespeech.train_phase1.split.${num_splits}.done
    fi
    
    if [ -f data/fbank/.kespeech.train_phase2.split.${num_splits}.done ]; then
      log "Spliting KeSpeech train_phase2"
      lhotse split ${num_splits} \
        data/fbank/kespeech/kespeech-asr_cuts_train_phase2_raw.jsonl.gz \
        data/fbank/kespeech/train_phase2_split_${num_splits}
      touch data/fbank/.kespeech.train_phase2.split.${num_splits}.done
    fi
    
    log "Compute KeSpeech fbank for train_phase1"
    ./local/compute_fbank_kespeech_splits.py --num-splits ${num_splits} --training-subset train_phase1

    log "Compute KeSpeech fbank for train_phase2"
    ./local/compute_fbank_kespeech_splits.py --num-splits ${num_splits} --training-subset train_phase2

    log "Compute KeSpeech fbank for test/dev"
    ./local/compute_fbank_kespeech_dev_test.py

    touch data/fbank/.kespeech.done
  fi
fi

if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]; then
  log "Stage 13: BPE model training"
  ./local/prepare_for_bpe_model.py --lang-dir ./data/lang_char --text ./data/lang_char/text

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    
    mkdir -p $lang_dir
    ./local/train_bpe_model.py \
      --lang-dir $lang_dir \
      --transcript ./data/lang_char/transcript_chars.txt \
      --vocab-size $vocab_size
  done
  
  ./local/train_bpe_model.py --lang-dir ./data/lang_bpe_${vocab_size}
fi