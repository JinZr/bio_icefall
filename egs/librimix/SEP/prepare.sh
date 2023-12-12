#!/usr/bin/env bash

set -eou pipefail

stage=-1
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/LibriSpeech
#      You can find audio and transcripts for LibriSpeech in this path.
#
#  - $dl_dir/wham_noise
#      This directory contains the following directories downloaded from
#       https://storage.googleapis.com/whisper-public/wham_noise.zip
#     
#     - cv
#     - tr
#     - tt
#     - metadata
# 
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech
#

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data
vocab_sizes=(
    # 5000
    # 2000
    # 1000
    500
)

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Stage 0: Download data"

    # If you have pre-downloaded it to /path/to/librispeech,
    # you can create a symlink
    #
    #   ln -sfv /path/to/librispeech $dl_dir/librispeech
    #
    if [ ! -d $dl_dir/librispeech ]; then
        lhotse download librispeech --full $dl_dir
    fi

    if [ ! -d LibriMix ]; then
        git clone https://github.com/JorisCos/LibriMix LibriMix
        log "We assume your python env fulfills the requirements of LibriMix"
    fi

    # If you have pre-downloaded it to /path/to/musan,
    # you can create a symlink
    #
    #   ln -sfv /path/to/musan $dl_dir/
    #
    if [ ! -d $dl_dir/musan ]; then
        lhotse download musan $dl_dir
    fi

fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Generating LibriMix mixtures. This may take a while."
    # We assume that you have downloaded the LibriSpeech corpus
    # to $dl_dir/LibriSpeech. 
    mkdir -p data/manifests
    offset_dir=$(pwd)/local/metadta_Libri2Mix_offset/
    cd LibriMix

    if [ ! -d metadata/Libri2Mix_offset ]; then
        ln -s ${offset_dir} metadata/Libri2Mix_offset
    fi
    if [ ! -f .patch.complete ]; then
        patch scripts/create_librimix_from_metadata.py \
            -i ../local/create_librimix_from_metadata.patch
        touch .patch.complete
    else
        log "already patched the script, skipping"
    fi

    python scripts/augment_train_noise.py --wham_dir ../download/wham_noise
    # bash generate_librimix.sh $dl_dir
        # simulate 2mix data with a random overlap offset about 1-1.5s
    # only support 2mix data currently
    for n_src in 2;
    do
        metadata_dir=metadata/Libri${n_src}"Mix"_offset
        python scripts/create_librimix_from_metadata.py \
            --librispeech_dir ${dl_dir}/LibriSpeech \
            --wham_dir ${dl_dir}/wham_noise \
            --metadata_dir ${metadata_dir} \
            --librimix_outdir ${dl_dir} \
            --n_src ${n_src} \
            --freqs "8k" \
            --modes "min" \
            --types mix_clean mix_both mix_single
     done

    cd ..
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Prepare LibriMix manifests"
    # We assume that you have downloaded the LibriSpeech corpus
    # to $dl_dir/LibriSpeech and performed generate_librimix.sh
    manifest_dir=data/manifests
    mkdir -p ${manifest_dir}
    for n_src in 2; do
      for set in train-100 train-360 dev test; do
        echo "Preparing manifest for num of speakers: ${n_src}, ${set}"
        
        lhotse prepare librimix \
            --sampling-rate 16000 \
            --with-precomputed-mixtures \
            ${dl_dir}/Libri${n_src}Mix/wav16k/max/metadata/mixture_${set}_mix_both.csv \
            ${manifest_dir}

        mv ${manifest_dir}/librimix_recordings_mix.jsonl.gz \
            ${manifest_dir}/librimix_${n_src}mix_${set}_recordings_mix_both.jsonl.gz
        mv ${manifest_dir}/librimix_recordings_noise.jsonl.gz \
            ${manifest_dir}/librimix_${n_src}mix_${set}_recordings_noise_both.jsonl.gz
        mv ${manifest_dir}/librimix_recordings_sources.jsonl.gz \
            ${manifest_dir}/librimix_${n_src}mix_${set}_recordings_sources_both.jsonl.gz

        mv ${manifest_dir}/librimix_supervisions_mix.jsonl.gz \
            ${manifest_dir}/librimix_${n_src}mix_${set}_supervisions_mix_both.jsonl.gz
        mv ${manifest_dir}/librimix_supervisions_sources.jsonl.gz \
            ${manifest_dir}/librimix_${n_src}mix_${set}_supervisions_sources_both.jsonl.gz
      done
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: Prepare musan manifest"
    # We assume that you have downloaded the musan corpus
    # to $dl_dir/musan
    mkdir -p data/manifests
    lhotse prepare musan $dl_dir/musan data/manifests 
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Compute fbank features for musan"
    mkdir -p data/fbank
    python local/compute_fbank_musan.py
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Prepare LibriMix manifest"

    if [ -e ../../librispeech/ASR/data/manifests/.librispeech.done ]; then
        cd data/manifests
        ln -svf $(realpath ../../../../librispeech/ASR/data/manifests/librispeech_supervisions_train-clean-100.jsonl.gz) .
        ln -svf $(realpath ../../../../librispeech/ASR/data/manifests/librispeech_supervisions_train-clean-360.jsonl.gz) .
        ln -svf $(realpath ../../../../librispeech/ASR/data/manifests/librispeech_supervisions_dev-clean.jsonl.gz) .
        ln -svf $(realpath ../../../../librispeech/ASR/data/manifests/librispeech_supervisions_test-clean.jsonl.gz) .

        cd ../..
    else
        log "Abort! Please run ../../librispeech/ASR/prepare.sh --stage 1 --stop-stage 1 first"
        exit 1
    fi

    # for n_src in 2; do
    #     for part in train-100 train-360 dev test; do
    #         log "Preparing Libri${n_src}Mix/wav16k/max/metadata/mixture_${part}_mix_both.csv."
    #         lhotse prepare librimix --with-precomputed-mixtures \
    #             $dl_dir/Libri${n_src}Mix/wav16k/max/metadata/mixture_${part}_mix_both.csv \
    #             data/manifests
    #         mv data/manifests/librimix_recordings_mix.jsonl.gz \
    #             data/manifests/librimix_${n_src}mix_${part}_recordings_mix_both.jsonl.gz
    #         mv data/manifests/librimix_supervisions_mix.jsonl.gz \
    #             data/manifests/librimix_${n_src}mix_${part}_supervisions_mix_both.jsonl.gz
    #     done
    # done

    touch data/manifests/.librimix.done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Stage 6: Compute fbank features for LibriMix"
    mkdir -p data/fbank

    if [ ! -f "./data/fbank/.librimix.done" ]; then
        ./local/compute_fbank_librimix.py

        if [ ! -f data/fbank/librimix_2mix_cuts_train-all-shuf.jsonl.gz ]; then
            cat <(gunzip -c data/fbank/librimix_2mix_cuts_train-100.jsonl.gz) \
            <(gunzip -c data/fbank/librimix_2mix_cuts_train-360.jsonl.gz) | \
            shuf | gzip -c > data/fbank/librimix_2mix_cuts_train-all-shuf.jsonl.gz
        fi
        gunzip -c data/fbank/librimix_2mix_cuts_train-all-shuf.jsonl.gz | \
        jq '.supervisions[0].text' | sed 's/"//g' > data/fbank/text
        touch ./data/fbank/.librimix.done
    fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    log "Stage 7: Prepare phone based lang"

    if [ -d "../../librispeech/ASR/data/lang_phone" ]; then
        cd data/
        log "Copying phone based lang from ../../librispeech/ASR/data/lang_phone due to the upcomming overwrite"
        if [ -d "./lang_phone" ]; then
            cp -r $(realpath ../../../librispeech/ASR/data/lang_phone) .
        fi
        cd ..
    else
        log "Abort! Please run ../../librispeech/ASR/prepare.sh --stage 5 --stop-stage 5 first"
        exit 1
    fi

    if [ -d "../../librispeech/ASR/download/lm" ]; then
        mkdir -p download
        cd download
        if [ -f "./lm" ]; then
            ln -svf $(realpath ../../../librispeech/ASR/download/lm) .
        fi
        cd ..
    else
        log "Abort! Please run ../../librispeech/ASR/local/download_lm.py --out-dir ../../librispeech/ASR/download/lm first"
        exit 1
    fi

    lang_dir=data/lang_phone
    (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; echo '<sc> <sc>';) |
    cat - $dl_dir/lm/librispeech-lexicon.txt |
    sort | uniq > $lang_dir/lexicon.txt

    ./local/prepare_lang.py --lang-dir $lang_dir

    log "Converting L.pt to L.fst"
    ./shared/convert-k2-to-openfst.py \
      --olabels aux_labels \
      $lang_dir/L.pt \
      $lang_dir/L.fst
    
    log "Converting L_disambig.pt to L_disambig.fst"
    ./shared/convert-k2-to-openfst.py \
      --olabels aux_labels \
      $lang_dir/L_disambig.pt \
      $lang_dir/L_disambig.fst
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    log "Stage 8: Prepare BPE based lang"

    for vocab_size in ${vocab_sizes[@]}; do
        lang_dir=data/lang_bpe_${vocab_size}
        mkdir -p $lang_dir
        # We reuse words.txt from phone based lexicon
        # so that the two can share G.pt later.
        cp data/lang_phone/words.txt $lang_dir

        if [ ! -f $lang_dir/transcript_words.txt ]; then
            gunzip -c data/fbank/librimix_2mix_cuts_train-all-shuf.jsonl.gz \
                | jq '.supervisions[0].text' \
                | sed 's/"//g' > $lang_dir/transcript_words.txt
        fi

        if [ ! -f $lang_dir/bpe.model ]; then
            ./local/train_bpe_model.py \
                --lang-dir $lang_dir \
                --vocab-size $vocab_size \
                --transcript $lang_dir/transcript_words.txt
        fi

        if [ ! -f $lang_dir/L_disambig.pt ]; then
            ./local/prepare_lang_bpe.py --lang-dir $lang_dir

            log "Validating $lang_dir/lexicon.txt"
            ./local/validate_bpe_lexicon.py \
                --lexicon $lang_dir/lexicon.txt \
                --bpe-model $lang_dir/bpe.model
        fi

        if [ ! -f $lang_dir/L.fst ]; then
            log "Converting L.pt to L.fst"
            ./shared/convert-k2-to-openfst.py \
                --olabels aux_labels \
                $lang_dir/L.pt \
                $lang_dir/L.fst
        fi

        if [ ! -f $lang_dir/L_disambig.fst ]; then
            log "Converting L_disambig.pt to L_disambig.fst"
            ./shared/convert-k2-to-openfst.py \
                --olabels aux_labels \
                $lang_dir/L_disambig.pt \
                $lang_dir/L_disambig.fst
        fi
    done
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Prepare G"
  # We assume you have install kaldilm, if not, please install
  # it using: pip install kaldilm

  mkdir -p data/lm
  if [ ! -f ./data/lm/G_3_gram.arpa ]; then
    ./shared/make_kn_lm.py \
        -ngram-order 3 \
        -text ./data/fbank/text \
        -lm ./data/lm/G_3_gram.arpa
  fi

  if [ ! -f ./data/lm/G_4_gram.arpa ]; then
    ./shared/make_kn_lm.py \
        -ngram-order 4 \
        -text ./data/fbank/text \
        -lm ./data/lm/G_4_gram.arpa
  fi

  if [ ! -f data/lm/G_3_gram.fst.txt ]; then
    # It is used in building HLG
    python3 -m kaldilm \
      --read-symbol-table="data/lang_bpe_500/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      ./data/lm/G_3_gram.arpa > data/lm/G_3_gram.fst.txt
  fi

  if [ ! -f data/lm/G_4_gram.fst.txt ]; then
    # It is used for LM rescoring
    python3 -m kaldilm \
      --read-symbol-table="data/lang_bpe_500/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      ./data/lm/G_4_gram.arpa > data/lm/G_4_gram.fst.txt
  fi
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  log "Stage 10: Compile HLG"

  # Note If ./local/compile_hlg.py throws OOM,
  # please switch to the following command
  #
  # ./local/compile_hlg_using_openfst.py --lang-dir data/lang_phone

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    ./local/compile_hlg.py --lang-dir $lang_dir

    # Note If ./local/compile_hlg.py throws OOM,
    # please switch to the following command
    #
    # ./local/compile_hlg_using_openfst.py --lang-dir $lang_dir
  done
fi