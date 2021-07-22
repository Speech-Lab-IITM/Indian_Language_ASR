#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch

stage=1       # start from -1 if you need to start from data download or start from the stage you want to run 

####README ON VARIOUS STAGE####
#stage=1 Feature extraction + cmvn.ark file creation + dumping the features in the dump directory mentioned
#stage=2 CHAR or BPE token generation and dumping it inside the lang_${nbpe} folder + data.json file creation and dumping it in the dump directory mentioned
#stage=3 Training stage
#stage=4 Decoding stage

stop_stage=100 # Basically how many epochs to run for training
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=0
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume= # Path to the snapshot from which you want to resume

# feature configuration
do_delta=false
preprocess_config=conf/specaug.yaml

# Training configurtaion
train_config=conf/train_pytorch_transformer.yaml

# LM configuration
lm_config=conf/lm.yaml

# Decode configuration
decode_config=conf/decode.yaml

# LM related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=5               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# bpemode (unigram or bpe)
nbpe=1000  # number of bpe
bpemode=unigram  # unigram or bigram etc.,

# exp tag   
tag="can_be_anything" # tag for differentiating multiple experiments.

#TO EDIT WHEN RUNNING A NEW EXPERIMENT        
datadir=/Path/to/the/data/folder/
exp=/Path/to/the/exp/folder/ 

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_Hindi  # Train folder name; example: train_Hindi, train_Tamil, train_English
train_dev=dev_Hindi  # Dev folder name; example:dev_Hindi, dev_Tamil, dev_English

recog_set="dev_hindi eval_hindi"  # Test folder name; example: dev_Hindi eval_Hindi dev_Tamil eval_Tamil dev_English eval_English

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

####RUN STAGE 1 BY LOGGING INTO A CPU####
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=${dumpdir}/fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    
    for x in ${train_set}_org ${train_dev}_org; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 16 --write_utt2num_frames true \
            $datadir/${x} $exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh $datadir/${x}
    done
    
    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 $datadir/${train_set}_org $datadir/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 $datadir/${train_dev}_org $datadir/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:$datadir/${train_set}/feats.scp $datadir/${train_set}/cmvn.ark

    dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
        $datadir/${train_set}/feats.scp $datadir/${train_set}/cmvn.ark $exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
        $datadir/${train_dev}/feats.scp $datadir/${train_set}/cmvn.ark $exp/dump_feats/dev ${feat_dt_dir}


# Block to be executed when decoding $recog_set aka test set    
<<"over"  
    for x in $recog_set; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true \
            $datadir/${x} $exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh $datadir/${x}
    done

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
            $datadir/${rtask}/feats.scp $datadir/${train_set}/cmvn.ark $exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
over

fi


###RUN STAGE 2 BY LOGGING INTO A CPU###
dict=$datadir/lang_${nbpe}/train_${bpemode}${nbpe}_units.txt
bpemodel=$datadir/lang_${nbpe}/train_${bpemode}${nbpe}
nlsyms=$datadir/lang_${nbpe}/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p $datadir/lang_${nbpe}/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " $datadir/${train_set}/text > $datadir/lang_${nbpe}/input.txt
    spm_train --input=$datadir/lang_${nbpe}/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < $datadir/lang_${nbpe}/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model --nlsyms ${nlsyms}  \
        $datadir/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model  --nlsyms ${nlsyms} \
        $datadir/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json


# Block to be executed when decoding $recog_set aka test set
<<"over"
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            $datadir/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
over

fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=$exp/${expname}
mkdir -p ${expdir}


####RUN STAGE 3 IN A GPU####
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

####RUN DECODE IN CPU####
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"

    recog_model=model.val5.avg.best

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then

        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}


# Block to be executed only when averaging LM models
<<"over"
        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${lm_n_average}.avg.best
            opt="--log ${lmexpdir}/log"
        else
            lang_model=rnnlm.last${lm_n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
over

    fi

    nj=16

  # pids=() # initialize pids
    for rtask in ${recog_set}; do
  # (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        # split data

        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0
     
        mkdir -p ${expdir}/${decode_dir}/log

	# set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

  # ) &
 #  pids+=($!) # store background pids
    done
 #  i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
 #  [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
