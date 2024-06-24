# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import torchaudio
from tqdm import tqdm
from glob import glob
from collections import defaultdict

from utils.util import has_existed
from utils.duration import verify_alignment
from preprocessors.ljspeech import textgird_extract


def prepare_align(dataset, dataset_path, cfg, output_path, jobs=8):
    libritts_path = dataset_path
    distribution2speakers2pharases2utts, unique_speakers = libritts_statistics(
        libritts_path
    )
    corpus_path = os.path.join(output_path, dataset, cfg.raw_data)
    textgrid_directory=os.path.join(output_path, dataset, "TextGrid")
    os.makedirs(corpus_path, exist_ok=True)
    os.makedirs(textgrid_directory, exist_ok=True)
    for distribution, speakers2pharases2utts in tqdm(
        distribution2speakers2pharases2utts.items()
    ):
        # print(f'distribution: {distribution}, speakers2pharases2utts: {speakers2pharases2utts}')
            # print('distribution2speakers2pharases2utts: {}'.format(distribution2speakers2pharases2utts))
        for speaker, pharases2utts in tqdm(speakers2pharases2utts.items()):
            # print(f'speaker: {speaker}, pharases2utts: {pharases2utts}')
            pharase_names = list(pharases2utts.keys())
            for chosen_pharase in pharase_names:
                for chosen_uid in pharases2utts[chosen_pharase]:
                    speaker_corpus_dir = os.path.join(corpus_path, speaker)
                    if not os.path.isdir(speaker_corpus_dir):
                        os.makedirs(speaker_corpus_dir)
                    wav_filename = chosen_uid + '.wav'
                    wav_path = f"{dataset_path}/{distribution}/{speaker}/{chosen_pharase}/{wav_filename}"
                    corpus_wav_path = os.path.join(speaker_corpus_dir, wav_filename)
                    text_filename = chosen_uid +'.normalized.txt'
                    text_path = f"{dataset_path}/{distribution}/{speaker}/{chosen_pharase}/{text_filename}"
                    corpus_text_path = os.path.join(speaker_corpus_dir, chosen_uid + '.lab')
                    if not os.path.isfile(corpus_wav_path):
                        os.symlink(wav_path, corpus_wav_path) # The MFA will default convert sampling rate to 16000 itself.
                    if not os.path.isfile(corpus_text_path):
                        os.symlink(text_path, corpus_text_path)
    lexicon_path = os.path.join(os.environ['WORK_DIR'], cfg.lexicon_path)
    mfa_path=os.path.join(
        os.environ['WORK_DIR'], "pretrained", "mfa", "montreal-forced-aligner", "bin", "mfa_train_and_align"
    )
    mfa_validate_path = os.path.join(
        os.environ['WORK_DIR'], "pretrained", "mfa", "montreal-forced-aligner", "bin", "mfa_validate_dataset"
    )
    assert os.path.exists(mfa_path), 'mfa path: {} is not exist'.format(mfa_path)
    assert os.path.exists(lexicon_path), 'lexicon_path: {} is not exist'.format(lexicon_path)
    assert os.path.exists(mfa_validate_path), 'mfa_validate_path: {} is not exist'.format(mfa_validate_path)
    
    saved_mfa_model = os.path.join(os.path.join(output_path, dataset, "mfa_model.zip"))
    if os.path.isfile(saved_mfa_model):
        mfa_align_path = os.path.join(
            os.environ['WORK_DIR'], "pretrained", "mfa", "montreal-forced-aligner", "bin", "mfa_align"
        )
        assert os.path.exists(mfa_align_path), 'mfa_align_path: {} is not exist'.format(mfa_align_path)
        print('Train and Align with saved MFA model {}'.format(saved_mfa_model))
        os.system(
            f"{mfa_align_path} {corpus_path} {lexicon_path} {saved_mfa_model} {textgrid_directory} -j {jobs} --clean"
        )
    else:
        print('Validate alignment data')
        os.system(f"{mfa_validate_path} {corpus_path} {lexicon_path}")
        print('Train and Align with MFA')
        
        os.system(
            f"{mfa_path} {corpus_path} {lexicon_path} {textgrid_directory} -o {saved_mfa_model} -j {jobs} --clean"
        )
    print('Alignment Completed')

def libritts_statistics(data_dir):
    speakers = []
    distribution2speakers2pharases2utts = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    distribution_infos = glob(data_dir + "/*")

    for distribution_info in distribution_infos:
        distribution = distribution_info.split("/")[-1]
        # print('distribution: {}'.format(distribution))
        
        speaker_infos = glob(distribution_info + "/*")
        
        
        if len(speaker_infos) == 0:
            continue

        for speaker_info in speaker_infos:
            speaker = speaker_info.split("/")[-1]
            # print('speaker: {}, speaker_info: {}'.format(speaker, speaker_info))
            speakers.append(speaker)
            pharase_infos = glob(speaker_info + "/*")
            # print('speaker: {}, pharase_infos: {}'.format(speaker, pharase_infos))
            
            for pharase_info in pharase_infos:
                pharase = pharase_info.split("/")[-1]
                utts = glob(pharase_info + "/*.wav")
                # print('pharase_info: {}, utts: {}'.format(pharase_info, utts))
                for utt in utts:
                    uid = utt.split("/")[-1].split(".")[0]
                    distribution2speakers2pharases2utts[distribution][speaker][
                        pharase
                    ].append(uid)

    unique_speakers = list(set(speakers))
    unique_speakers.sort()

    print("Speakers: \n{}".format("\t".join(unique_speakers)))
    return distribution2speakers2pharases2utts, unique_speakers


def main(output_path, dataset_path, cfg):
    print("-" * 10)
    print("Preparing samples for libritts...\n")

    save_dir = os.path.join(output_path, "libritts")
    os.makedirs(save_dir, exist_ok=True)
    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")
    valid_output_file = os.path.join(save_dir, "valid.json")
    singer_dict_file = os.path.join(save_dir, "singers.json")
    utt2singer_file = os.path.join(save_dir, "utt2singer")
    # print('train_output_file: {}'.format(train_output_file))
    # if has_existed(train_output_file):
    #     return
    utt2singer = open(utt2singer_file, "w")

    # Load
    libritts_path = dataset_path
    text_grid_path=os.path.join(save_dir, 'TextGrid')
    # print('libritts_path: {}'.format(libritts_path))
    distribution2speakers2pharases2utts, unique_speakers = libritts_statistics(
        libritts_path
    )
    # print('distribution2speakers2pharases2utts: {}'.format(distribution2speakers2pharases2utts))
    # We select pharases of standard spekaer as test songs
    train = []
    test = []
    valid = []

    train_index_count = 0
    test_index_count = 0
    valid_index_count = 0

    train_total_duration = 0
    test_total_duration = 0
    valid_total_duration = 0

    for distribution, speakers2pharases2utts in tqdm(
        distribution2speakers2pharases2utts.items()
    ):
        # print(f'distribution: {distribution}, speakers2pharases2utts: {speakers2pharases2utts}')
            # print('distribution2speakers2pharases2utts: {}'.format(distribution2speakers2pharases2utts))

        for speaker, pharases2utts in tqdm(speakers2pharases2utts.items()):
            # print(f'speaker: {speaker}, pharases2utts: {pharases2utts}')
            pharase_names = list(pharases2utts.keys())

            for chosen_pharase in pharase_names:
                 

                for chosen_uid in pharases2utts[chosen_pharase]:
                    # if chosen_uid == '2428_83705_000000_000000':
                    #     print('tg file: {}'.format(os.path.join(text_grid_path, speaker, chosen_uid + '.TextGrid')))
                    #     os._exit(1)
                    text_grid_file = os.path.join(text_grid_path, speaker, chosen_uid + '.TextGrid')
                    if text_grid_path is not None and (not os.path.isfile(text_grid_file) or not verify_alignment(text_grid_file, cfg)):
                        # Only choose the aligned files
                        # print('text_grid_path: {}'.format(text_grid_path))
                        # print('text_grid_file: {}'.format(text_grid_file))
                        continue
                    # print("distribution: {}, speaker: {}, chosen_pharase: {}, chosen_uid: {}".format(distribution, speaker, chosen_pharase, chosen_uid))
                    # os._exit(1)
                    res = {
                        "Dataset": "libritts",
                        "Singer": speaker,
                        "speaker": speaker,
                        # "Uid": "{}#{}#{}#{}".format(
                        #     distribution, speaker, chosen_pharase, chosen_uid
                        # ),
                        "Uid": chosen_uid,
                    }
                    res["Path"] = "{}/{}/{}/{}.wav".format(
                        distribution, speaker, chosen_pharase, chosen_uid
                    )
                    res["Path"] = os.path.join(libritts_path, res["Path"])
                    assert os.path.exists(res["Path"])

                    text_file_path = os.path.join(
                        libritts_path,
                        distribution,
                        speaker,
                        chosen_pharase,
                        chosen_uid + ".normalized.txt",
                    )
                    with open(text_file_path, "r") as f:
                        lines = f.readlines()
                        assert len(lines) == 1
                        text = lines[0].strip()
                        res["Text"] = text

                    waveform, sample_rate = torchaudio.load(res["Path"])
                    duration = waveform.size(-1) / sample_rate
                    res["Duration"] = duration

                    if "test" in distribution:
                        res["index"] = test_index_count
                        test_total_duration += duration
                        test.append(res)
                        test_index_count += 1
                    elif "train" in distribution:
                        res["index"] = train_index_count
                        train_total_duration += duration
                        train.append(res)
                        train_index_count += 1
                    elif "dev" in distribution:
                        res["index"] = valid_index_count
                        valid_total_duration += duration
                        valid.append(res)
                        valid_index_count += 1

                    utt2singer.write("{}\t{}\n".format(res["Uid"], res["Singer"]))

    print(
        "#Train = {}, #Test = {}, #Valid = {}".format(len(train), len(test), len(valid))
    )
    print(
        "#Train hours= {}, #Test hours= {}, #Valid hours= {}".format(
            train_total_duration / 3600,
            test_total_duration / 3600,
            valid_total_duration / 3600,
        )
    )

    # Save train.json and test.json
    with open(train_output_file, "w") as f:
        json.dump(train, f, indent=4, ensure_ascii=False)
    with open(test_output_file, "w") as f:
        json.dump(test, f, indent=4, ensure_ascii=False)
    with open(valid_output_file, "w") as f:
        json.dump(valid, f, indent=4, ensure_ascii=False)

    # Save singers.json
    singer_lut = {name: i for i, name in enumerate(unique_speakers)}
    with open(singer_dict_file, "w") as f:
        json.dump(singer_lut, f, indent=4, ensure_ascii=False)
