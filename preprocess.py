# -*- coding: utf-8 -*-

import os
import fnmatch
from ppg import BASE_DIR
from ppg.utils import exist, load_json, dump_json
from ppg.signal import smooth_ppg_signal, extract_ppg_single_waveform
from ppg.signal import extract_rri, interpolate_rri


def preprocess():
    segmented_data_dir = os.path.join(BASE_DIR, 'data', 'segmented_calculate_rest_5min')
    preprocessed_data_dir = os.path.join(BASE_DIR, 'data', 'preprocessed_calculate_rest_5min')

    if exist(pathname=segmented_data_dir):
        for filename_with_ext in fnmatch.filter(os.listdir(segmented_data_dir), '*.json'):
            pathname = os.path.join(segmented_data_dir, filename_with_ext)
            json_data = load_json(pathname=pathname)
            if json_data is not None:
                for session_id in json_data:
                    if json_data[session_id]['ppg']['signal'] is not None:
                        json_data[session_id]['ppg']['single_waveforms'] = extract_ppg_single_waveform(signal=smooth_ppg_signal(signal=json_data[session_id]['ppg']['signal'], sample_rate=json_data[session_id]['ppg']['sample_rate']))
                    else:
                        json_data[session_id]['ppg']['single_waveforms'] = None
                    del json_data[session_id]['ppg']['signal']
            dump_json(data=json_data, pathname=os.path.join(preprocessed_data_dir, filename_with_ext), overwrite=True)


if __name__ == '__main__':
    preprocess()
