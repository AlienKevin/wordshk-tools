import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import hashlib
import subprocess
import os

def extract_line(line):
    res = ''
    for (_seg_type, seg) in line:
        res += seg

    return res

def extract_egs(data):
    egs = {}
    for entry in data.values():
        variants = entry.get('variants', [])
        variants = [variant.get('w', '') for variant in variants]

        for definition in entry.get('defs', []):
            for eg in definition.get('egs', []):
                zho = eg.get('zho', None)
                yue = eg.get('yue', None)
                if zho is not None:
                    pr = None
                    if len(zho) == 2 and zho[1] is not None:
                        pr = zho[1]
                    sent = extract_line(zho[0])
                    if sent not in egs or egs[sent] is None:
                        egs[sent] = pr
                if yue is not None:
                    pr = None
                    if len(yue) == 2 and yue[1] is not None:
                        pr = yue[1]
                    sent = extract_line(yue[0])
                    if sent not in egs or egs[sent] is None:
                        egs[sent] = pr
    return egs


if __name__ == '__main__':
    # Load the data from dict.json
    with open('../dict.json', 'r') as file:
        data = json.load(file)
    
    egs = extract_egs(data)
    print(f'Number of egs: {len(egs)}')

    # with open('egs.txt', 'w') as file:
    #     for eg in egs.items():
    #         file.write(f'{eg[0]}: {eg[1]}\n')

    if not os.path.exists('audio'):
        os.makedirs('audio')
    else:
        # Get all existing .m4a audios in audio/ and remove them from egs
        existing_audios = {f.split('.')[0] for f in os.listdir('audio') if f.endswith('.m4a')}
        egs = {k: v for k, v in egs.items() if hashlib.sha256(k.encode()).hexdigest() not in existing_audios}
        print(f'Number of audios already present: {len(existing_audios)}')

    with ThreadPoolExecutor(max_workers=10) as executor:
        def process_eg(eg):
            sent, pr = eg
            if pr is None:
                pr = sent
            sha256_hash = hashlib.sha256(sent.encode()).hexdigest()
            audio_file_path = f"audio/{sha256_hash}.m4a"
            if not os.path.exists(audio_file_path):
                # Set macOS system default voice to Chinese (Hong Kong) Siri Voice 2 (Female)
                command = f'say "{pr.replace('"', '')}" -o {audio_file_path}'
                subprocess.run(command, shell=True)
            return None

        list(tqdm(executor.map(process_eg, egs.items()), total=len(egs)))
