import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def extract_egs(data):
    egs = set()
    for entry in data.values():
        variants = entry.get('variants', [])
        variants = [variant.get('w', '') for variant in variants]

        for definition in entry.get('defs', []):
            for eg in definition.get('egs', []):
                zho = eg.get('zho', None)
                yue = eg.get('yue', None)
                if zho is not None and len(zho) == 2 and zho[1] is not None:
                    egs.add(zho[1])
                if yue is not None and len(yue) == 2 and yue[1] is not None:
                    egs.add(yue[1])
    return egs


if __name__ == '__main__':
    # Load the data from dict.json
    with open('../dict.json', 'r') as file:
        data = json.load(file)
    
    egs = extract_egs(data)
    print(f'Number of egs: {len(egs)}')

    import os
    if not os.path.exists('audio'):
        os.makedirs('audio')

    with ThreadPoolExecutor(max_workers=10) as executor:
        def process_eg(eg):
            import hashlib
            import subprocess
            sha256_hash = hashlib.sha256(eg.encode()).hexdigest()
            audio_file_path = f"audio/{sha256_hash}.m4a"
            if not os.path.exists(audio_file_path):
                # Set macOS system default voice to Chinese (Hong Kong) Siri Voice 2 (Female)
                command = f'say "{eg.replace('"', '')}" -o {audio_file_path}'
                subprocess.run(command, shell=True)
            return None

        list(tqdm(executor.map(process_eg, egs), total=len(egs)))
