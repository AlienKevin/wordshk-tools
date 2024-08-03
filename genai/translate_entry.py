from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import json
from threading import Lock
import copy


with open('deepseek_api_key.txt', 'r') as file:
    api_key = file.read().strip()

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

with open('translate_entry_prompt.txt', 'r') as file:
    prompt = file.read()

def translate(entry):
    entry_copy = copy.deepcopy(entry)
    entry_copy.pop('id', None)
    entry_copy.pop('defIndex', None)
    entry_copy['yueVariants'] = entry_copy.pop('variants')
    entry_content = json.dumps(entry_copy, ensure_ascii=False)

    attempts = 0
    while True:
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{
                            "role": "system",
                            "content": prompt
                          },
                          {
                            "role": "user",
                            "content": entry_content
                          }],
                response_format={
                    'type': 'json_object'
                },
                max_tokens=256,
                temperature=1.1,
                stream=False
            )
            result = response.choices[0].message.content
            result_json = json.loads(result)
            if not all(key in result_json for key in ["zhoVariants", "zhoDef", "zhoEgs"]):
                raise ValueError(f"Missing required keys in the result: {result}")
            if not isinstance(result_json["zhoVariants"], list) or not all(isinstance(item, str) for item in result_json["zhoVariants"]):
                raise ValueError(f"zhoVariants must be a list of strings: {result}")
            if not isinstance(result_json["zhoDef"], str):
                raise ValueError(f"zhoDef must be a single string: {result}")
            if not isinstance(result_json["zhoEgs"], list) or not all(isinstance(item, str) for item in result_json["zhoEgs"]) or len(result_json["zhoEgs"]) != len(entry['egs']):
                raise ValueError(f"zhoEgs must be a list of strings and have the same length as egs: {result}")
            return result_json
        except Exception as e:
            print(f"{entry['id']} encountered an error: {e}")
            time.sleep(1)
            attempts += 1
            if attempts >= 3:
                return {'error': str(e)}

def extract_line(line):
    res = ''
    for (_seg_type, seg) in line:
        res += seg
    return res


def extract_yue_variants_and_defs(data):
    entries = []
    for entry in data.values():
        variants = entry.get('variants', [])
        variants = [variant.get('w', '') for variant in variants]

        for def_index, definition in enumerate(entry.get('defs', [])):
            yue_def_lines = []
            for line in definition.get('yue', []):
                yue_def_line = extract_line(line)
                yue_def_lines.append(yue_def_line)
            
            if definition.get('eng') is None:
                continue

            eng_def_lines = []
            for line in definition.get('eng', []):
                eng_def_line = extract_line(line)
                eng_def_lines.append(eng_def_line)
            
            egs = []
            for eg in definition.get('egs'):
                zho = extract_line(eg.get('zho')[0]) if eg.get('zho') else None
                yue = extract_line(eg.get('yue')[0]) if eg.get('yue') else None
                eng = extract_line(eg.get('eng')) if eg.get('eng') else None
                lzh = extract_line(eg.get('lzh')[0]) if eg.get('lzh') else None
                eg_entry = {}
                if zho is not None:
                    eg_entry['zho'] = zho
                if yue is not None:
                    eg_entry['yue'] = yue
                if eng is not None:
                    eg_entry['eng'] = eng
                if lzh is not None:
                    eg_entry['lzh'] = lzh
                egs.append(eg_entry)

            entries.append({"id": entry.get('id'), "variants": variants, "defIndex": def_index, "yueDef": '\n'.join(yue_def_lines), "engDef": '\n'.join(eng_def_lines), "egs": egs})
    return entries


if __name__ == '__main__':
    # Load the data from dict.json
    with open('../dict.json', 'r') as file:
        data = json.load(file)

    # Extract variants and definitions using the predefined function
    extracted_entries = extract_yue_variants_and_defs(data)

    print(len(extracted_entries))
    exit(0)

    # print('Example parsed entry:')
    # for entry in extracted_entries:
    #     if entry['id'] in [69809, 79604, 102346, 70006, 121119, 109367, 8775, 105340, 90930, 5445]:
    #         print(entry)

    existing_entries = set()
    try:
        with open('entry_results.jsonl', 'r') as file:
            for line in file:
                existing_entry = json.loads(line)
                existing_entries.add((existing_entry['id'], existing_entry['defIndex']))
    except FileNotFoundError:
        pass

    print(f'Skipping {len(existing_entries)} already generated entries')

    extracted_entries = [entry for entry in extracted_entries if (entry['id'], entry['defIndex']) not in existing_entries]

    with ThreadPoolExecutor(max_workers=100) as executor:
        with open('entry_results.jsonl', 'a+') as file:
            lock = Lock()
            def process_entry(entry):
                translation = translate(entry)
                if 'error' in translation:
                    print(f"Translation failed for {entry['id']}: {translation['error']}")
                else:
                    result = {"id": entry['id'],
                            "variants": entry['variants'],
                            "zhoVariants": translation['zhoVariants'],
                            "defIndex": entry['defIndex'],
                            "yueDef": entry['yueDef'],
                            "zhoDef": translation['zhoDef'],
                            "engDef": entry['engDef'],
                            "egs": entry['egs'],
                            "zhoEgs": translation['zhoEgs']
                            }
                    with lock:
                        file.write(json.dumps(result, ensure_ascii=False) + '\n')
                        file.flush()
            list(tqdm(executor.map(process_entry, extracted_entries), total=len(extracted_entries)))
