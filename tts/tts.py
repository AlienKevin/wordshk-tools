import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import regex as re
import doctest
from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing
import os
from constants import polly_jyutping_syllables


# Create a session using the credentials stored in environment variables
session = Session(aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                  aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                  region_name='ap-southeast-1')
polly = session.client("polly")


def extract_line(line):
    res = ''
    for (_seg_type, seg) in line:
        res += seg
    return res

def normalize(sent):
    return sent.replace(' ', '')


def jyutping_to_ssml(pr):
    """
    >>> jyutping_to_ssml('  teng1 man4  nei5 sing1 zo2 zik1 wo3  , gung1 hei2 saai3 !  ')
    '<speak><phoneme alphabet="x-amazon-jyutping" ph="teng1-man4-nei5-sing1-zo2-zik1-wo3"></phoneme>,<phoneme alphabet="x-amazon-jyutping" ph="gung1-hei2-saai3"></phoneme>!</speak>'
    >>> jyutping_to_ssml('m1')
    Found pr "m1" with missing syllable m1
    >>> jyutping_to_ssml('ng1')
    Found pr "ng1" with missing syllable ng1
    >>> jyutping_to_ssml('ng2')
    Found pr "ng2" with missing syllable ng2
    >>> jyutping_to_ssml('ng3 m1')
    Found pr "ng3 m1" with missing syllable ng3
    >>> jyutping_to_ssml('keoi5 oi5')
    Found pr "keoi5 oi5" with missing syllable oi5
    """
    if pr is None:
        return None

    # Split by any unicode punctuation
    segments = re.split(r'(\p{P})', pr)
    result = []

    for segment in segments:
        if re.match(r'\p{P}', segment):
            result.append(segment)
        else:
            syllables = segment.split()
            # Skip spaces
            if len(syllables) == 0:
                continue
            for syllable in syllables:
                if syllable not in polly_jyutping_syllables:
                    print(f'Found pr "{pr}" with missing syllable {syllable}')
                    return None
            # Split by spaces and join with '-'
            phonemes = '-'.join(syllables)
            # Surround segment by <phoneme> tags
            result.append(f'<phoneme alphabet="x-amazon-jyutping" ph="{phonemes}"></phoneme>')

    return f"<speak>{''.join(result)}</speak>"


def extract_defs_and_egs(data):
    sents = {}
    for entry in data.values():
        variants = entry.get('variants', [])
        variants = [variant.get('w', '') for variant in variants]

        for definition in entry.get('defs', []):
            for yue_line in definition.get('yue', []):
                yue_def = extract_line(yue_line)
                if yue_def not in sents:
                    sents[yue_def] = None
            for eg in definition.get('egs', []):
                zho = eg.get('zho', None)
                yue = eg.get('yue', None)
                if zho is not None:
                    pr = None
                    if len(zho) == 2 and zho[1] is not None:
                        pr = zho[1]
                    sent = extract_line(zho[0])
                    if sent not in sents or sents[sent] is None:
                        sents[sent] = jyutping_to_ssml(pr)
                if yue is not None:
                    pr = None
                    if len(yue) == 2 and yue[1] is not None:
                        pr = yue[1]
                    sent = extract_line(yue[0])
                    if sent not in sents or sents[sent] is None:
                        sents[sent] = jyutping_to_ssml(pr)
    return sents


def polly_tts(ssml, output_path):
    try:
        # Request speech synthesis
        response = polly.synthesize_speech(Text=ssml, Engine="neural", LanguageCode="yue-CN", OutputFormat="mp3", SampleRate="24000", TextType="ssml",
                                            VoiceId="Hiujin")
    except (BotoCoreError, ClientError) as error:
        # The service returned an error, exit gracefully
        print(error)

    # Access the audio stream from the response
    if "AudioStream" in response:
        # Note: Closing the stream is important because the service throttles on the
        # number of parallel connections. Here we are using contextlib.closing to
        # ensure the close method of the stream object will be called automatically
        # at the end of the with statement's scope.
        with closing(response["AudioStream"]) as stream:
            try:
            # Open a file for writing the output as a binary stream
                with open(output_path, "wb") as file:
                    file.write(stream.read())
            except IOError as error:
                # Could not write to file, exit gracefully
                print(error)
    else:
        # The response didn't contain audio data, exit gracefully
        print("Could not stream audio")


def normalize_file_name(sent):
    # Aliyun OSS Object的命名规范如下：
    # 使用UTF-8编码。
    # 长度必须在1~1023字节之间。
    # 不能以正斜线（/）或者反斜线（\）开头。
    # 区分大小写。
    return sent.replace('/', '／').replace('\\', '＼').replace(':', '：')


def process_sent_for_pr(sent):
    """
    >>> process_sent_for_pr("hou2多coi2")
    '<phoneme alphabet="x-amazon-jyutping" ph="hou2"></phoneme>多<phoneme alphabet="x-amazon-jyutping" ph="coi2"></phoneme>'
    
    >>> process_sent_for_pr("This is a test (with parentheses).")
    'This is a test ，with parentheses，.'
    
    >>> process_sent_for_pr("個/塊/張")
    '個，塊，張'

    >>> process_sent_for_pr("(這是測試)")
    '，這是測試，'
    
    >>> process_sent_for_pr("（這是測試）")
    '，這是測試，'

    >>> process_sent_for_pr("（個/塊/張）")
    '，個，塊，張，'
    """
    # Surround raw jyutping syllables with <phoneme> tags
    for syllable in polly_jyutping_syllables:
        sent = re.sub(rf'([^a-z0-9]|\b){syllable}([^a-z0-9]|\b)', rf'\1<phoneme alphabet="x-amazon-jyutping" ph="{syllable}"></phoneme>\2', sent)
    # Replace parentheses with commas to add a pause before the parenthesized segment
    sent = re.sub(r'\((.*?)\)', r'，\1，', sent)
    sent = re.sub(r'（(.*?)）', r'，\1，', sent)
    # Add a pause between two chinese characters separated by a slash, eg 個/塊
    while re.search(r'([\p{sc=Han}])[/／]([\p{sc=Han}])', sent):
        sent = re.sub(r'([\p{sc=Han}])[/／]([\p{sc=Han}])', r'\1，\2', sent)
    return sent


if __name__ == '__main__':
    doctest.testmod()

    # Load the data from dict.json
    with open('../dict.json', 'r') as file:
        data = json.load(file)
    
    sents = extract_defs_and_egs(data)
    sents = {k: v for k, v in list(sents.items())}
    # print(f'Number of sents: {len(sents)}')
    char_count = sum(len(key) for key in sents.keys())
    print(f'Total number of characters in sents keys: {char_count}')

    if not os.path.exists('audio'):
        os.makedirs('audio')
    else:
        # Get all existing .mp3 audios in audio/ and remove them from egs
        existing_audios = {f.split('.')[0] for f in os.listdir('audio') if f.endswith('.mp3')}
        sents = {k: v for k, v in sents.items() if normalize(k) not in existing_audios}
        print(f'Number of audios already present: {len(existing_audios)}')

    with ThreadPoolExecutor(max_workers=16) as executor:
        def process_sent(sent):
            sent, pr = sent
            if pr is None:
                pr = f'<speak>{process_sent_for_pr(sent)}</speak>'
            sent = normalize(sent)
            audio_file_path = f"audio/{normalize_file_name(sent)}.mp3"
            if not os.path.exists(audio_file_path):
                polly_tts(pr, audio_file_path)
            return None

        list(tqdm(executor.map(process_sent, sents.items()), total=len(sents)))
