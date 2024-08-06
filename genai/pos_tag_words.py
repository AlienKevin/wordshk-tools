from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import json
from threading import Lock
import random
import pycantonese
import re
import argparse

random.seed(42)

parser = argparse.ArgumentParser(description="POS Tagging with OpenAI API")
parser.add_argument('--model', type=str, choices=['deepseek-chat', 'deepseek-coder', 'gpt-4o', 'gpt-4o-mini', 'qwen-max', 'qwen-plus', 'qwen-turbo'], required=True, help='Model to use for POS tagging')
parser.add_argument('--sample_size', type=int, default=100, help='Number of samples to test')
parser.add_argument('--prompt_version', type=str, choices=['v1', 'v2'], required=True, help='Prompt version to use for POS tagging')
parser.add_argument('--prompt_dataset', type=str, choices=['hkcancor', 'ud_yue'], required=True, help='Dataset to use for POS tagging')
parser.add_argument('--eval_dataset', type=str, choices=['hkcancor', 'ud_yue'], required=True, help='Dataset to use for evaluation')
parser.add_argument('--max_workers', type=int, default=100, help='Maximum number of workers to use for parallel processing')
args = parser.parse_args()

if args.model.startswith('deepseek'):
    base_url = "https://api.deepseek.com"
    with open('deepseek_api_key.txt', 'r') as file:
        api_key = file.read().strip()
elif args.model.startswith('gpt-4o'):
    base_url = None
    with open('openai_api_key.txt', 'r') as file:
        api_key = file.read().strip()
elif args.model.startswith('qwen'):
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    with open('qwen_api_key.txt', 'r') as file:
        api_key = file.read().strip()

client = OpenAI(api_key=api_key, base_url=base_url)

valid_pos_tags = {"ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"}

def segment_words(sentence):
    attempts = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{
                            "role": "system",
                            "content": pos_prompt
                          },
                          {
                            "role": "user",
                            "content": sentence
                          }],
                response_format={
                    'type': 'json_object'
                },
                max_tokens=2000,
                temperature=1.1,
                stream=False
            )
            result = response.choices[0].message.content
            try:
                result_json = json.loads(result)
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse JSON response: {str(e)}")
            if "pos_tagged_words" in result_json and isinstance(result_json["pos_tagged_words"], list) and \
                all(isinstance(item, list) and len(item) == 2 and (all(isinstance(sub_item, str) for sub_item in item)) for item in result_json["pos_tagged_words"]):
                concatenated_words = "".join([word for word, pos in result_json["pos_tagged_words"]])
                if concatenated_words == sentence:
                    for word, pos in result_json["pos_tagged_words"]:
                        if pos not in valid_pos_tags:
                            raise Exception(f"Invalid POS tag '{pos}' in the result")
                    return result_json["pos_tagged_words"]
                else:
                    raise Exception(f"Segmentation result does not match the input sentence")
            else:
                raise Exception(f"Invalid segmentation result format")
        except Exception as e:
            time.sleep(1)
            attempts += 1
            if attempts >= 3:
                return {'error': str(e), 'result': result}


def load_hkcancor():
    # Load HKCanCor data
    hkcancor_data = pycantonese.hkcancor()

    # Gather all word segmented utterances
    utterances = [[(token.word, pycantonese.pos_tagging.hkcancor_to_ud(token.pos)) for token in utterance] for utterance in hkcancor_data.tokens(by_utterances=True)]

    return utterances


def split_hkcancor():
    random.seed(42)

    utterances = load_hkcancor()

    latin_fragmented_utterances = []
    latin_complete_utterances = []
    non_latin_utterances = []

    for utterance in utterances:
        if any(re.search(r'[a-zA-Z0-9]', char) for (word, pos) in utterance for char in word):
            if ''.join(word for word, pos in utterance).count('"') % 2 != 0:
                latin_fragmented_utterances.append(utterance)
            else:
                latin_complete_utterances.append(utterance)
        else:
            non_latin_utterances.append(utterance)

    # Sample 3 random latin_complete_utterances, 3 random latin_fragmented_utterances, and 4 random non_latin_utterances for in_context_samples
    random.shuffle(latin_complete_utterances)
    random.shuffle(latin_fragmented_utterances)
    random.shuffle(non_latin_utterances)
    in_context_samples = latin_complete_utterances[:3] + latin_fragmented_utterances[:3] + non_latin_utterances[:4]
    random.shuffle(in_context_samples)
    
    return in_context_samples


def upos_id_to_str(upos_id):
    names=[
        "NOUN",
        "PUNCT",
        "ADP",
        "NUM",
        "SYM",
        "SCONJ",
        "ADJ",
        "PART",
        "DET",
        "CCONJ",
        "PROPN",
        "PRON",
        "X",
        "X",
        "ADV",
        "INTJ",
        "VERB",
        "AUX",
    ]
    return names[upos_id]


def load_ud_yue():
    from datasets import load_dataset

    # Load the universal_dependencies dataset from Hugging Face
    dataset = load_dataset('universal-dependencies/universal_dependencies', 'yue_hk', trust_remote_code=True)

    # Gather all word segmented utterances
    utterances = [[(token, upos_id_to_str(pos)) for token, pos in zip(sentence['tokens'], sentence['upos'])] for sentence in dataset['test']]

    return utterances


def split_ud_yue():
    import random

    random.seed(42)

    utterances = load_ud_yue()

    # Shuffle the utterances to ensure randomness
    random.shuffle(utterances)

    # Sample 10 random utterances for in_context_samples
    in_context_samples = utterances[:10]

    # The rest of the utterances will be used as testing_samples
    testing_samples = utterances[10:]

    return in_context_samples


def generate_prompt_v1(prompt_dataset):
    if prompt_dataset == 'hkcancor':
        in_context_samples = split_hkcancor()
    elif prompt_dataset == 'ud_yue':
        in_context_samples = split_ud_yue()
    
    # Format in-context samples for the prompt
    in_context_prompt = "\n\n".join([
        f'EXAMPLE INPUT SENTENCE:\n{"".join([word for word, pos in sample])}\n\nEXAMPLE JSON OUTPUT:\n{json.dumps({"pos_tagged_words": [[word, pos] for word, pos in sample]}, ensure_ascii=False)}'
        for sample in in_context_samples
    ])

    # Update the word segmentation prompt with in-context samples
    pos_prompt = f"""You are an expert at Cantonese word segmentation and POS tagging. Output the segmented words with their parts of speech as JSON arrays. ALWAYS preserve typos, fragmented chunks, and punctuations in the original sentence. Output the parts of speech in the Universal Dependencies v2 tagset with 17 tags:
ADJ: adjective
ADP: adposition
ADV: adverb
AUX: auxiliary
CCONJ: coordinating conjunction
DET: determiner
INTJ: interjection
NOUN: noun
NUM: numeral
PART: particle
PRON: pronoun
PROPN: proper noun
PUNCT: punctuation
SCONJ: subordinating conjunction
SYM: symbol
VERB: verb
X: other

{in_context_prompt}"""
    
    return pos_prompt



def generate_prompt_v2(prompt_dataset):
    if prompt_dataset == 'hkcancor':
        in_context_samples = split_hkcancor()
    elif prompt_dataset == 'ud_yue':
        in_context_samples = split_ud_yue()
    
    # Format in-context samples for the prompt
    in_context_prompt = "\n\n".join([
        f'EXAMPLE INPUT SENTENCE:\n{"".join([word for word, pos in sample])}\n\nEXAMPLE JSON OUTPUT:\n{json.dumps({"pos_tagged_words": [[word, pos] for word, pos in sample]}, ensure_ascii=False)}'
        for sample in in_context_samples
    ])

    # Update the word segmentation prompt with in-context samples
    pos_prompt = f"""You are an expert at Cantonese word segmentation and POS tagging. Output the segmented words with their parts of speech as JSON arrays. ALWAYS preserve typos, fragmented chunks, and punctuations in the original sentence. Output the parts of speech in the Universal Dependencies v2 tagset with 17 tags:


ADJ: Adjectives are words that typically modify nouns and specify their properties or attributes. They may also function as predicates. These include the categories known as 區別詞 and 形容詞.

好/ADJ 風景
天氣 好 好/ADJ

The adjective may by accompanied by the particle 嘅 when functioning as a prenominal modifier (for either 區別詞 or 形容詞), and often obligatorily when functioning as a predicate if it is a 區別詞.

嚴重/ADJ 嘅 問題
呢 種 癌症 係 慢性/ADJ 嘅

Note that ordinal numerals such as 第一 and 第三 are to be treated as adjectives and tagged ADJ per UD specifications, even though they are traditionally classified as numerals in Chinese.

Examples
形容詞: 好, 靚, 細, 老
區別詞: 女, 慢性, 親生
Ordinal numbers: 第一, 第三, 第五十三


ADP: The Cantonese ADP covers three categories of function words analyzed as adpositions: (1) prepositions, (2) valence markers, and (3) “localizers”/postpositions.

Prepositions introduce an extra argument to the event/main verb in a clause or give information about the time or the location or direction, etc.

佢 喺/ADP 2000年 出世

Valence markers such as 將 and the formal 被 are also tagged ADP.

你 將/ADP 啲 嘢 擺 低 喇
兇手 凌晨 一點 被/ADP 捕

Localizers (also known as 方位詞), typically indicate spatial information in relation to the noun preceding it. Some localizers have also grammaticalized into clausal markers indicating temporal information. Localizers with the clausal function are still tagged as ADP (but are labeled with the dependency relation mark).

我 擺 咗 啲 梨 喺 枱 度/ADP
佢 老婆 死 咗 之後/ADP ， 佢 都 唔 想 生存 落去 嘑

Examples
Prepositions: 喺, 畀, 由, 為咗
Valence markers: 將, 被 (passive)
Localizers / postpositions: 度, 之後, 之前, 時


ADV:Adverbs (副詞 / fu3ci4) typically modify verbs, adjectives, and other adverbs for such categories as time, manner, degree, frequency, or negation.

你 慢慢/ADV 講
佢 非常之/ADV 開心
佢哋 好/ADV 早/ADV 起身

Some adverbs also modify clauses with conjunctive and discursive functions.

佢 反而/ADV 冇 同 我 講
今日 當然/ADV 冇 人 喇

A small number of adverbs may also modify numerals and determiners, or nouns and pronouns.

差唔多/ADV 五千
連/ADV 佢 都 唔 去 嘑

There is a closed subclass of pronominal adverbs that refer to circumstances in context, rather than naming them directly; similarly to pronouns, these can be categorized as interrogative, demonstrative, etc. These should be treated as adverbs when modifying a predicate, but otherwise some of them can function as a nominal in the syntax, in which case they should be tagged PRON.

你 點解/ADV 唔 來
我 係 咁樣/ADV 做 嘅

Note that although some adverbs express temporal information, many common time expressions (e.g., 今日, 舊年, 夜晚) are actually nouns and should be tagged NOUN.

Examples
Manner adverbs: 慢慢, 互相
Temporal, aspectual, and modal adverbs: 就嚟, 喺度, 寧可, 可能 (NB: 可能 can also function as ADJ “possible” and NOUN “possibility”)
Conjunctive adverbs: 所以, 反而, 然後, 噉, 就
Frequency and duration adverbs: 成日, 一陣
Negation adverbs: 唔, 未
Numeral-modifying adverbs: 大概, 差唔多, 至少, 最多
Noun-modifying adverbs: 連
Pronominal adverbs: 點樣, 噉樣, 咁, 點解
Other: 都, 亦都, 先至, 越嚟越, 當然, 譬如, 譬如話, 例如, 好似 (note 好似 can also function as a main verb; an example of the adverbial usage is 好似 佢 琴日 噉 講…)


AUX: An auxiliary is a word that accompanies the lexical verb of a verb phrase and expresses grammatical distinctions not carried by the lexical verb, such as person, number, tense, mood, aspect, and voice.

In Cantonese, auxiliaries can be divided into modal and modal-like auxiliaries which are mostly preverbal (except for the postverbal usage of 得) and do not have to be adjacent to the verb, and aspect markers, which must come immediately after the verb and certain verb compounds. Note that some modal auxiliaries can also function as main verbs, usually when they have a direct object or full clausal complement.

Examples
Modal and modal-like auxiliaries: 能夠, 會, 可以, 應該, 肯, 敢, 有 (perfective), 冇 (negative perfective), 得
Aspect markers: 咗 (perfective), 住 (durative), 過 (experiential), 緊 (progressive), 吓 (delimitative), 開 (habitual)


CCONJ: A coordinating conjunction is a word that links words or larger constituents without syntactically subordinating one to the other and expresses a semantic relationship between them.

Examples
“and”: 同, 同埋, 而, 而且. Note that 同 may also function as a preposition (ADP)
“or”: 或者, 定(係)
“but”: 但係


DET: Determiners are words that modify nouns or noun phrases and express the reference of the noun phrase in context.

Note that Chinese does not traditionally define determiners as a separate word class, but categorizes them as pronouns and/or adjectives. For this reason, in the UD framework some words in Cantonese may function as determiners and be tagged DET in one syntactic context (i.e., when modifying a noun), and as pronouns (tagged PRON) when behaving as a nominal or the head of a nominal phrase.

Examples
Demonstrative determiners: 呢, 嗰
Possessive determiner: 本
Quantifying determiners:
Definite: 每, 成, 全, 全部, 所有
Indefinite: 任何, 有啲, (好)多, (好)少, 幾, 大部分, 唔少, 多數
Interrogative determiners: 邊, 乜(嘢), 咩, 幾多
Other: 上, 下, 前, 後, 頭, 其餘, 某, 其他, 另(外), 同


INTJ: An interjection is a word that is used most often as an exclamation or part of an exclamation. It typically expresses an emotional reaction, is not syntactically related to other accompanying expressions, and may include a combination of sounds not otherwise found in the language.

Note that words primarily belonging to another part of speech retain their original category when used in exclamations. For example, 係！ would still be tagged VERB.

Onomatopoeic words (擬聲詞) should only be treated as interjections if they are used as an exclamation (e.g., 喵!), otherwise they should be tagged according to their syntactic function in context (often as adverbs in Chinese, e.g., 佢哋 吱吱喳喳/ADV 噉 叫).

Examples: 哦, 哎呀, 咦


NOUN: Nouns are a part of speech typically denoting a person, place, thing, animal, or idea.

The NOUN tag is intended for common nouns only. See PROPN for proper nouns and PRON for pronouns.

As a special case, classifiers (量詞) are also tagged NOUN per UD guidelines. Their classifier status may be preserved in the feature column (FEATS) as NounType=CLf.

Examples
Nouns: 杯, 草, 氧氣, 地方, 能力, 歷史
Classifiers: 個, 條, 本, 對, 杯, 磅, 年


NUM: A numeral is a word, functioning most typically as a determiner, a pronoun or an adjective, that expresses a number and a relation to the number, such as quantity, sequence, frequency or fraction.

Cardinal numerals are covered by NUM regardless of syntactic function and regardless of whether they are expressed as words (五 / ng5 “five”) or digits (5). By contrast, ordinal numerals (such as 第一) are always tagged ADJ.

Examples: 1, 2, 3, 4, 5, 100, 10,358, 5.23, 3/4, 一, 二, 三, 一百, 五十六, 一萬三百五十八, 四分之三


PART: Particles are function words that must be associated with another word, phrase, or clause to impart meaning and that do not satisfy definitions of other universal parts of speech (such as ADP, AUX, CCONJ, or SCONJ).

In Cantonese, particles include the genitive/associative/relativizer/nominalizer marker 嘅; 得 and 到 in V-得/到 extent/descriptive constructions; the manner adverbializer 噉; the “et cetera” marker 等(等); sentence-final particles; the quantifiers 埋 and 晒; the adversative 親; and so on.

Examples
Genitive 的: 狗 嘅/PART 尾巴
Associative 的: 開心 嘅/PART 小朋友
Relativizer 的: 跴 單車 嘅/PART 人
Nominalizer 的: 食 唔 晒 嘅/PART 唔 准 走
Extent/descriptive 得到:
佢 跑 得/PART 好 快
佢 跑 到/PART 出 晒 汗
Adverbializer 地: 佢 慢慢 噉/PART 跑
Etc. marker: 呢度 有 梨，橙，等等/PART
Sentence-final particles:
你 想 去 呀/PART ？
我哋 一齊 去 喇/PART
佢 一 個 人 啫/PART
佢哋 一定 嚟 嘅/PART
Quantifiers:
畀 埋/PART 我
佢哋 走 晒/PART
Adversative 親: 佢 跌 親/PART


PRON: Pronouns are words that substitute for nouns or noun phrases, whose meaning is recoverable from the linguistic or extralinguistic context. Some pronouns – in particular certain demonstrative, total, indefinite, and interrogatives pronouns – may also function as determiners (DET) and are tagged as such when functioning as a modifier of a noun.

Examples
Personal pronouns: 我, 你, 佢, 我哋, 你哋, 佢哋, 人哋
Reciprocal and reflexive pronouns: 自己, 我自己, 你自己, 佢哋自己
Total pronouns: 全部, 大家
Indefinite pronouns: 有啲, 好多, 大部分, 唔少, 多數, 其他
Locational pronouns: 呢度, 呢邊, 呢處, 嗰度, 嗰邊, 嗰處
Manner pronouns: 咁樣 (also ADV)
Interrogative pronouns: 邊個, 乜(嘢), 咩, 幾多, 邊度, 點樣 (also ADV)
Other:
X + 方: 對方, 雙方
X + 者: 前者, 後者, 兩者, 三者
其餘


PROPN: A proper noun is a noun (or nominal content word) that is the name (or part of the name) of a specific individual, place, or object. For institutional names that contain regular words belonging to other parts of speech such as nouns (e.g., 公司, 大學, etc.), those words should be segmented as their own tokens and still tagged their native part of speech; only the proper nouns in such complex names should be tagged PROPN.

Examples
孔子/PROPN
亞洲/PROPN
威威/PROPN 木材 公司


PUNCT: Punctuation marks are character groups used to delimit linguistic units in printed text.

Punctuation is not taken to include logograms such as $, %, and §, which are instead tagged as SYM.

Examples
Period: 。
Comma: ， 、
Quotation marks: ‘ ’ “ ”「 」 『 』
Title marks: 《 》


SCONJ: A subordinating conjunction is a conjunction that links constructions by making one of them a constituent of the other. The subordinating conjunction typically marks the incorporated constituent which has the status of a subordinate clause.

Subordinating conjunctions in Cantonese include all markers of subordinate clauses, including conditional clauses, purpose clauses, etc.

In paired clauses where both clauses are marked by a conjunctive word and the first is subordinate to the second, we treat the conjunctive word in the first clause as SCONJ, whereas the one in the second, main clause as an adverb (ADV) (e.g., 雖然/SCONJ… 但係/ADV…).

Examples: 如果, 嘅話, 雖然, 即管, 無論
嚟: 你 慳 嚟/SCONJ 做 乜 呀 ?


SYM: A symbol is a word-like entity that differs from ordinary words by form, function, or both.

Many symbols are or contain special non-alphanumeric, non-standard logographic characters, similarly to punctuation. What makes them different from punctuation is that they can be substituted by normal words. This involves all currency symbols, e.g. $ 75 is identical to 七十五 蚊.

Mathematical operators form another group of symbols.

Another group of symbols is emoticons and emoji.

Examples
＄, ％, §, ©
+, −, ×, ÷, =, <, >
:), ^_^, 😝
siu.ming@universal.org, http://universaldependencies.org/, 1-800-COMPANY


VERB: A verb is a member of the syntactic class of words that typically signal events and actions, can constitute a minimal predicate in a clause, and govern the number and types of other constituents which may occur in the clause.

Despite its use in copular constructions, 係 is tagged as a verb due to its other non-copular meanings and functions.

Examples: 游水, 聽, 呼吸, 鍾意, 決定, 有, 係


X: The tag X is used for words that for some reason cannot be assigned a real part-of-speech category. It should be used very restrictively.

A special usage of X is for cases of code-switching where it is not possible (or meaningful) to analyze the intervening language grammatically (and where the dependency relation foreign is typically used in the syntactic analysis). This usage does not extend to ordinary loan words which should be assigned a normal part-of-speech.

Example
佢 突然 話 ：「 lkjwe/X 」


{in_context_prompt}"""
    
    return pos_prompt


if __name__ == "__main__":
    if args.prompt_version == 'v1':
        pos_prompt = generate_prompt_v1(args.prompt_dataset)
    elif args.prompt_version == 'v2':
        pos_prompt = generate_prompt_v2(args.prompt_dataset)

    # Write the updated word segmentation prompt to the file
    with open(f'data/pos_{args.prompt_dataset}_prompt_{args.prompt_version}.txt', 'w', encoding='utf-8') as f:
        f.write(pos_prompt)

    if args.eval_dataset == 'ud_yue':
        testing_samples = load_ud_yue()
    elif args.eval_dataset == 'hkcancor':
        testing_samples = load_hkcancor()
    
    random.seed(42)
    random.shuffle(testing_samples)
    testing_samples = testing_samples[:args.sample_size]

    with open(f'outputs/pos_{args.eval_dataset}_{args.model}_prompt_{args.prompt_version}.jsonl', 'w', encoding='utf-8') as file, open(f'outputs/pos_errors_{args.eval_dataset}_{args.model}_prompt_{args.prompt_version}.jsonl', 'w', encoding='utf-8') as error_file:
        lock = Lock()
        def process_sample(sample):
            input_sentence = "".join([word for word, pos in sample])
            pos_result = segment_words(input_sentence)
            if 'error' in pos_result:
                print(f"POS tagging failed for sentence: {input_sentence}")
                print(f"Error: {pos_result['error']}")
                error_result = {
                    "reference": sample,
                    "error": pos_result['error'],
                    "hypothesis": pos_result['result']
                }
                with lock:
                    error_file.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                    error_file.flush()
                result = {
                    "reference": sample,
                    "hypothesis": [(char, "X") for char in list(input_sentence)]
                }
            else:
                result = {
                    "reference": sample,
                    "hypothesis": pos_result
                }
            with lock:
                file.write(json.dumps(result, ensure_ascii=False) + '\n')
                file.flush()
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            list(tqdm(executor.map(process_sample, testing_samples), total=len(testing_samples)))
