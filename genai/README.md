(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model deepseek-chat --prompt_version v1 --eval_dataset ud_yue
POS Tagging Accuracy: 0.8875878220140515
Token Accuracy: 0.973159509202454
Token F1 Score: 0.9292001530807501
Token Precision: 0.9309815950920245
Token Recall: 0.9274255156608098

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model deepseek-chat --prompt_version v1 --eval_dataset ud_yue --segmentation_given True
POS Tagging Accuracy: 0.9136745607333843
Token Accuracy: 1.0
Token F1 Score: 1.0
Token Precision: 1.0
Token Recall: 1.0

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model deepseek-chat --prompt_version v1_max_diversity --eval_dataset ud_yue --segmentation_given True
POS Tagging Accuracy (Normalized): 0.9264705882352942
POS Tagging Accuracy: 0.9174836601307189
Token Accuracy: 1.0
Token F1 Score: 1.0
Token Precision: 1.0
Token Recall: 1.0

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model deepseek-chat --prompt_version v2 --eval_dataset ud_yue
POS Tagging Accuracy: 0.9036519036519037
Token Accuracy: 0.9776751347190146
Token F1 Score: 0.9409509202453988
Token Precision: 0.9445727482678984
Token Recall: 0.9373567608861727

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model deepseek-chat --prompt_version v2 --eval_dataset ud_yue --segmentation_given True
POS Tagging Accuracy: 0.920550038197097
Token Accuracy: 1.0
Token F1 Score: 1.0
Token Precision: 1.0
Token Recall: 1.0

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model deepseek-chat --prompt_version v2_max_diversity --eval_dataset ud_yue --segmentation_given True
POS Tagging Accuracy (Normalized): 0.9244406922752215
POS Tagging Accuracy: 0.9151540734487125
Token Accuracy: 1.0
Token F1 Score: 0.9987336428872942
Token Precision: 0.9983122362869198
Token Recall: 0.9991554054054054

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model deepseek-chat --prompt_version v2 --eval_dataset ud_yue --segmentation_given True --to_simplified True
POS Tagging Accuracy (Normalized): 0.9197860962566845
POS Tagging Accuracy: 0.9197860962566845
Token Accuracy: 1.0
Token F1 Score: 1.0
Token Precision: 1.0
Token Recall: 1.0

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model deepseek-coder --prompt_version v1 --eval_dataset ud_yue
POS Tagging Accuracy: 0.890795631825273
Token Accuracy: 0.9730769230769231
Token F1 Score: 0.9298581832119586
Token Precision: 0.933076923076923
Token Recall: 0.9266615737203973

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model deepseek-coder --prompt_version v1 --eval_dataset ud_yue --segmentation_given True
POS Tagging Accuracy: 0.9136745607333843
Token Accuracy: 1.0
Token F1 Score: 1.0
Token Precision: 1.0
Token Recall: 1.0

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model deepseek-coder --prompt_version v2 --eval_dataset ud_yue
POS Tagging Accuracy: 0.9033760186263097
Token Accuracy: 0.9807987711213517
Token F1 Score: 0.9467636920720031
Token Precision: 0.9493087557603687
Token Recall: 0.9442322383498855

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model deepseek-coder --prompt_version v2 --eval_dataset ud_yue --segmentation_given True
POS Tagging Accuracy: 0.9182582123758595
Token Accuracy: 1.0
Token F1 Score: 1.0
Token Precision: 1.0
Token Recall: 1.0


(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model doubao-pro-32k --prompt_version v1 --eval_dataset ud_yue
POS Tagging Accuracy (Normalized): 0.8669783255418614
POS Tagging Accuracy: 0.858478538036549
Token Accuracy: 0.9671440606571188
Token F1 Score: 0.9245518966235932
Token Precision: 0.9342881213142376
Token Recall: 0.915016501650165

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model doubao-lite-32k --prompt_version v1 --eval_dataset ud_yue
POS Tagging Accuracy (Normalized): 0.8437190900098912
POS Tagging Accuracy: 0.8348170128585558
Token Accuracy: 0.9887535145267105
Token F1 Score: 0.8807339449541284
Token Precision: 0.8547328959700093
Token Recall: 0.9083665338645418

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model doubao-pro-32k --prompt_version v2 --eval_dataset ud_yue
POS Tagging Accuracy (Normalized): 0.8941009239516703
POS Tagging Accuracy: 0.8898365316275765
Token Accuracy: 0.9674220963172805
Token F1 Score: 0.9329608938547487
Token Precision: 0.9461756373937678
Token Recall: 0.9201101928374655


(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model doubao-lite-32k --prompt_version v2 --eval_dataset ud_yue
POS Tagging Accuracy (Normalized): 0.8508771929824561
POS Tagging Accuracy: 0.841130604288499
Token Accuracy: 0.9889400921658986
Token F1 Score: 0.8767253688719657
Token Precision: 0.8488479262672811
Token Recall: 0.906496062992126


(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model gpt-4o-mini --prompt_version v1 --eval_dataset ud_yue
POS Tagging Accuracy: 0.8630904723779024
Token Accuracy: 0.9770206022187005
Token F1 Score: 0.9328593996840443
Token Precision: 0.9358161648177497
Token Recall: 0.9299212598425197

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model gpt-4o-mini --prompt_version v2 --eval_dataset ud_yue
POS Tagging Accuracy: 0.8869286287089013
Token Accuracy: 0.9762658227848101
Token F1 Score: 0.9344909234411998
Token Precision: 0.9367088607594937
Token Recall: 0.9322834645669291

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model gpt-4o --prompt_version v1 --eval_dataset ud_yue
POS Tagging Accuracy: 0.8715740015661706
Token Accuracy: 0.9706336939721792
Token F1 Score: 0.9266231271609682
Token Precision: 0.9319938176197836
Token Recall: 0.9213139801375095

(wordshk) kevin@Kevins-MacBook-Pro-10 genai % python eval_pos_tagging.py --model gpt-4o --prompt_version v2 --eval_dataset ud_yue
POS Tagging Accuracy: 0.883430799220273
Token Accuracy: 0.9744384198295895
Token F1 Score: 0.9369230769230769
Token Precision: 0.9434546862896979
Token Recall: 0.93048128342246


# Sources

data/t2s.txt: https://github.com/tyrchen/fast2s/blob/0940d611d6cbf4fd79f096f8b681ef271e1c4573/src/t2s.txt

# Doubao SFT

这里的训练文本，不是简单的计算训练集的总 tokens 数，而是实际训练的文本总 tokens 数
精调预估 token 数公式如下：
tokens-per-batch（模型窗口长度）
batch-size（固定配置，现在线上大部分是 8，覆盖 doubao lite 和 pro 六个模型）
假设用户实际 token 数为 M（包括混数据之后的），tokens-per-batch * batch-size = N，则预估 token 数 = (int(M / N) + 1) * N
例如用户用 Doubao-lite-128k 进行训练，那么就会有一个最小启动成本：128 * 1024 * 8
前端根据后端返回 token 数 * epoch * 单价得到预估费用
