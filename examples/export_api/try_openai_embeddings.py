from openai import OpenAI
client = OpenAI()

# defs = [
#     "have a good time",
#     "good (美麗)",
#     "not good (欠佳)",
#     "(of business) not good (淡市)",
#     "good at (善)",
#     "all the time",
#     "this time (呢輪)",
#     "good night (早抖)",
#     "night time (夜麻麻)",
#     "time (時光)",
#     "in time (嚟得切)",
#     "what time (何時)",
#     "good morning (早安)",
#     "to have a good time, to have fun (happy)",
# ]

defs = [
    "everyone",
    "everybody",
    "everybody, often excluding the speaker when used without other personal pronouns",
    "everyone; without exception",
    "not everything; not everyone; it depends",
    "everyone has his own opinion; everyone talking at once; literally: seven mouths eight tongues",
    "as everyone knows; to be known to all",
    "everyone's hand",
    "the whole school; everyone from the school",
    "an expression meaning something is obvious or self-evident; no doubt; everyone knows this; literally: no need to ask Mr Gwai",
    "praised by everyone"
]

result = client.embeddings.create(
    model="text-embedding-ada-002",
    input=defs,
    encoding_format="float"
    )
embeddings = [emd.embedding for emd in result.data]

# Rank the cosine similarity between the first def and the rest
# and print out the ranked list of defs
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarities = cosine_similarity([embeddings[0]], embeddings[1:])[0]
ranked_indices = np.argsort(cosine_similarities)[::-1]
for i in ranked_indices:
    print(defs[i+1])
