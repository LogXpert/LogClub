


from abc import ABC, abstractmethod
import token
from typing import cast, Collection, IO, Iterable, List, MutableMapping, MutableSequence, Optional, Sequence, Tuple,\
    TYPE_CHECKING, TypeVar, Union
from enum import Enum
from cachetools import LRUCache, Cache
from itertools import groupby
import random
import string

from simple_profiler import Profiler, NullProfiler
from scope import ScopeBase
from collections import defaultdict
import logging
import logging.config
import re
import nltk
import en_core_web_md

import math
import json
import requests
import os
from openai import OpenAI
import spacy
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex
from modelscope import AutoTokenizer
import torch
import torch.nn as nn
import math
from collections import Counter
from postprocess import correct_single_template, correct_single_template_1
from sklearn.cluster import DBSCAN
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools
import time
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


nlp = en_core_web_md.load()


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

model_path = "/root/.cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L3-v2/snapshots/4ca70771034acceecb2e72475f72050fcdde4ddc"




model = SentenceTransformer(model_path)


NLTK_DATA_PATH = "/root/nltk_data"


if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)


def ensure_nltk_resource(resource_name):
    resource_path = os.path.join(NLTK_DATA_PATH, "corpora", resource_name)
    if not os.path.exists(resource_path):
        print(f"Resource '{resource_name}' not found locally. Downloading...")
        nltk.download(resource_name, download_dir=NLTK_DATA_PATH)
    else:
        print(f"Resource '{resource_name}' already exists locally. Loading...")


ensure_nltk_resource("wordnet")

prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
infix_re  = compile_infix_regex(nlp.Defaults.infixes)

pattern = r'\b\w+\([^)]+\)\b|\b\w+\[[^\]]+\]\b'

nlp.tokenizer = Tokenizer(
    nlp.vocab,
    rules=nlp.Defaults.tokenizer_exceptions,
    prefix_search=None,
    suffix_search=None,
    infix_finditer=None,
    token_match=re.compile(pattern).match
)

""" from flair.models import SequenceTagger
from flair.data import Sentence
logging.getLogger("flair").setLevel(logging.WARNING) """

special_words = ["<*>", "<PATH>", "<ID>", "<IP>", "<HEX>", "<DATE>", "<URL>", "<SLOT>", "<NUM>", "<CMD>", "<UNIT>", "-", "<L=3>", "<L=4>"]

for word in special_words:
    nlp.tokenizer.add_special_case(word, [{ORTH: word}])

"""
path = nltk.data.find('taggers/averaged_perceptron_tagger')
if path is None:
    exit(1)
else:
    print(path)
nltk.data.path.append(path)
#nltk.download('averaged_perceptron_tagger', download_dir=path)
"""










if os.path.exists("./pos.log"):
    os.rename("./pos.log", "./pos.log.prev")
pos_logger = logging.getLogger("pos_logger")



pos_handler = logging.FileHandler("./pos.log", mode="w")
pos_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
pos_logger.addHandler(pos_handler)
pos_logger.propagate = False

pos_logger.info("start datetime: %s", str(os.popen("date").read().strip()))



















SYMMETRIC_PAIRS = {


    '(': ')',

}
OPENING = set(SYMMETRIC_PAIRS.keys())
CLOSING = set(SYMMETRIC_PAIRS.values())

class LlmQA:
    """
    Class to store and manage LLM QA results for pairs of log message token lists.
    """

    def __init__(self):

        self.qAHistory = {}

    def checkQA(self, templateTokens: Iterable[str], newMessageTokens: Iterable[str]) -> tuple[bool, object]:
        """
        Check if the QA result for the given key exists.
        """
        key = (tuple(templateTokens), tuple(newMessageTokens))
        pos_logger.debug("checkQA: key: %s", key)
        if key in self.qAHistory:
            return True, self.qAHistory[key]
        return False, None

    def saveQA(self, templateTokens: Iterable[str], newMessageTokens: Iterable[str], result: object) -> None:
        """
        Save the QA information: only keep the latest QA result, replacing any previous one.
        """
        key = (tuple(templateTokens), tuple(newMessageTokens))
        pos_logger.debug("saveQA: key: %s, result: %s", key, result)
        self.qAHistory[key] = result

class Template():
    def __init__(self, templateTokens: Iterable[str] = "This is a default log template", templateId: int = 0, isPosSupported: bool = False, precomputedPosTag: Iterable[str] = None) -> None:
      self.templateId = templateId
      self.matchedLogSize = 1
      self.preemptedTokenSet = defaultdict(set)
      self.isPosSupported = isPosSupported
      self.setTemplate(templateTokens)
      self.setTokenPosTag(templateTokens, precomputedPosTag)
      self.isLLMCalculated = False

    def getTemplateStr(self) -> str:
        return ' '.join(self.templateTokens)

    def getTemplateTokens(self) -> str:
        return self.templateTokens

    def getPreemptedTokenSet(self) -> MutableMapping[int, set]:
        return self.preemptedTokenSet

    def setTemplate(self, templateTokens: Iterable[str]) -> None:
        self.templateTokens = templateTokens
        for index, token in enumerate(templateTokens):
            self.preemptedTokenSet[index].add(token)
            pos_logger.debug("preemptedTokenSet[{}]:{}".format(index, self.preemptedTokenSet[index]))

    def increaseMatchedLogSize(self) -> None:
        self.matchedLogSize += 1

    def getMatchedLogSize(self) -> int:
        return self.matchedLogSize

    def getTokenPosTag(self) -> MutableMapping[int, str]:
        return self.tokenPosTag

    def setTokenPosTag(self, templateTokens: Iterable[str], precomputedPosTag: Iterable[str] = None) -> None:
        if precomputedPosTag is not None:
            self.tokenPosTag = precomputedPosTag
        else:
            self.tokenPosTag = self.tokensPosTagger(templateTokens)

    def tokensPosTagger(self, tokens: Iterable[str]) -> Iterable[str]:
        if self.isPosSupported:
            lowercaseTokens = [element.lower() for element in tokens]
            posTagger = nltk.pos_tag(lowercaseTokens)
            tokenPosTag = [pos for token, pos in posTagger]
        else:
            tokenPosTag = ["unknown"] * len(tokens)
        pos_logger.debug("template token: %s", tokens)
        pos_logger.debug("tokenPosTag: %s", tokenPosTag)
        return tokenPosTag

    def setLLMCalculated(self, isLLMCalculated: bool) -> None:
        if isLLMCalculated:
            self.isLLMCalculated = isLLMCalculated

    def getLLMCalculated(self) -> bool:
        return self.isLLMCalculated

_T = TypeVar("_T")
if TYPE_CHECKING:
    class _LRUCache(LRUCache[int, Optional[Template]]):

        ...
else:
    _LRUCache = LRUCache

class LogClusterCache(_LRUCache):
    """
    Least Recently Used (LRU) cache which allows callers to conditionally skip
    cache eviction algorithm when accessing elements.
    """

    def __missing__(self, key: int) -> None:
        return None

    def get(self, key: int, _: Union[Optional[Template], _T] = None) -> Optional[Template]:
        """
        Returns the value of the item with the specified key without updating
        the cache eviction algorithm.
        """
        return Cache.__getitem__(self, key)

class NodeType(Enum):
    ROOT = 1
    STATICSEQ = 2
    INTERMEDIATE = 3
    LEAF = 4

class SequenceType(Enum):
    FORWARD = 1
    REVERSE = 2

class Node():
    __slots__ = ["nodeType", "keyToChildNode", "templateIds", "tokensInWildcard"]

    def __init__(self, nodeType: NodeType) -> None:
        self.nodeType: NodeType = nodeType
        self.keyToChildNode: MutableMapping[str, Node] = {}
        self.templateIds: Sequence[int] = set()
        self.tokensInWildcard = set()

class TemplateInfo():
    def __init__(self, templateId: int, logStaticSeqString: str, templateStr: str) -> None:
        self.logStaticSeqString = logStaticSeqString
        self.templateId = templateId
        self.templateStr = templateStr
        self.logIndexes = []
        self.matchedLogSize = 0

class LogClub(ScopeBase):
    def __init__(self,
                 depth: int = 4,
                 sim_th: float = 0.4,
                 max_children: int = 100,
                 max_clusters: Optional[int] = None,
                 extra_delimiters: Sequence[str] = (),
                 profiler: Profiler = NullProfiler(),
                 param_str: str = "<*>",
                 parametrize_numeric_tokens: bool = True,
                 bi_tree_support: bool = False,
                 template_pool_support: bool = False,
                 SLM_support: bool = False,
                 LLM_support: bool = False,
                 LLM_provider: str = "openai",
                 LLM_model: str = "gpt-3.5-turbo",
                 LLM_api_key: Optional[str] = None,
                 LLM_thinking: bool = False,
                 similarityMeasure: str = "lcs",
                 ngram_tokens: int = 3
                 ) -> None:
        """
        Create a new Scope instance.

        :param depth: max depth levels of log clusters. Minimum is 3.
            For example, for depth==4, Root is considered depth level 1.
            Token count is considered depth level 2.
            First log token is considered depth level 3.
            Log clusters below first token node are considered depth level 4.
        :param sim_th: similarity threshold - if percentage of similar tokens for a log message is below this
            number, a new log cluster will be created.
        :param max_children: max number of children of an internal node
        :param max_clusters: max number of tracked clusters (unlimited by default).
            When this number is reached, model starts replacing old clusters
            with a new ones according to the LRU policy.
        :param extra_delimiters: delimiters to apply when splitting log message into words (in addition to whitespace).
        :param parametrize_numeric_tokens: whether to treat tokens that contains at least one digit
            as template parameters.
        """
        if depth < 3:
            raise ValueError("depth argument must be at least 3")

        self.log_cluster_depth = depth
        self.max_node_depth = depth - 2
        self.sim_th = sim_th
        self.max_children = max_children
        self.root_node = Node(NodeType.ROOT)
        self.profiler = profiler
        self.extra_delimiters = extra_delimiters
        self.max_clusters = max_clusters
        self.param_str = param_str
        self.parametrize_numeric_tokens = parametrize_numeric_tokens
        self.bi_tree_support = bi_tree_support
        self.template_pool_support = template_pool_support
        self.SLM_support = SLM_support
        self.LLM_support = LLM_support
        self.LLM_provider = LLM_provider
        self.LLM_model = LLM_model
        self.LLM_api_key = LLM_api_key
        self.LLM_thinking = LLM_thinking
        self.ngram_tokens = ngram_tokens

        self.idToTemplateCluster: MutableMapping[int, Template] = {}
        self.lengthToTemplateIds = defaultdict(list)
        self.templateId = 0
        self.TplUpdByFwdTree = 0
        self.TplUpdByRevTree = 0
        self.TplUpdByPool = 0
        self.totalToken = 0
        self.lengthNodeCount = 0
        self.llmQAHistory = LlmQA()
        self.llmCallCnt = 0
        self.llmCallTokens = 0


        self.posSeqToTemplateInfos: MutableMapping[str, List[TemplateInfo]] = defaultdict(list)
        self.idToTemplateStr: MutableMapping[int, str] = {}
        self.idToTemplateInfos: MutableMapping[int, TemplateInfo] = {}

        self.logStaticSeqToTemplateInfos: MutableMapping[str, List[TemplateInfo]] = defaultdict(list)
        self.fwdTokenHashToTemplateIDs: MutableMapping[tuple[str, str, str], List[int]] = defaultdict(list)
        self.revTokenHashToTemplateIDs: MutableMapping[tuple[str, str, str], List[int]] = defaultdict(list)

        self.similarityMeasure = similarityMeasure

        self.templateMatchingTime = 0
        self.clusterGroupingTime = 0
        self.templateExtractionTime = 0
        self.templateSelfCorrectingTime = 0
        self.staticSeqNodeCount = 0
        self.lastChunk = False

        self.nltk_static_pos_tags = [
            "CC",

            "DT",
            "EX",
            "FW",
            "IN",



            "LS",
            "MD",




            "PDT",
            "POS",
            "PRP",
            "PRP$",
            "RB",
            "RBR",
            "RBS",
            "RP",
            "SYM",
            "TO",
            "UH",
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",




        ]

        self.spacy_static_pos_tags = [

            "VERB",
            "ADJ",
            "ADV",
            "PRON",
            "DET",
            "ADP",

            "CCONJ",
            "SCONJ",
            "PART",
            "AUX",
            "INTJ",

            "PUNCT",
            "SYM",

        ]


    def getNewTemplateId(self) -> int:
        self.templateId += 1
        return self.templateId

    @property
    def clusters(self) -> Collection[Template]:

        return cast(Collection[Template], self.idToTemplateInfos.values())

    @property
    def lengthNodeCounts(self) -> int:
        return self.lengthNodeCount

    @property
    def templateMatchingDuration(self) -> float:
        return self.templateMatchingTime

    @property
    def clusterGroupingDuration(self) -> float:
        return self.clusterGroupingTime

    @property
    def templateExtractionDuration(self) -> float:
        return self.templateExtractionTime

    @property
    def templateSelfCorrectingDuration(self) -> float:
        return self.templateSelfCorrectingTime

    @property
    def has_numbers(s: Iterable[str]) -> bool:
        return any(char.isdigit() for char in s)

    def split_string(self, s):

        s = s.replace("-", "__DASH__")

        punctuation_pattern = r'^[,.\!?;:]+|[,.\!?;:]+$'


        match = re.match(punctuation_pattern, s)
        if match:

            front_punctuation = match.group(0)
        else:
            front_punctuation = ''


        stripped_s = re.sub(r'^[,.\!?;:]+', '', s)


        match = re.search(punctuation_pattern, stripped_s)
        if match:

            back_punctuation = match.group(0)

            stripped_s = re.sub(r'[,.\!?;:]+$', '', stripped_s)
        else:
            back_punctuation = ''


        result = []


        if front_punctuation:

            i = 0
            while i < len(front_punctuation):
                char = front_punctuation[i]

                j = i
                while j < len(front_punctuation) and front_punctuation[j] == char:
                    j += 1

                result.append(char * (j - i))
                i = j


        if ('=' in stripped_s) and (':' in stripped_s):

            if stripped_s.index('=') < stripped_s.index(':'):
                left, sep1, remainder = re.split(r"(=)", stripped_s, 1)
                mid, sep2, right = re.split(r"(:)", remainder, 1)
            else:
                left, sep1, remainder = re.split(r"(:)", stripped_s, 1)
                mid, sep2, right = re.split(r"(=)", remainder, 1)
            result.extend([p for p in [left, sep1, mid, sep2, right] if p])


        elif stripped_s.count('=') == 1 or (stripped_s.count('=') > 1 and not "==" in stripped_s):
            left, sep, right = re.split(r"(=)", stripped_s, 1)
            result.extend([p for p in [left, sep, right] if p])


        elif stripped_s.count(':') == 1 or (stripped_s.count(':') > 1 and not "::" in stripped_s):

            left, sep, right = re.split(r"(:)", stripped_s, 1)
            result.extend([p for p in [left, sep, right] if p])


        elif stripped_s.count('+') == 1 or (stripped_s.count('+') > 1 and not "++" in stripped_s):

            left, sep, right = re.split(r"(\+)", stripped_s, 1)
            result.extend([p for p in [left, sep, right] if p])


        else:
            if stripped_s:
                result.append(stripped_s)


        if back_punctuation:

            i = 0
            while i < len(back_punctuation):
                char = back_punctuation[i]

                j = i
                while j < len(back_punctuation) and back_punctuation[j] == char:
                    j += 1

                result.append(char * (j - i))
                i = j

        tokens = []
        for part in result:
            if (
                len(part) > 1
                and part[0] in SYMMETRIC_PAIRS and part[-1] == SYMMETRIC_PAIRS[part[0]]):
                tokens.append(part[0])
                tokens.append(part[1:-1])
                tokens.append(part[-1])
            elif (len(part) > 1
                and part[0] in SYMMETRIC_PAIRS and not SYMMETRIC_PAIRS[part[0]] in part):
                tokens.append(part[0])
                tokens.append(part[1:])
            elif (len(part) > 1
                and part[-1] in CLOSING and\
                not any(key for key in SYMMETRIC_PAIRS if SYMMETRIC_PAIRS[key] == part[-1] and key in part[0:-1])):
                tokens.append(part[:-1])
                tokens.append(part[-1])
            elif len(part) > 2:
                prefixIdx = suffixIdx = 0
                splitDone = False
                i = 1
                while i < len(part):

                    if part[i] in SYMMETRIC_PAIRS:

                        prefixIdx = i

                        close_sym = SYMMETRIC_PAIRS[part[i]]

                        for k in range(prefixIdx+1, len(part)):
                            if part[k] == close_sym:
                                suffixIdx = k
                                if k == prefixIdx + 1:
                                    splitDone = True
                                    tokens.append(part)
                                    break
                                tokens.append(part[0:prefixIdx])
                                tokens.append(part[prefixIdx])
                                tokens.append(part[prefixIdx+1:suffixIdx])
                                tokens.append(part[suffixIdx])
                                tokens.append(part[suffixIdx+1:])
                                splitDone = True
                                break
                        if splitDone == False:

                            tokens.append(part[0:prefixIdx+1])
                            tokens.append(part[prefixIdx+1:])
                            splitDone = True
                    elif part[i] in CLOSING:
                        tokens.append(part[0:i])
                        tokens.append(part[i])
                        tokens.append(part[i+1:])
                        splitDone = True

                    if splitDone == True:
                        break
                    else:
                        i += 1
                if not splitDone:

                    tokens.append(part)
            else:
                tokens.append(part)

        pos_logger.debug("original string: %s, split_string result: %s", s, tokens)
        return tokens

    def addLogMsgInNgramNodeTreeForCluster(self, logIndex: int, stiaticSeqString: str, logContent: str) -> None:
        class NodeType(Enum):
            ROOT = 1
            STATICSEQ = 2
            INTERMEDIATE = 3
            LEAF = 4
        from typing import Tuple

        class Node():
            __slots__ = ["nodeType", "keyToChildNode", "logMessages", "staticSeqString"]

            def __init__(self, nodeType: NodeType) -> None:
                self.nodeType: NodeType = nodeType
                self.keyToChildNode: MutableMapping[Tuple[str, str, str], Node] = {}
                self.logMessages: list[Tuple[int, str]] = []
                self.staticSeqString: str = ""

        if stiaticSeqString not in self.root_node.keyToChildNode:
            staticSeqNode = Node(NodeType.STATICSEQ)
            self.root_node.keyToChildNode[stiaticSeqString] = staticSeqNode
            pos_logger.debug("create STATICSEQ node: %s", stiaticSeqString)
            self.staticSeqNodeCount += 1
        else:
            staticSeqNode = self.root_node.keyToChildNode[stiaticSeqString]
            pos_logger.debug("STATICSEQ node: %s is existed", stiaticSeqString)

        currentNode = staticSeqNode
        logTokens = logContent.split()
        tokenCount = len(logTokens)

        if tokenCount == 0:
            currentNode.logMessages = [(logIndex, logContent)]
            currentNode.nodeType = NodeType.LEAF
            return


        window_size = 3
        current_depth = 1

        max_len = len(logTokens) // 2
        if len(logTokens) % 2 != 0:
            max_len += 1
        processing_len = max_len

        for i in range(processing_len):

            nGramNodeTuple = tuple(logTokens[i:i+window_size])

            while len(nGramNodeTuple) < window_size:
                nGramNodeTuple = nGramNodeTuple + (None,)





            found_key = None
            if nGramNodeTuple in currentNode.keyToChildNode:
                currentNode = currentNode.keyToChildNode[nGramNodeTuple]
                found_key = nGramNodeTuple
            else:

                for key in currentNode.keyToChildNode:

                    count_equal = 0
                    non_none_positions = []

                    for j in range(len(nGramNodeTuple)):
                        if nGramNodeTuple[j] is not None:
                            non_none_positions.append(j)

                    if len(non_none_positions) == 1:
                        pos = non_none_positions[0]
                        if nGramNodeTuple[pos] == key[pos]:
                            found_key = key
                            break
                    elif len(non_none_positions) == 2:

                        pos1, pos2 = non_none_positions
                        match_cnt = (nGramNodeTuple[pos1] == key[pos1]) + (nGramNodeTuple[pos2] == key[pos2])
                        if match_cnt >= 1:
                            found_key = key
                            break
                    else:

                        match_cnt = sum(
                            nGramNodeTuple[j] == key[j] or nGramNodeTuple[j] == '<*>' or key[j] == '<*>' for j in range(window_size)

                        )

                        match_cnt_non_wildcard = sum(
                            (nGramNodeTuple[j] != '<*>' and key[j] != '<*>' and nGramNodeTuple[j] == key[j])
                            for j in range(window_size)
                        )
                        def isConsecutiveWildcast(a, b):


                            A = set()
                            B = set()
                            for i in range(len(a)):
                                if a[i] == '<*>' or b[i] == '<*>':
                                    A.add(i)
                                if a[i] != '<*>' and b[i] != '<*>'and a[i] != b[i]:
                                    B.add(i)


                            for b in B:
                                if (b - 1) in A or (b + 1) in A:
                                    return True
                            return False

                        if match_cnt >= 2 and match_cnt_non_wildcard >= 1 and not isConsecutiveWildcast(nGramNodeTuple, key):
                            found_key = key
                            break
                        else:
                            pos_logger.debug("No match to key: %s. match_cnt: %d, match_cnt_non_wildcard: %s, isConsecutiveWildcast: %s", key, match_cnt, match_cnt_non_wildcard, isConsecutiveWildcast(nGramNodeTuple, key))

                if found_key is not None:
                    currentNode = currentNode.keyToChildNode[found_key]
                else:

                    newNode = Node(NodeType.INTERMEDIATE)
                    currentNode.keyToChildNode[nGramNodeTuple] = newNode
                    currentNode = newNode

            pos_logger.debug("key found: %s, add nGramNodeTuple: %s to key: %s", True if found_key is not None else False, nGramNodeTuple, found_key)
            current_depth += 1

            if current_depth >= self.max_node_depth or current_depth > processing_len:

                currentNode.logMessages.append((logIndex, logContent))
                currentNode.staticSeqString = "¦¦¦".join(stiaticSeqString.split("¦¦¦")[0:2])
                pos_logger.debug("xxxxxxcurrentNode.staticSeqString: %s", currentNode.staticSeqString)
                currentNode.nodeType = NodeType.LEAF
                break

    def isSnakeMode(self, token: str) -> bool:
        """
        Check if the token is in snake_case format.
        """
        return bool(re.match(r'^[a-zA-Z]+(_[a-zA-Z]+)+$', token))

    def isCamelMode(self, token: str) -> bool:
        """
        Check if the token is in camelCase format.
        """




        if not re.match(r'^[a-zA-Z]+([A-Z][a-z0-9]+)+(\(\))?$', token):
            return False


        parts = re.findall(r'[a-z]+|[A-Z][a-z0-9]*', token)
        if len(parts) < 2:
            return False


        for p in parts:
            if len(p) > 1 and not any(ch.isdigit() for ch in p) and nlp.vocab.has_vector(p):
                pos_logger.debug(f"Part '{p}' in token '{token}' is in vocabulary.")
                return True

        return False

    def is_snake_or_camel(self, identifier):
        snake_case_pattern = r'^[a-z]+(_[a-z]+)+$'
        camel_case_pattern = r'^[a-zA-Z]+([A-Z][a-z0-9]+)+$'
        return bool(re.match(snake_case_pattern, identifier) or re.match(camel_case_pattern, identifier))


    def isAllAlphaCapital(self, token: str) -> bool:
        if token == "<*>":
            return False

        if any(ch.isdigit() for ch in token):
            return False

        return all(char.isupper() for char in token if char.isalpha())


    def addTemplateToHashTables(self, templateInfo: TemplateInfo) -> None:
        """
        Add the templateInfo to the forward (fwd) and reverse (rev) hash tables based on its templateStr.

        The fwdTokenHashToTemplateIDs uses a key of the first 3 tokens as a tuple, with templateId as value.
        The revTokenHashToTemplateIDs uses a key of the last 3 tokens as a tuple, with templateId as value.
        """
        tokens = templateInfo.templateStr.strip().split()


        if len(tokens) >= 3:
            fwd_key = tuple(tokens[:3])
        else:
            fwd_key = tuple(tokens[i] if i < len(tokens) else "¦¦¦" for i in range(3))
        self.fwdTokenHashToTemplateIDs[fwd_key].append(templateInfo.templateId)


        if len(tokens) >= 3:
            rev_key = tuple(tokens[-3:])
        else:
            rev_key = tuple(tokens[i] if i < len(tokens) else "¦¦¦" for i in range(3))
        self.revTokenHashToTemplateIDs[rev_key].append(templateInfo.templateId)

    def removeTemplateFromHashTables(self, templateInfo: TemplateInfo) -> None:
        """
        Remove the templateInfo from the forward (fwd) and reverse (rev) hash tables based on its templateStr.
        Safe against key mismatch (e.g. after templateStr was modified by correct_single_template).
        """
        tokens = templateInfo.templateStr.strip().split()

        if len(tokens) >= 3:
            fwd_key = tuple(tokens[:3])
        else:
            fwd_key = tuple(tokens[i] if i < len(tokens) else "¦¦¦" for i in range(3))
        if templateInfo.templateId in self.fwdTokenHashToTemplateIDs[fwd_key]:
            self.fwdTokenHashToTemplateIDs[fwd_key].remove(templateInfo.templateId)

        if len(tokens) >= 3:
            rev_key = tuple(tokens[-3:])
        else:
            rev_key = tuple(tokens[i] if i < len(tokens) else "¦¦¦" for i in range(3))
        if templateInfo.templateId in self.revTokenHashToTemplateIDs[rev_key]:
            self.revTokenHashToTemplateIDs[rev_key].remove(templateInfo.templateId)

    def tokenize1(self, log_content, tokenize_pattern=r'[ ,|]', removeDight=True):
        words = re.split(tokenize_pattern, log_content)
        new_words = []
        for word in words:
            if 0 and '=' in word:
                ws = word.split('=')
                if len(ws) <= 2:
                    new_words.append(ws[0])
                else:

                    pass

            elif removeDight and re.search(r'\d', word):
                pass
            elif '/' in word.lower() or re.match(r"^[a-zA-Z][+-]$|^[+-][a-zA-Z]$", word):
                pass
            else:
                word = re.sub(r"\([^)]*\)", "", word)
                new_words.append(word)
        new_words = [word for word in new_words if word]
        if new_words == []:
            new_words.append(re.sub(r'\d+(\.\d+)?', '0', log_content))

        return new_words

    def parseChunkLogs(self, logBase: int, chunkSize: int, chunkLogs: list, lastChunk: bool):
        """
        Parse a chunk of logs and return the template IDs and their string representations.
        :param chunkLogs: List of log messages in the chunk.
        :return: Tuple containing a list of template IDs and a list of template strings.
        """
        self.lastChunk = lastChunk

        def clear_tree(node):

            if hasattr(node, 'keyToChildNode'):
                for child in list(node.keyToChildNode.values()):
                    clear_tree(child)
                node.keyToChildNode.clear()

            if hasattr(node, 'logIndexes'):
                node.logIndexes.clear()
            if hasattr(node, 'templateInfo'):
                node.templateInfo = None
            if hasattr(node, 'tokensCount'):
                node.tokensCount = 0

        logToParseClusters = defaultdict(list)






        for logIndex, logInfo in enumerate(chunkLogs):
            templateMatchingStartTime = time.perf_counter()
            logIndex = logBase * chunkSize + logIndex
            pos_logger.info(f"========input log({logIndex}) is============: {logInfo}")
            logComponent = logInfo.split("¦¦¦")[0].strip()
            logLevel = logInfo.split("¦¦¦")[1].strip()
            logContent = logInfo.split("¦¦¦")[2].strip()
            logContent = correct_single_template_1(logContent)

            logTokens, tokenPosTag = self.extractTokensOfMsg1(logContent)

            logContent = " ".join(logTokens)

            logTokens = logContent.split()
            pos_logger.debug(f"logComponent: {logComponent}; logLevel: {logLevel}; logTokens: {logTokens}")



            templateId = self.findMatchedTemplateFromCache(logComponent + "¦¦¦" + logLevel, logContent)

            templateMatchingEndTime = time.perf_counter()
            self.templateMatchingTime += (templateMatchingEndTime - templateMatchingStartTime)
            if templateId is not None:
                pos_logger.info(f"Matched template ID: {templateId} for log content: {logContent}")


                self.idToTemplateInfos[templateId].matchedLogSize += 1
                self.idToTemplateInfos[templateId].logIndexes.append(logIndex)
                continue
            else:


                clusterGroupingStartTime = time.perf_counter()

                def length_to_group(length: int) -> str:
                    if length <= 3:
                        return f"G {length}"
                    else:
                        return f"G {(length - 4) // 1 + 4}"

                lengthGroup = length_to_group(len(logTokens))
                staticTokensSequence = logComponent + "¦¦¦" + logLevel + "¦¦¦" + lengthGroup

                self.addLogMsgInNgramNodeTreeForCluster(logIndex, staticTokensSequence, logContent)
                clusterGroupingEndTime = time.perf_counter()
                self.clusterGroupingTime += (clusterGroupingEndTime - clusterGroupingStartTime)


        def collect_leaf_nodes_from_root(root_node):
            def extract_token_count_from_static_key(key):
                if isinstance(key, str) and "¦¦¦" in key:
                    token_count_str = key.split("¦¦¦")[-1].split(" ")[-1]
                    try:
                        return int(token_count_str)
                    except ValueError:
                        return 0
                return 0

            leaf_nodes = []
            idx = 0
            def dfs(node, parent_node):
                nonlocal idx
                if hasattr(node, 'nodeType') and node.nodeType.name == "LEAF":
                    logMsgContents = [logMessage[1] for logMessage in node.logMessages]
                    logTypeNum = len(set(logMsgContents))
                    if logTypeNum > 2 or len(node.logMessages) > 100 or self.lastChunk == True:
                        pos_logger.info(f"Collect Leaf node:{idx}:{logMsgContents[0]} has {logTypeNum} different log types and length: {len(node.logMessages)}.")
                        for i, log in enumerate(set(logMsgContents)):
                            pos_logger.debug(f"{i}:{log}")
                        idx += 1
                        leaf_nodes.append(node)
                        if parent_node is not None:

                            if parent_node is not None and hasattr(parent_node, 'keyToChildNode'):
                                keys_to_remove = [k for k, v in parent_node.keyToChildNode.items() if v is node]
                                for k in keys_to_remove:
                                    del parent_node.keyToChildNode[k]
                child_nodes = getattr(node, 'keyToChildNode', {})

                for key, child in sorted(
                    child_nodes.items(),
                    key=lambda item: extract_token_count_from_static_key(item[0]),
                    reverse=True
                ):
                    dfs(child, node)
            dfs(root_node, None)
            return leaf_nodes


        all_leaf_nodes = collect_leaf_nodes_from_root(self.root_node)
        pos_logger.info(f"Total leaf nodes found in prefix tree: {len(all_leaf_nodes)}")
        templateExtractionStartTime = time.perf_counter()
        clusterGroupingStartTime = time.perf_counter()
        logClusters = []

        semanticClustering = False
        posTokenVarianceComparison = True

        for idx, leaf_node in enumerate(all_leaf_nodes):
            pos_logger.info(f"Leaf node:{idx}: {leaf_node.logMessages[0][0]}:{leaf_node.logMessages[0][1]}")
            logIndexContents = leaf_node.logMessages
            logStaticSeqString = leaf_node.staticSeqString
            if semanticClustering:
                logSemanticsClusters = self.generateClustersFromSemanticSimilarity(logIndexContents)
                if len(logSemanticsClusters) > 1:
                    pos_logger.info(f"Semantic clustering of leaf node produced {len(logSemanticsClusters)} clusters.")
                for idx, cluster in enumerate(logSemanticsClusters):
                    logContentsOfsameCluster = [logIndexContents[i][1] for i in cluster]
                    logIndexesOfsameCluster = [logIndexContents[i][0] for i in cluster]
                    logClusters.append((logStaticSeqString, list(zip(logIndexesOfsameCluster, logContentsOfsameCluster))))
            elif posTokenVarianceComparison:
                logPosTokenCompareClusters = self.generateClustersFromPosTokenVarianceComparison(logIndexContents)
                if len(logPosTokenCompareClusters) > 1:
                    pos_logger.info(f"Pos token variance comparison of leaf node produced {len(logPosTokenCompareClusters)} clusters.")
                for idx, cluster in enumerate(logPosTokenCompareClusters):
                    logContentsOfsameCluster = [logIndexContents[i][1] for i in cluster]
                    logIndexesOfsameCluster = [logIndexContents[i][0] for i in cluster]
                    logClusters.append((logStaticSeqString, list(zip(logIndexesOfsameCluster, logContentsOfsameCluster))))
            else:
                logClusters.append((logStaticSeqString, logIndexContents))
        pos_logger.info(f"Total log clusters from leaf nodes: {len(logClusters)} with semanticClustering: {semanticClustering}")
        clusterGroupingEndTime = time.perf_counter()
        self.clusterGroupingTime += (clusterGroupingEndTime - clusterGroupingStartTime)




        LLMOneCall = True
        if LLMOneCall == True:
            if logClusters:
                self.extractAllTemplatesByOneLLMCall(logClusters)
            else:
                pos_logger.info(f"No log clusters found from leaf nodes, skip template extraction.")

            if self.lastChunk == True:
                for templateInfo in self.idToTemplateInfos.values():
                    templateInfo.templateStr = correct_single_template(templateInfo.templateStr)
                self.mergeDuplicateTemplateStrInIdToTemplateInfos()
            return



    def mergeDuplicateTemplateStrInIdToTemplateInfos(self) -> None:
        """
        If multiple TemplateInfo in idToTemplateInfos share the same templateStr,
        merge them into one item: combine logIndexes, keep one templateStr, remove duplicates.
        """
        strToInfos = defaultdict(list)
        for info in list(self.idToTemplateInfos.values()):
            key = info.templateStr.strip() if info.templateStr else ""
            strToInfos[key].append(info)

        for templateStrKey, infos in strToInfos.items():
            if len(infos) <= 1:
                continue
            mergedLogIndexes = []
            logStaticSeqString = infos[0].logStaticSeqString
            for info in infos:
                mergedLogIndexes.extend(info.logIndexes)
                self.removeTemplateFromHashTables(info)
                del self.idToTemplateInfos[info.templateId]
            newInfo = self.createNewTemplateInfo(logStaticSeqString, templateStrKey, mergedLogIndexes)
            self.idToTemplateInfos[newInfo.templateId] = newInfo
            self.addTemplateToHashTables(newInfo)
            pos_logger.debug(f"Merged {len(infos)} duplicate templateStr into one: ID {newInfo.templateId}, templateStr: {templateStrKey}, logIndexes count {len(mergedLogIndexes)}")

    def collectCanMergeTemplateInfos(self, templateStr: str) -> list[TemplateInfo]:
        """
        Collect all TemplateInfo objects that can be merged with the given templateStr.
        :param templateStr: The template string to check for merging.
        :return: List of TemplateInfo objects that can be merged with the given templateStr.
        """
        canMergeTemplateInfos = []
        sortedTemplates = self.getCandidateTemplatesFromCache(templateStr)
        for templateInfo in sortedTemplates:



            pos_logger.debug(f"collectCanMergeTemplateInfos: check with existing template: ID:{templateInfo.templateId}, Info:{templateInfo.templateStr}")
            oldTokens = templateInfo.templateStr.split()
            newTokens = templateStr.split()
            matchNeeded = False
            for i, tmpl_token in enumerate(oldTokens):

                if tmpl_token != "<*>":


                    if i < len(newTokens) and newTokens[i] == tmpl_token:
                        matchNeeded = True
                    else:
                        matchNeeded = False
                    break
            if not matchNeeded:
                continue
            """
            if len(templateStr) < len(templateInfo.templateStr): #HPC <*> ok <*> ok.....
                if templateStr in templateInfo.templateStr and newTokens[-1] == oldTokens[-1]: #not so common
                    canMergeTemplateInfos.append(templateInfo)
                    continue
            elif len(templateStr) > len(templateInfo.templateStr):
                if templateInfo.templateStr in templateStr and newTokens[-1] == oldTokens[-1]:
                    canMergeTemplateInfos.append(templateInfo)
                    continue
            """
            if abs(len(oldTokens) - len(newTokens)) > 3:

                continue
            elif len(oldTokens) == len(newTokens):

                mismatchTokenNum = 0
                for oldToken, newToken in zip(oldTokens, newTokens):
                    if oldToken != newToken and oldToken != "<*>" and newToken != "<*>":

                        mismatchTokenNum += 1
                else:

                    if mismatchTokenNum < 1:
                        pos_logger.debug(f"existing TemplateStr: {templateInfo.templateStr} can be merged with new templateStr: {templateStr}")
                        canMergeTemplateInfos.append(templateInfo)


            elif '<*>' in templateStr or '<*>' in templateInfo.templateStr:
                pos_logger.debug(f"check new template: {templateStr} with existing template: ID:{templateInfo.templateId}, Info:{templateInfo.templateStr}")
                if self.isTemplateMachedWithWildcard(templateInfo.templateStr, templateStr) or self.isTemplateMachedWithWildcard(templateStr, templateInfo.templateStr):
                    pos_logger.debug(f"Existing TemplateStr: {templateInfo.templateStr} matches with new templateStr: {templateStr}")
                    canMergeTemplateInfos.append(templateInfo)
                """
                else:
                    similarityScore = self.calculateSimilarity(templateInfo.templateStr.split(), templateStr.split())
                    #length = max(len(templateInfo.templateStr.split()), len(templateStr.split()))
                    #similarityScoreThres = (length - math.log(length, 2)) / (length + math.log(length, 2))
                    similarityScoreThres = 0.9
                    pos_logger.debug(f"similarityScore: {similarityScore}, similarityScoreThres: {similarityScoreThres}")
                    if similarityScore >= similarityScoreThres:
                        pos_logger.debug(f"Existing TemplateStr: {templateInfo.templateStr} matches with new templateStr: {templateStr} with similarity score: {similarityScore}")
                        canMergeTemplateInfos.append(templateInfo)
                """
            """
            else:
                similarityScore = self.calculateSimilarity(templateInfo.templateStr.split(), templateStr.split())
                length = max(len(templateInfo.templateStr.split()), len(templateStr.split()))
                #similarityScoreThres = (length - math.log(length, 2)) / (length + math.log(length, 2))
                similarityScoreThres = (length - 1) / length
                #similarityScoreThres = 0.9
                pos_logger.debug(f"similarityScore: {similarityScore}, similarityScoreThres: {similarityScoreThres}")
                if similarityScore >= similarityScoreThres:
                    pos_logger.debug(f"Existing TemplateStr: {templateInfo.templateStr} matches with new templateStr: {templateStr} with similarity score: {similarityScore}")
                    canMergeTemplateInfos.append(templateInfo)
            """


        return canMergeTemplateInfos

    def generateClustersFromPosTokenVarianceComparison(self, logIndexContents: list[tuple[int, str]]) -> list[list[int]]:
        """
        Generate clusters of logs based on pos token variance comparison.
        :param logIndexContents: List of tuples containing log index and log content.
        :return: List of clusters, where each cluster is a list of log indexes.
        """
        if len(logIndexContents) == 0:
            return []
        log_contents = [log_content for _, log_content in logIndexContents]
        if len(log_contents) == 1 or len(set(log_contents)) == 1:
            pos_logger.debug("All logs have identical content, returning single cluster.")
            return [[idx for idx in range(len(logIndexContents))]]

        log_tokens_list = [log.split() for log in log_contents]

        max_len = max(len(toks) for toks in log_tokens_list)


        def filter_columns_by_token(log_tokens_list):
            """
            Filters out columns (token positions) where any token should be removed by tokenize1.
            Returns filtered_logs: list of lists (tokens per line for each log).
            """




            max_cols = max(len(row) for row in log_tokens_list)
            num_logs = len(log_tokens_list)


            padded = [row + [None]*(max_cols - len(row)) for row in log_tokens_list]

            columns = [[padded[row_idx][col_idx] for row_idx in range(num_logs)] for col_idx in range(max_cols)]


            def is_removed(token, tokenize1_fn):

                if token is None:
                    return False
                out = tokenize1_fn(token)

                return token not in out


            columns_to_keep = []
            for col_idx, col in enumerate(columns):
                non_none = [t for t in col if t is not None]
                if all(self.tokenize1(t) == [t] for t in non_none):
                    columns_to_keep.append(col_idx)
            return columns_to_keep


        columns_to_keep = filter_columns_by_token(log_tokens_list)


        pos_token_counts = defaultdict(Counter)
        max_len = max(len(toks) for toks in log_tokens_list)
        for toks in log_tokens_list:

            for i, tok in enumerate(toks):
                pos_token_counts[i][tok] += 1



        pos_entropy_norm, pos_types_norm = [], []


        max_types = 5


        for pos, counter in pos_token_counts.items():
            if pos not in columns_to_keep:
                pos_entropy_norm.append(1)
                pos_types_norm.append(1)
                continue
            if '<*>' in counter:
                pos_entropy_norm.append(1)
                pos_types_norm.append(1)
                continue
            total = sum(counter.values())
            max_count = max(counter.values()) if counter else 1
            repeat_ratio = max_count / total if total > 0 else 0
            probs = [c / total for c in counter.values()]

            H = -sum(p * math.log(p) for p in probs)
            Vp = len(counter)
            H_max = math.log(Vp) if Vp > 1 else 1.0
            H_norm = H / H_max if H_max > 0 else 0
            Vp_norm = min((Vp - 1) / (max_types - 1), 1)
            pos_entropy_norm.append(H_norm)
            pos_types_norm.append(Vp_norm)

        weight_pos_dict = defaultdict(list)
        weight_list = []
        pos_list = []
        for idx, Hn, Tn in zip(pos_token_counts.keys(), pos_entropy_norm, pos_types_norm):
            weight = Hn * Tn
            weight_list.append(weight)
            pos_logger.info(f"pos={idx}, weight={weight:.4f}")


        threshold= 0.25



        for idx, weight in enumerate(weight_list):
            if weight <= threshold:
                weight_pos_dict[weight].append(idx)

        if len(weight_pos_dict) < 1:
            return [[idx for idx in range(len(logIndexContents))]]
        else:




            zero_pos = weight_pos_dict.get(0, [])

            nonzero_weights = [w for w in weight_pos_dict.keys() if w != 0]
            min_nonzero_pos = []
            if nonzero_weights:
                min_weight = min(nonzero_weights)
                min_nonzero_pos = weight_pos_dict[min_weight]
            pos_list.extend(zero_pos)
            pos_list.extend(min_nonzero_pos)

        pos_list.sort()


        clusters = {}
        for idx, tokens in enumerate(log_tokens_list):
            key = tuple(tokens[pos] if pos < len(tokens) else "<EMPTY>" for pos in pos_list)
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(idx)
        pos_logger.info(f"Split into {len(clusters)} clusters")
        for idx, (key, value) in enumerate(clusters.items()):
            pos_logger.info(f"{idx}:Cluster {key}: {log_tokens_list[value[0]]}")

        return list(clusters.values())



        pos_types_non_one = [x for x in pos_types if x != 1]
        if pos_types_non_one and max(pos_types_non_one) / min(pos_types_non_one) > 3:
            min_change_pos = pos_types.index(min(pos_types_non_one))


            clusters = {}
            for idx, tokens in enumerate(tokenized_log_tokens):
                if len(tokens) > min_change_pos:
                    key = tokens[min_change_pos]
                else:
                    key = "<EMPTY>"
                if key not in clusters:
                    clusters[key] = []
                clusters[key].append(idx)

            return list(clusters.values())
        else:
            return [[idx for idx in range(len(logIndexContents))]]



    def generateClustersFromSemanticSimilarity(self, logIndexContents: list[tuple[int, str]]) -> list[list[int]]:
        """
        Generate clusters of logs based on semantic similarity using embeddings.
        :param logIndexContents: List of tuples containing log index and log content.
        :return: List of clusters, where each cluster is a list of log indexes.
        """

        if len(logIndexContents) == 0:
            return []





        log_contents = [log_content for _, log_content in logIndexContents]
        if len(log_contents) == 1 or len(set(log_contents)) == 1:
            pos_logger.debug("All logs have identical content, returning single cluster.")
            return [[idx for idx in range(len(logIndexContents))]]



        log_contents = [log_content for _, log_content in logIndexContents]






        solution = 'Entropy+WeightedTF-IDF'

        if solution == 'Bert':

            v_sbert = model.encode(log_contents, convert_to_numpy=True)


            normalized_embeddings = v_sbert / np.linalg.norm(v_sbert, axis=1, keepdims=True)


        elif solution == 'Bert+TF-IDF':
            v_sbert = model.encode(log_contents, convert_to_numpy=True)
            v_sbert = normalize(v_sbert, norm='l2', axis=1)


            vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1,1), min_df=1, stop_words=None, token_pattern=r"(?u)<\*>|\b\w+\b")
            X_tfidf = vectorizer.fit_transform(log_contents)
            n_features = X_tfidf.shape[1]

            v_tfidf = X_tfidf.toarray()

            v_tfidf = normalize(v_tfidf, axis=1)


            w_s, w_t = 1, 0.8
            v_concat = np.hstack([w_s * v_sbert, w_t * v_tfidf])
            v_concat = normalize(v_concat, norm='l2', axis=1)

            normalized_embeddings = v_concat

        elif solution == 'IDF':

            tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words=None, token_pattern=r"(?u)<\*>|\b\w+\b")
            X_tfidf = tfidf.fit_transform(log_contents)
            n_features = X_tfidf.shape[1]

            v_tfidf = X_tfidf.toarray()

            v_tfidf = normalize(v_tfidf, axis=1)

            normalized_embeddings = v_tfidf

        elif solution == 'WeightedTF-IDF':

            tokenized = [log.split() for log in log_contents]
            max_len = max(len(t) for t in tokenized)



            pos_types = []
            for pos in range(max_len):
                tokens_at_pos = [log[pos] for log in tokenized if len(log) > pos]
                pos_types.append(len(set(tokens_at_pos)))



            vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1,1), min_df=1, stop_words=None, token_pattern=r"(?u)<\*>|\b\w+\b")
            X_tfidf = vectorizer.fit_transform(log_contents).toarray()
            vocab = vectorizer.get_feature_names_out()
            vocab_index = {t: i for i, t in enumerate(vocab)}



            X_weighted = np.copy(X_tfidf)
            for log_idx, tokens in enumerate(tokenized):
                for pos, token in enumerate(tokens):
                    if token in vocab_index:
                        i = vocab_index[token]

                        safe_pos_type = max(pos_types[pos], 2)
                        if pos > 0 and tokens[pos-1] in [":", "="]:
                            w_pos = 0
                        else:
                            w_pos = 1 / math.log(safe_pos_type)
                        X_weighted[log_idx, i] *= w_pos


            normalized_embeddings = X_weighted


        elif solution == 'Bert+WeightedTF-IDF':
            v_sbert = model.encode(log_contents, convert_to_numpy=True)
            v_sbert = normalize(v_sbert, norm='l2', axis=1)


            tokenized = [log.split() for log in log_contents]
            max_len = max(len(t) for t in tokenized)




            pos_types = []
            for pos in range(max_len):
                tokens_at_pos = [log[pos] for log in tokenized if len(log) > pos]
                pos_types.append(len(set(tokens_at_pos)))



            vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1,1), min_df=1, stop_words=None, token_pattern=r"(?u)<\*>|\b\w+\b")
            X_tfidf = vectorizer.fit_transform(log_contents).toarray()
            vocab = vectorizer.get_feature_names_out()
            vocab_index = {t: i for i, t in enumerate(vocab)}



            X_weighted = np.copy(X_tfidf)
            for log_idx, tokens in enumerate(tokenized):
                for pos, token in enumerate(tokens):
                    if token in vocab_index:
                        i = vocab_index[token]

                        safe_pos_type = max(pos_types[pos], 2)
                        w_pos = 1 / math.log(safe_pos_type)
                        X_weighted[log_idx, i] *= w_pos


            v_tfidf = X_weighted
            v_tfidf = normalize(v_tfidf, axis=1)


            w_s, w_t = 1, 0.8
            v_concat = np.hstack([w_s * v_sbert, w_t * v_tfidf])
            v_concat = normalize(v_concat, norm='l2', axis=1)

            normalized_embeddings = v_concat

        if solution == 'Entropy+WeightedTF-IDF':

            tokenized_log_tokens = []



            log_tokens_list = [log.split() for log in log_contents]

            max_len = max(len(toks) for toks in log_tokens_list)


            def filter_columns_by_token(log_tokens_list):
                """
                Filters out columns (token positions) where any token should be removed by tokenize1.
                Returns filtered_logs: list of lists (tokens per line for each log).
                """




                max_cols = max(len(row) for row in log_tokens_list)
                num_logs = len(log_tokens_list)


                padded = [row + [None]*(max_cols - len(row)) for row in log_tokens_list]

                columns = [[padded[row_idx][col_idx] for row_idx in range(num_logs)] for col_idx in range(max_cols)]


                def is_removed(token, tokenize1_fn):

                    if token is None:
                        return False
                    out = tokenize1_fn(token)

                    return token not in out


                columns_to_keep = []
                for col_idx, col in enumerate(columns):
                    col_str = " ".join(col)
                    output = self.tokenize1(col_str)
                    if output == col:
                        columns_to_keep.append(col_idx)


                filtered_logs = []
                for row_idx in range(num_logs):
                    new_tokens = []
                    for col_idx in columns_to_keep:
                        token = padded[row_idx][col_idx]
                        if token is not None:
                            new_tokens.append(token)
                    filtered_logs.append(new_tokens)
                return filtered_logs


            tokenized_log_tokens = filter_columns_by_token(log_tokens_list)



            pos_token_counts = defaultdict(Counter)
            max_len = max(len(toks) for toks in tokenized_log_tokens)
            for toks in tokenized_log_tokens:

                for i, tok in enumerate(toks):
                    if tok == 'SOCKS5':
                        pos_logger.debug(f"Found 'SOCKS5' at position {i}")
                    pos_token_counts[i][tok] += 1



            pos_entropy_norm, pos_token_types, pos_repeat_ratios = [], [], []
            for pos, counter in pos_token_counts.items():
                total = sum(counter.values())
                max_count = max(counter.values()) if counter else 1
                repeat_ratio = max_count / total if total > 0 else 0
                probs = [c / total for c in counter.values()]

                H = -sum(p * math.log(p) for p in probs)
                Vp = len(counter)
                H_max = math.log(Vp) if Vp > 1 else 1.0
                H_norm = H / H_max if H_max > 0 else 0
                if '<*>' in counter:
                    Vp = 10
                pos_entropy_norm.append(H_norm)
                pos_token_types.append(Vp)
                pos_repeat_ratios.append(repeat_ratio)
                pos_logger.debug(f"pos={pos}, H={H:.4f}, H_norm={H_norm:.4f}, |V_p|={Vp}, repeat_ratio={repeat_ratio:.4f}, tokens={dict(counter)}")

                """
                pos_types = []
                for pos in range(max_len):
                    tokens_at_pos = [logList[pos] for logList in tokenized if len(logList) > pos]
                    type_num = 1 if '<*>' in tokens_at_pos else len(set(tokens_at_pos))
                    pos_types.append(type_num)

                pos_types_non_one = [x for x in pos_types if x != 1]
                if pos_types_non_one and max(pos_types_non_one) / min(pos_types_non_one) > 3: #token change at postion don't have same tendancy, split logs based on min change position
                    min_change_pos = pos_types.index(min(pos_types_non_one))
                """


            alpha = 0.5
            max_types = 5


            pos_types_norm = [min((pos_token_types[i]-1) / max_types, 1) for i in range(len(pos_token_types))]


            pos_weight = [(1-Hn) * Tn for Hn, Tn, Rn in zip(pos_entropy_norm, pos_types_norm, pos_repeat_ratios)]


            pos_weight = [1-(Hn*Tn) for Hn, Tn, Rn in zip(pos_entropy_norm, pos_types_norm, pos_repeat_ratios)]



            r"""
            vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1,1), min_df=1, stop_words=None, token_pattern=r"(?u)<\*>|\b[\w]+\b")
            #binary=True: 0/1  lowercase=False: no lowercase; ngram_range=(1,1): 1-gram; min_df=1: at least appear once in all logs; stop_words=None: no stop words; token_pattern=r"(?u)<\*>|\b\w+\b": <*> or word
            tokenized_log_contents = []
            for log_content in log_contents:
                log_tokens = self.tokenize1(log_content)
                log_content = " ".join(log_tokens)
                tokenized_log_contents.append(log_content)
            X_tfidf = vectorizer.fit_transform(tokenized_log_contents).toarray()
            """

            vectorizer = CountVectorizer(binary=True, tokenizer=lambda x: x, lowercase=False, token_pattern=None, ngram_range=(1,1), min_df=1, stop_words=None)

            X_tfidf = vectorizer.fit_transform(tokenized_log_tokens).toarray().astype(float)


            pos_logger.debug(f"X_tfidf: {X_tfidf}")
            vocab = vectorizer.get_feature_names_out()
            vocab_index = {t: i for i, t in enumerate(vocab)}
            pos_logger.debug(f"vocab_index: {vocab_index}")

            for pos, counter in pos_token_counts.items():
                for token in counter.keys():
                    if token not in vocab_index or ("<*>" in token and len(counter) > 1):
                        pos_weight[pos] = 0



            X_weighted = X_tfidf.copy()

            for log_idx, tokens in enumerate(tokenized_log_tokens):

                for pos, token in enumerate(tokens):

                    if token in vocab_index and pos < len(pos_weight):
                        i = vocab_index[token]
                        w_pos = pos_weight[pos]
                        X_weighted[log_idx, i] *= w_pos

                        pos_logger.debug(f"token='{token}', pos={pos}, weight={w_pos:.4f}, TFIDF={X_tfidf[log_idx,i]:.4f}, new={X_weighted[log_idx,i]:.4f}")

            normalized_embeddings = X_weighted

        """
        # Determine epsilon dynamically based on the average length of logs
        avg_length = sum(len(content.split()) for content in log_contents) / len(log_contents)
        epsilon = max(0.15, min(0.3, 0.1 + (0.01 * avg_length)))  # Adjust based on log complexity
        """

        tokenized_log_tokens = []



        log_tokens_list = [log.split() for log in log_contents]

        max_len = max(len(toks) for toks in log_tokens_list)


        def filter_columns_by_token(log_tokens_list):
            """
            Filters out columns (token positions) where any token should be removed by tokenize1.
            Returns filtered_logs: list of lists (tokens per line for each log).
            """




            max_cols = max(len(row) for row in log_tokens_list)
            num_logs = len(log_tokens_list)


            padded = [row + [None]*(max_cols - len(row)) for row in log_tokens_list]

            columns = [[padded[row_idx][col_idx] for row_idx in range(num_logs)] for col_idx in range(max_cols)]


            def is_removed(token, tokenize1_fn):

                if token is None:
                    return False
                out = tokenize1_fn(token)

                return token not in out


            columns_to_keep = []
            for col_idx, col in enumerate(columns):
                col_str = " ".join(col)
                output = self.tokenize1(col_str)
                if output == col:
                    columns_to_keep.append(col_idx)


            filtered_logs = []
            for row_idx in range(num_logs):
                new_tokens = []
                for col_idx in columns_to_keep:
                    token = padded[row_idx][col_idx]
                    if token is not None:
                        new_tokens.append(token)
                filtered_logs.append(new_tokens)
            return filtered_logs


        tokenized_log_tokens = filter_columns_by_token(log_tokens_list)


        pos_token_counts = defaultdict(Counter)

        from itertools import zip_longest

        type_count_per_pos = []
        for col in zip_longest(*tokenized_log_tokens, fillvalue=None):
            tokens = [tok for tok in col if tok is not None]
            cnt = Counter(tokens)
            if "<*>" in cnt:
                type_count_per_pos.append(1)
            else:
                type_count_per_pos.append(len(cnt))



        non_one_type_pos_count = sum(1 for cnt in type_count_per_pos if cnt != 1)
        pos_logger.debug(f"Number of positions with type count != 1: {non_one_type_pos_count}")

        max_types = 5
        sim = (max_len - non_one_type_pos_count) / (max_len -((2*max_types-3)/((max_types-1)*(max_types-1)))*non_one_type_pos_count)
        sim = min(sim, 0.9)



        epsilon = 1 - sim
        pos_logger.debug(f"epsilon: {epsilon}")



        minSamples = 1
        dbscan = DBSCAN(eps=epsilon, min_samples=minSamples, metric='cosine')
        cluster_labels = dbscan.fit_predict(normalized_embeddings)
        pos_logger.debug(f"cluster_labels: {cluster_labels}")


        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)



        result_clusters = []
        for label, indices in clusters.items():
            if label == -1:
                result_clusters.extend([[i] for i in indices])
            else:
                result_clusters.append(indices)

        pos_logger.info(f"DBSCAN found {len(result_clusters)} clusters with epsilon={epsilon}")

        return result_clusters

    def computeSimilarityStats(self, sentences):
        embeddings = model.encode(sentences, convert_to_numpy=True)
        cos_sim_matrix = cosine_similarity(embeddings)
        n = len(sentences)
        sims = [cos_sim_matrix[i,j] for i in range(n) for j in range(i+1, n)]
        max_sim = np.max(sims)
        min_sim = np.min(sims)
        mean_sim = np.mean(sims)
        return max_sim, min_sim, mean_sim

    def extractOneTemplateFromCluster(self, logContents: list[str], logIndexes: list[int], param_str="<*>"):
        """
        Extract a single template from a list of log contents by comparing tokens at each position.
        If tokens at the same position are identical, keep the token. Otherwise, replace with param_str.

        :param logContents: List of log contents in the cluster
        :param logIndexes: List of original log indexes
        :param param_str: Parameter string to use as a wildcard (default: "<*>")
        :return: Tuple of (template_str, log_indexes)
        """
        if not logContents:
            return [("", [])]


        if len(set(logContents)) == 1:
            return [(logContents[0], logIndexes)]


        tokenized_logs = [log.split() for log in logContents]


        max_length = max(len(tokens) for tokens in tokenized_logs)


        template_tokens = []


        for pos in range(max_length):

            tokens_at_pos = [log_tokens[pos] for log_tokens in tokenized_logs if pos < len(log_tokens)]


            if len(set(tokens_at_pos)) == 1:
                template_tokens.append(tokens_at_pos[0])
            else:

                template_tokens.append(param_str)


        template_str = " ".join(template_tokens)

        return [(template_str, logIndexes)]

    def extractTemplatesFromClusterByTokenSemantic(self, logContents: list[str], logIndexes: list[int], param_str="<*>"):
        """
        递归模板抽取函数
        logContents: list of log strings
        logIndexes: 原始日志索引
        nlp: NLP 对象，用于检查词向量
        computeSimilarityStats: 函数，输入 token 列表，输出 max_sim, min_sim, mean_sim
        param_str: 变量 token 占位符
        """


        logTokens = [log.split() for log in logContents]
        n_positions = max(len(tokens) for tokens in logTokens)


        positionTokensList = [list(col) for col in zip(*logTokens)]
        vocabPos = []
        for i, positionTokens in enumerate(positionTokensList):


            if len(set(positionTokens)) == 1 and positionTokens[0] != "<*>"\
                and nlp.vocab.has_vector(positionTokens[0])\
                and not (positionTokens[0] in string.punctuation and positionTokens[0] not in {":", "="}):
                vocabPos.append(i)
                pos_logger.debug(f"Position {i} with token '{positionTokens[0]}' is considered vocabulary position.")

        def recursive_grouping(logIndices, pos_start, templateToken):
            """
            logIndices: 当前处理的日志索引
            pos_start: 当前 token 位置
            templateToken: 当前模板 token 列表
            """
            pos_logger.debug(f"Recursive grouping at position {pos_start} starts")
            templates = []


            if len(logIndices) == 0 or pos_start >= n_positions:


                if len(logIndices) == 0 or pos_start >= n_positions:

                    template_str = " ".join(templateToken)
                    template_str = correct_single_template(template_str)

                    original_indexes = [logIndexes[i] for i in logIndices]
                    return [(template_str, original_indexes)]




            positionTokens = positionTokensList[pos_start]
            tokenCounts = Counter(positionTokens)
            pos_logger.debug(f"Position {pos_start}: Token counts = {tokenCounts}")

            if len(positionTokensList) == 1:
                grouped_indices = defaultdict(list)
                for idx in logIndices:
                    key = logTokens[idx][pos_start]
                    grouped_indices[key].append(idx)
                for key, sub_indices in grouped_indices.items():
                    newToken = templateToken.copy()
                    newToken[pos_start] = key
                    templates.extend(recursive_grouping(sub_indices, pos_start + 1, newToken))
                return templates


            if len(set(positionTokens)) == 1:
                newToken = templateToken.copy()
                newToken[pos_start] = positionTokens[0]
                templates.extend(recursive_grouping(logIndices, pos_start + 1, newToken))
                return templates



            if len(tokenCounts) <= 2:
                grouped_indices = defaultdict(list)
                for idx in logIndices:
                    key = logTokens[idx][pos_start]
                    grouped_indices[key].append(idx)
                for key, sub_indices in grouped_indices.items():
                    newToken = templateToken.copy()
                    newToken[pos_start] = key
                    templates.extend(recursive_grouping(sub_indices, pos_start + 1, newToken))
                return templates

            if len(tokenCounts) >= 8 and pos_start != 0:
                newToken = templateToken.copy()
                newToken[pos_start] = '<*>'
                templates.extend(recursive_grouping(logIndices, pos_start + 1, newToken))
                return templates


            sampleTokens = [token for token, _ in tokenCounts.most_common(5)]
            pos_logger.debug(f"Checking sample tokens: {sampleTokens} on position {pos_start}")
            def getSimilarityWithLinguisticFeatures(tokens, pos):
                def existsInWordnet(word):
                    """
                    检查一个单词是否在 WordNet 中存在
                    自动小写并尝试名词、动词、形容词、副词
                    """
                    word = word.strip("'\"").lower()
                    for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:
                        lemma = lemmatizer.lemmatize(word, pos=pos)
                        pos_logger.debug(f"Checking WordNet for word '{word}' with lemma '{lemma}' and POS {pos}.")
                        if wordnet.synsets(lemma, pos=pos):
                            pos_logger.debug(f"Word {word} is in WordNet.")
                            return True
                        pos_logger.debug(f"Word {word} NOT in WordNet.")
                    return False

                def isStopWord(word):
                    return word.lower() in nlp.Defaults.stop_words

                def isVariable(word, pos):
                    pass

                if all(self.isSnakeMode(token)\
                       or self.isCamelMode(token)\
                       or re.match(r"^[^:]+::[^:]+$", token)\

                       or re.match(r"^([a-z]+\.)+[A-Z0-9_]+$", token)\

                       or re.match(r'^(IPV|IPv)[0-9A-Za-z]*$', token, re.IGNORECASE) for token in tokens):
                    return 0
                if any(self.isAllAlphaCapital(token) for token in tokens):
                    return 0
                if all(self.has_numbers(token) and not nlp.vocab.has_vector(token) for token in tokens):
                    return 1


                if all(((existsInWordnet(token) or isStopWord(token))\
                        and (pos == 0 or (pos > 0 and not templateToken[pos - 1] in ["=", ":", "user"]))) or re.match(r'^(IPV|IPv)[0-9A-Za-z]*$', token, re.IGNORECASE) for token in tokens):
                    pos_logger.debug(f"All tokens {tokens} exist in WordNet, likely static words.")
                    return 0
                if pos > 1 and pos - 1 in vocabPos:
                    if templateToken[pos - 1] in ["=", ":", "user"]:
                        if pos + 1 in vocabPos:
                            if positionTokensList[pos + 1][0] in {"=", ":"}:
                                return 0
                        return 1
                return 0.5

            simValue = 1
            simValue = getSimilarityWithLinguisticFeatures(sampleTokens, pos_start)
            pos_logger.debug(f"Position {pos_start}: Tokens = {sampleTokens}, Calculate Sim with rules = {simValue}")
            if simValue == 0.5:

                prefixSampleTokens = sampleTokens
                if pos_start >= 2 and pos_start - 1 in vocabPos:
                    prefixSampleTokens = [templateToken[pos_start - 1] + " " + token for token in prefixSampleTokens]
                    if pos_start - 2 in vocabPos:
                        prefixSampleTokens = [templateToken[pos_start - 2] + " " + token for token in prefixSampleTokens]
                elif pos_start >= 1 and pos_start - 1 in vocabPos:
                    prefixSampleTokens = [templateToken[pos_start - 1] + " " + token for token in prefixSampleTokens]
                elif pos_start == 0 and pos_start + 1 in vocabPos:
                    prefixSampleTokens = [token + " " + templateToken[pos_start + 1] for token in prefixSampleTokens]
                    if pos_start + 2 in vocabPos:
                        prefixSampleTokens = [token + " " + templateToken[pos_start + 2] for token in prefixSampleTokens]

                max_sim, min_sim, mean_sim = self.computeSimilarityStats(prefixSampleTokens)
                pos_logger.debug(f"Position {pos_start}: Tokens = {prefixSampleTokens}, Calculate Sim with LM model, Max Sim = {max_sim:.4f}, Min Sim = {min_sim:.4f}, Mean Sim = {mean_sim:.4f}")
                simValue = mean_sim




            if simValue < 0.5:

                grouped_indices = defaultdict(list)
                for idx in logIndices:
                    key = logTokens[idx][pos_start]
                    grouped_indices[key].append(idx)
                for key, sub_indices in grouped_indices.items():
                    newToken = templateToken.copy()
                    newToken[pos_start] = key
                    templates.extend(recursive_grouping(sub_indices, pos_start + 1, newToken))
            else:

                newToken = templateToken.copy()
                newToken[pos_start] = param_str
                templates.extend(recursive_grouping(logIndices, pos_start + 1, newToken))

            return templates


        templateTokenInit = [""] * n_positions

        templatesList = recursive_grouping(list(range(len(logContents))), 0, templateTokenInit)


        if not templatesList:
            templateStr = " ".join([t if t else param_str for t in templateTokenInit])

            templatesList = [(templateStr, logIndexes)]

        return templatesList

    def getTemplatesByUniqueLog(self, logContents: list[str], logIndexes: list[int]) -> list[Tuple[str, list[int]]]:
        """ Get the unique logs and their indexes as templates."""
        uniqueLogs = set(logContents)
        templateGroups = defaultdict(list)
        for idx, log in enumerate(logContents):
            templateGroups[log].append(logIndexes[idx])
        templatesList = []
        for templateStr, logIndexes in templateGroups.items():



            templatesList.append((templateStr, logIndexes))
        pos_logger.debug(f"Unique logs found: {len(uniqueLogs)}")
        pos_logger.debug(f"Templates extracted from unique logs: {templatesList}")
        return templatesList

    def createNewTemplateInfo(self, logStaticSeqString: str, templateStr: str, logIndexes: list[int]) -> Optional[TemplateInfo]:
        templateInfo = TemplateInfo(self.getNewTemplateId(), logStaticSeqString, templateStr)
        templateInfo.matchedLogSize += len(logIndexes)
        templateInfo.logIndexes.extend(logIndexes)
        return templateInfo

    def extractTemplatesFromClusterByLLM(self, logContents: list[str], logIndexes: list[int]):
        """
        Extract templates from a cluster of logs.
        :param cluster: List of tuples containing log index and log content.
        :return: List of (TemplateStr, [log index]) objects created from the cluster.
        """
        pos_logger.info(f"\n {'*'*50} \n")
        candidateLogs = []
        templatesList = []

        logTokens = [logString.split() for logString in logContents]

        logTokenLengths = [len(tokens) for tokens in logTokens]
        logTypeNum = len(set(logContents))
        pos_logger.debug(f"Logs token lengths: {set(logTokenLengths)}; {logIndexes[0]}:{logContents[0]}")
        if logTypeNum == 1:
            pos_logger.info("All logs are identical, using the first log as the template.")
            templateStr = logContents[0]



            return [(templateStr, logIndexes)]

        if len(set(logTokenLengths)) != 1:
            pos_logger.info("Logs have different lengths, extracting template from candidates and using LLM.")
            for length in set(logTokenLengths):
                logIds = [i for i, l in enumerate(logTokenLengths) if l == length]
                candidateLogs.append(logContents[logIds[0]])
            templateStr = self.extractTemplateByLLM(candidateLogs, 2)
            if templateStr is None:
                templatesList = self.getTemplatesByUniqueLog(logContents, logIndexes)
            else:



                templatesList = [(templateStr, logIndexes)]
            return templatesList
        else:
            pos_logger.debug("All logs have the same length, using entropy to extract template.")

            if not templatesList:

                positionTokensList = [list(col) for col in zip(*logTokens)]
                for i, positionTokens in enumerate(positionTokensList):
                   if "<*>" in positionTokens and not any(token in string.punctuation for token in positionTokens):
                        positionTokensList[i] = ["<*>"] * len(positionTokens)

                entropyList = []
                for i, positionTokens in enumerate(positionTokensList):
                    entropy = self.shannon_entropy(positionTokens)
                    entropyList.append(entropy)

                templateToken = []
                LLMisNeeded = False


                for pos, ent in enumerate(entropyList):
                    pos_logger.info(f"Position {pos}: log: {logTokens[0][pos]} Entropy = {ent:.4f}")
                    if ent == 0:
                        templateToken.append(positionTokensList[pos][0])

                    elif any(not nlp.vocab.has_vector(token) for token in positionTokensList[pos]) and not self.is_snake_or_camel(logTokens[0][pos]) and ent > 1.8:
                        templateToken.append(self.param_str)
                    elif pos > 0 and nlp.vocab.has_vector(logTokens[0][pos]) and logTokens[0][pos-1] in {"="}:
                        templateToken.append(self.param_str)
                    else:
                        LLMisNeeded = True



                template_tokens_no_punct = [token for token in templateToken if token not in string.punctuation]
                if len(template_tokens_no_punct) == 1 and template_tokens_no_punct[0] == "<*>":
                    LLMisNeeded = True

                if not LLMisNeeded:
                    pos_logger.debug("Entropy is enough to extract template, creating template from tokens.")
                    templateStr = " ".join(templateToken)



                    return [(templateStr, logIndexes)]
                if LLMisNeeded:
                    pos_logger.info("Entropy is not enough to extract template, using LLM.")



                    logContents = [" ".join(tokens) for tokens in logTokens]
                    logCounts = Counter(logContents)
                    num = 0
                    sampleLogStrings = []
                    for log, count in logCounts.items():
                        sampleLogStrings.append(log)
                        num += 1
                        if num >= 5:
                            break


                    templateStr = self.extractTemplateByLLM(sampleLogStrings)
                    pos_logger.debug(f"Extracted template by LLM: {templateStr}")
                    if templateStr == "":
                        templatesList = self.getTemplatesByUniqueLog(logContents, logIndexes)
                    else:





                        (matchedLogContents, matchedLogIndexes), (unmatchedLogContents, unmatchedLogIndexes) = self.matchLogsWithTemplate(templateStr, logContents, logIndexes)
                        if not unmatchedLogContents:
                            templatesList = [(templateStr, logIndexes)]
                        else:
                            templatesList = self.getTemplatesByUniqueLog(unmatchedLogContents, unmatchedLogIndexes)
                            templatesList.extend([(templateStr, matchedLogIndexes)])


                    return templatesList

    def matchLogsWithTemplate(self, templateStr: str, logContents: list[str], logIndexes: list[int]):
        """
        Match logs with the given template string.
        :param templateStr: The template string to match against.
        :param logContents: List of log contents to be matched.
        :param logIndexes: List of original log indexes.
        :return: Tuple of (matchedLogContents, matchedLogIndexes), (unmatchedLogContents, unmatchedLogIndexes)
        """
        matchedLogContents = []
        matchedLogIndexes = []
        unmatchedLogContents = []
        unmatchedLogIndexes = []

        from collections import defaultdict

        content_to_indexes = defaultdict(list)
        for logContent, logIndex in zip(logContents, logIndexes):
            content_to_indexes[logContent].append(logIndex)



        for logContent, logIndexes in content_to_indexes.items():
            if logContent == templateStr or self.isTemplateMachedWithWildcard(logContent, templateStr):
                matchedLogContents.extend([logContent]*len(logIndexes))
                matchedLogIndexes.extend(logIndexes)
            else:
                unmatchedLogContents.extend([logContent]*len(logIndexes))
                unmatchedLogIndexes.extend(logIndexes)
        return (matchedLogContents, matchedLogIndexes), (unmatchedLogContents, unmatchedLogIndexes)

    def shannon_entropy(self, tokens):
        total = len(tokens)
        counts = Counter(tokens)
        return -sum((count/total) * math.log2(count/total) for count in counts.values())

    def extractTemplateByLLM(self, logs: list, mode: int = 1) -> str:
        """
        Extract a template from the candidate logs using a language model.
        :param candidateLogs: List of candidate log contents.
        :return: Template object created from the candidate logs.
        """
        pos_logger.info(f"LLM candidate logs:\n" + "\n".join([f"  {log}" for log in logs]))

        def _normalize_line_local(s: str) -> str:
            s = s.strip()
            if s.startswith("`") and s.endswith("`"):
                s = s[1:-1]
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def _equal_order_insensitive(a, b) -> bool:
            na = [_normalize_line_local(x) for x in a]
            nb = [_normalize_line_local(x) for x in b]
            return Counter(na) == Counter(nb)

        ReuseLLMCallResult = True

        response = None
        isLLMCallResultFound = False
        if ReuseLLMCallResult:

            candidate_paths = [
                "LLMCalls.json",
                os.path.join("evaluator", "LLMCalls.json"),
            ]
            llm_calls = None
            for llm_calls_file in candidate_paths:
                if os.path.exists(llm_calls_file):
                    try:
                        with open(llm_calls_file, 'r', encoding='utf-8') as f:
                            llm_calls = json.load(f)

                        for call in llm_calls:
                            if "inputs" in call and "output" in call:

                                if _equal_order_insensitive(call["inputs"], logs):
                                    pos_logger.info("Found matching LLM call result in cache")
                                    response = call["output"]
                                    isLLMCallResultFound = True
                                    self.llmCallCnt += 1
                                    pos_logger.info(f"LLM JSON response(cache): {response}")
                                    print("==== LLM JSON response(cache) ===")
                                    print(f"LLM JSON response(cache): {response}")
                                    print("===========================")
                                    break

                    except Exception as e:
                        pos_logger.error(f"Error reading {llm_calls_file}: {e}")
                    break
        if not isLLMCallResultFound:


            instruction = "You are a log parse assistant to help extract log template from input logs."
            inputLogMsgs = '\n' + '\n'.join([f'`{log}`'for log in logs])


            instruction = "You are a log parse assistant to help extract log template from input logs."
            inputLogMsgs = '\n' + '\n'.join([f'`{log}`'for log in logs])

            if mode == 1:
                """
                prompt = f" " "
                You will be provided with some log messages separated by line break. You must abstract variables with `<*>` to extract the corresponding template. use one <*> to represent consective duplicate part.
                The variable type in log messages can be any of the following: ['url', 'IPv4_port', 'host_port', 'package_host', 'IPv6', 'Mac_address', 'time', 'path', 'id', 'date', 'duration', 'size', 'numerical', 'weekday_months', 'user_name'].
                Constant text and strings should not be recognized as variables.
                ## Rules for keep tokens as constants:
                - If a token is an **adjective, adverb(e.g., `succeeded`, `failed`), verb, or proper noun or domain-specific term(e.g., `HTTPS`, `IPv4`, `BSSID`, `SCREEN_ON`, `SCREEN_OFF`, `JOB_SETUP`)**, keep it as a constant.
                - If a token is a **modifier in a compound noun**(e.g., `Removable`, `illegal`, `Idle`, `Active`) and the modifier **represents a fixed attribute or label**, keep the compound noun as a constant.
                - If a token serves a subject-verb-object structure to represent a configuration key or property name(e.g., `Failed none`), then it must be treated as **constants** and not abstracted.
                - If a token **reflects behavioral information(e.g., `Accepted`, `Failed`, `Timeout`)**  — keep it as a constant.
                ##Output Format:
                {{
                "template": "your abstracted template string, e.g., \"Removable disk <*> is <*>\""
                }}

                ## Input Log Messages:
                {inputLogMsgs}

                Please generate Output result as described above with strictly valid JSON (not Python-style), parseable by json.loads.
                " " "
                """


                prompt = f"""
                Your task is to abstract log template from input log messages, variable part is marked with `<*>`.
                You should decide whether the different token in same position is a constant or variable with following rules:

                ## Rules for keep tokens as constants:
                - If a token is an **adjective, adverb, verb, or proper noun or domain-specific term(e.g., `HTTPS`, `IPv4`, `BSSID`, `SCREEN_ON`, `SCREEN_OFF`, `JOB_SETUP`, `bytes`)**, keep it as a constant.
                - If a token is a **modifier in a compound noun**(e.g., `Removable`, `illegal`, `Idle`, `Active`, `Accepted`, `Failed`, `Timeout`) and the modifier **represents a fixed attribute or label**, keep the compound noun as a constant.
                - If a token serves as the subject in a sentence with a subject-verb-object structure to represent a configuration key or property name(e.g., `Failed none`), then it must be treated as **constants** and not abstracted.
                - Do not treat "user" and "users", "service" and "services", or other singular/plural variants as the same — keep it as a constant.
                - If a token appears as the value in a key:value or key=value pair, abstract it as <*> while keeping the key constant — unless the value is domain-specific (e.g., SCREEN_ON) or a proper noun.
                - If a token is in a `from X to Y` pattern, X and Y are treated as variable.
                - If the value is a boolean (e.g., true, false), always abstract it as <*>, keeping the key unchanged.
                - Replace variable parts (IDs, numbers, IPs, timestamps, user names, etc.) with wildcards `<*>`, e.g., `a1`, `B2` etc.
                - Abstract only **complete tokens** (not partial word fragments). For example, `ide0` and `ide1` should NOT become ide<*>, instead use `<*>` to replace the whole token.

                ## Step 1: Token-by-Token Template Comparison

                Iterate over all token positions.
                - For each position where the tokens **differ**, generate a separate JSON object (Output-1) for this position, following the format below.
                - **Skip positions** where all logs share the same token.

                Output-1 Format:
                {{
                    "position": <index of the token position>,
                    "tokens": [list of tokens at this position across all logs],
                    "Explanation": "Your reasoning for labeling as constant or variable, based on the rules.",
                    "result": "constant" or "variable"
                }}

                ## Step 2: Template Extraction

                - If **any** position in Output-1 is labeled as `"constant"`, the logs do **not** share the same template. Return an **empty** template.
                - If **all** differing positions are labeled as `"variable"`, the logs share the same template. Replace variable tokens with `<*>` and return the template string.

                Output-2 Format:
                If logs share the same template:
                {{
                "template": "your abstracted template string, e.g., \"Removable disk <*> is <*>\""
                }}

                If logs do **not** share the same template:
                {{
                "template": ""
                }}

                ## Input Log Messages:
                {inputLogMsgs}

                Please generate ordered Output-1, Output-2 result seperately as described above with strictly valid JSON (not Python-style), parseable by json.loads:
                Output-1: <your JSON result for Step 1>
                Output-2: <your JSON result for Step 2>

                """


            else:
                prompt = f"""
                Your task is to abstract log template from input log messages with different log length.
                The common part of logs are abstracted as template, the different part are abstracted as variable marked with `<*>`. use one <*> to represent consective duplicate part.
                Treat missing fields as optional placeholders, not removable tokens. Do not drop optional fields; keep their positions using "<*>".
                Output Format:
                {{
                "template": "your abstracted template string, e.g., \"Removable disk <*> is <*>\""
                }}

                ## Input Log Messages:
                {inputLogMsgs}

                Please generate Output result as described above with strictly valid JSON (not Python-style), parseable by json.loads.
                """



            print(f"Mode={mode}, LLM input messages: {inputLogMsgs}")
            response = self.call_llm(prompt, instruction)
            pos_logger.info(f"LLM JSON response: {response}")
            print("==== LLM JSON response ===")
            print(f"LLM JSON response: {response}")
            print("===========================")

        try:

            if isinstance(response, str):
                try:

                    parsed_response = json.loads(response)
                    templates = parsed_response.get("template", "")
                    for i, tpl in enumerate(templates, 1):
                        print(f"Template {i}: {tpl}")
                    return templates[0] if templates else ""
                except json.JSONDecodeError:

                    matches = re.findall(r'\{.*?\}', response, re.DOTALL)
                    if matches:
                            json_str = matches[-1]
                            parsed_response = json.loads(json_str)
                            template = parsed_response.get("template", "")
                    else:
                        pos_logger.error(f"Could not find JSON in response: {response}")
                        template = ""

            elif isinstance(response, dict):
                pos_logger.info(f"LLM response is a dictionary: {response}")
                template = response.get("template", "")


                return template if template else ""
            else:
                pos_logger.error(f"LLM response is not a string: {response}")
                template = ""
            return template
        except Exception as e:
            pos_logger.error(f"Failed to process LLM response: {e}")
            pos_logger.error(f"Response was: {response}")
            return ""

    def generateClustersFromSimilarityMatrix(self, logSimilarityMatrix: list, logs: list[Tuple[int, str]]) -> list:
        """
        Generate clusters from the similarity matrix based on the logs.
        :param logSimilarityMatrix: Similarity matrix as a list of lists.
        :param logs: List of tuples containing log index and log content.
        :return: List of clusters generated from the similarity matrix.
        """
        import networkx as nx
        import numpy as np


        graph = nx.Graph()
        numLogs = len(logs)
        graph.add_nodes_from(range(numLogs))


        for i in range(numLogs):
            for j in range(i+1, numLogs):
                if i < len(logSimilarityMatrix) and j < len(logSimilarityMatrix[i]) and logSimilarityMatrix[i][j] == 1:
                    graph.add_edge(i, j)


        logClusters = list(nx.connected_components(graph))


        for clusterIdx, cluster in enumerate(logClusters):
            logIndices = [logs[i][0] for i in cluster]


        return logClusters

    def generateLogSimilarityMatrix(self, logs: list) -> list:
        """
        Generate a similarity matrix for the given logs based on their static tokens.
        :param logs: List of tuples containing log index and log content.
        :return: Similarity matrix as a list of lists.
        """

        matrix = [[0] * len(logs) for _ in range(len(logs))]


        for i, (logIndex1, logContent1) in enumerate(logs):
            for j, (logIndex2, logContent2) in enumerate(logs):

                if i != j:

                    similarityScore = self.calculateSimilarity(logContent1.split(), logContent2.split())

                    if self.sim_th == 0.0:
                        similarityScoreThres = 1 - (math.log(len(logContent1.split()), 2) / len(logContent1.split()))
                        similarityScore = 1 if similarityScore >= similarityScoreThres else 0

                    else:
                        similarityScore = 1 if similarityScore >= self.sim_th else 0
                else:
                    similarityScore = 1


                matrix[i][j] = similarityScore
        return matrix

    def calculateSimilarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Calculate similarity score between two token lists.
        :param tokens1: First list of tokens.
        :param tokens2: Second list of tokens.
        :return: Similarity score as a float.
        """
        if self.similarityMeasure == "jaccard":

            set1 = set(tokens1)
            set2 = set(tokens2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            if union == 0:
                return 0.0
            return intersection / union
        if self.similarityMeasure == "cosine":
            return 1
        if self.similarityMeasure == "lcs":

            if abs(len(tokens1) - len(tokens2)) > 3:
                return 0.0

            str1 = " ".join(tokens1)
            str2 = " ".join(tokens2)
            if str1 == str2:
                return 1.0
            longerStr = str1 if str1 == max(str1, str2, key=len) else str2
            shorterStr = str2 if longerStr == str1 else str1
            if shorterStr in longerStr:
                return 1.0
            if abs(len(tokens1) - len(tokens2)) > min(len(tokens1), len(tokens2)):
                return 0.0
            lcsSeq = self.LCS(tokens1, tokens2)

            return len(lcsSeq) / min(len(tokens1), len(tokens2)) if min(len(tokens1), len(tokens2)) > 0 else 0.0

    def LCS(self, seq1, seq2):
        lengths = [[0 for j in range(len(seq2)+1)] for i in range(len(seq1)+1)]

        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])


        result = []
        lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
        while lenOfSeq1!=0 and lenOfSeq2 != 0:
            if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1-1][lenOfSeq2]:
                lenOfSeq1 -= 1
            elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2-1]:
                lenOfSeq2 -= 1
            else:
                assert seq1[lenOfSeq1-1] == seq2[lenOfSeq2-1]
                result.insert(0,seq1[lenOfSeq1-1])
                lenOfSeq1 -= 1
                lenOfSeq2 -= 1
        return result

    def getCandidateTemplatesFromCache(self, logContent: str) -> list:
        """
        Get candidate templates from fwd/rev hash tables by matching first/last 3 tokens.
        Returns sorted list of TemplateInfo (by template length descending).
        """
        logTokens = logContent.strip().split()
        n = len(logTokens)


        A_keys = []
        if n >= 3:
            first3 = logTokens[:3]
            A_keys.append(tuple(first3))
            A_keys.append(("<*>", first3[1], first3[2]))
            A_keys.append((first3[0], "<*>", first3[2]))
            A_keys.append((first3[0], first3[1], "<*>"))
            A_keys.append(("<*>", first3[1], "<*>"))
        else:
            pad = tuple([logTokens[i] if i < n else "¦¦¦" for i in range(3)])
            A_keys.append(pad)
            A_keys.append(("¦¦¦", pad[1], pad[2]))
            A_keys.append((pad[0], "¦¦¦", pad[2]))
            A_keys.append((pad[0], pad[1], "¦¦¦"))
            A_keys.append(("¦¦¦", pad[1], "¦¦¦"))

        A_set = set()
        for ak in A_keys:
            ids = self.fwdTokenHashToTemplateIDs.get(ak, [])
            A_set.update(ids)


        B_keys = []
        if n >= 3:
            last3 = logTokens[-3:]
            B_keys.append(tuple(last3))
            B_keys.append(("<*>", last3[1], last3[2]))
            B_keys.append((last3[0], "<*>", last3[2]))
            B_keys.append((last3[0], last3[1], "<*>"))
            B_keys.append(("<*>", last3[1], "<*>"))
        else:
            pad = tuple([logTokens[-(n-i)] if i < n else "¦¦¦" for i in range(3)])
            B_keys.append(pad)
            B_keys.append(("¦¦¦", pad[1], pad[2]))
            B_keys.append((pad[0], "¦¦¦", pad[2]))
            B_keys.append((pad[0], pad[1], "¦¦¦"))
            B_keys.append(("¦¦¦", pad[1], "¦¦¦"))

        B_set = set()
        for bk in B_keys:
            ids = self.revTokenHashToTemplateIDs.get(bk, [])
            B_set.update(ids)

        C_set = A_set & B_set

        if not C_set:
            C_set = A_set | B_set
        pos_logger.debug(f"getCandidateTemplatesFromCache: A_set: {A_set}, B_set: {B_set}, C_set: {C_set}")
        matchedTemplateInfos = [self.idToTemplateInfos[tid] for tid in C_set if tid in self.idToTemplateInfos]
        return sorted(matchedTemplateInfos, key=lambda x: len(x.templateStr.split()), reverse=True)

    def findMatchedTemplateFromCache(self, staticTokensSequence: str, logContent: str) -> int:
        """
        Find a matched template ID from the cache based on the static tokens sequence.
        :param logContent: The original log content.
        :return: Matched template ID or -1 if not found.
        """
        """
        #sortedTemplates = sorted(self.idToTemplateInfos.values(), key=lambda x: x.matchedLogSize, reverse=True)
        logTokens = logContent.split()
        a = len(logTokens)
        #sortedTemplates = self.idToTemplateInfos.values()
        sortedTemplates = self.logStaticSeqToTemplateInfos[staticTokensSequence]
        sortedTemplates = sorted(sortedTemplates, key=lambda x: len(x.templateStr.split()), reverse=True)
        """
        sortedTemplates = self.getCandidateTemplatesFromCache(logContent)
        logTokens = logContent.strip().split()
        a = len(logTokens)
        for templateInfo in sortedTemplates:

            tmplTokens = templateInfo.templateStr.split()
            b = len(tmplTokens)
            matchNeeded = False
            for i, tmpl_token in enumerate(tmplTokens):

                if tmpl_token != "<*>":


                    if i < len(logTokens) and logTokens[i] == tmpl_token:
                        matchNeeded = True
                    else:
                        matchNeeded = False
                    break
            if not matchNeeded:
                continue
            if a == b:
                func = self.isTemplateMached




            else:
                func = self.isTemplateMachedWithWildcard


            if func(logContent, templateInfo.templateStr):
                pos_logger.debug(f"Matched template ID: {templateInfo.templateId} for log content: {logContent}")
                return templateInfo.templateId

        return None

    def isTemplateMached(self, logContent: str, templateStr: str) -> bool:

        log_tokens = logContent.strip().split()
        template_tokens = templateStr.strip().split()


        if len(log_tokens) != len(template_tokens):
            return False


        for log_tok, tpl_tok in zip(log_tokens, template_tokens):
            if log_tok == tpl_tok:
                continue
            elif log_tok == "<*>" or tpl_tok == "<*>":
                continue
            else:
                return False

        return True

    def isTemplateMachedWithWildcard(self, logContent: str, templateStr: str) -> bool:
        """
        Check if the log content matches the template string.
        :param logContent: The original log content.
        :param templateStr: The template string to match against.
        :return: True if the log content matches the template, False otherwise.
        """
        log = re.sub(r'\s+', ' ', logContent.strip())
        pattern_parts = [part.strip() for part in templateStr.split("<*>")]

        def escape_except_space(s):

            return re.sub(r'([.^$*+?{}[\]|()\\])', r'\\\1', s)

        pattern_parts_escaped = [escape_except_space(part) for part in pattern_parts]
        regex_pattern = r"\s*(.*?)\s*".join(pattern_parts_escaped)
        regex = r"^" + regex_pattern + r"$"
        pos_logger.debug(f"Checking if log content:{log} matches template: {regex}")


        import signal

        class TimeoutException(Exception):
            pass

        def timeout_handler(signum, frame):
            raise TimeoutException()

        def safe_search(pattern, string, timeout=1):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            try:
                result = re.search(pattern, string)
            except TimeoutException:
                result = None
            finally:
                signal.alarm(0)
            return result

        matches = safe_search(regex, log)
        if matches:
            wildcardValues = matches.groups()
            pos_logger.debug(f"Wildcard values found: {wildcardValues}")
            for value in wildcardValues:
                tokens = value.split()
                if len(tokens) == 1 or len(tokens) == 0:
                    continue
                if any(self.is_snake_or_camel(token) for token in tokens):
                    return False
                if len(tokens) > 1:
                    if '<*>' in tokens:
                        left = [token for token in tokens if '<*>' not in token]
                        if left != [] and (any(token in [',', '.', ';', ':', '='] for token in left) or all(token.isalpha() and nlp.vocab.has_vector(token) for token in left)):
                            return False










            return True
        else:
            pos_logger.debug(f"Log content:{log} does not match template: {regex}")
            return False
        if matches:
            return True
        else:
            return False








    def extract_last_json(self, text):
        stack = []
        start_index = None
        last_json = None

        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start_index = i
                stack.append(ch)
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack and start_index is not None:

                        last_json = text[start_index:i+1]
        return last_json

    import json

    def parse_llm_stream(self, response):
        import json

        reasoning_log = []
        answer_log = []

        buffer = ""

        for chunk in response:

            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")


            buffer += chunk


            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line.startswith("data:"):
                    continue

                json_str = line[len("data:"):].strip()
                if json_str == "[DONE]":
                    break

                try:
                    chunk_data = json.loads(json_str)
                except Exception as e:
                    print("JSON decode error:", e)
                    continue

                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                reasoning_chunk = delta.get("reasoning_content", "")
                answer_chunk = delta.get("content", "")

                if reasoning_chunk:
                    reasoning_log.append(reasoning_chunk)
                if answer_chunk:
                    answer_log.append(answer_chunk)

                if chunk_data.get("choices", [{}])[0].get("finish_reason") == "stop":
                    break

        return {
            "answer": ''.join(answer_log)
        }



    def call_llm(self, prompt: str, instruction: str = "") -> str:
        """Call an LLM API with the given prompt and return the response."""
        try:
            client = OpenAI(
                api_key=self.LLM_api_key,
                base_url=self.LLM_provider
            )

            if self.LLM_thinking:

                extra_body = {
                    "enable_thinking": True,
                    "thinking_budget": 4096,
                    "stream": True
                }
            else:

                extra_body = {
                    "enable_thinking": False,
                    "stream": False
                }
            if instruction == "":
                instruction = "You are a log parse assistant to help extract log template from input logs."

            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model = self.LLM_model,
                messages=messages,
                temperature = 0,
                extra_body=extra_body
            )
            if not response:
                pos_logger.error("No response received from LLM API")
                return '{"template": ""}'

            log_content = ""
            if self.LLM_thinking:
                result = self.parse_llm_stream(response)
                log_content = result["answer"]
            else:

                log_content = response.choices[0].message.content

            pos_logger.info("LLM response contents: %s", log_content)
            self.llmCallCnt += 1
            """
            if 'modelscope' in self.LLM_provider.lower():
                tokenizer = AutoTokenizer.from_pretrained(self.LLM_model, trust_remote_code=True)

                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_token_ids = tokenizer.encode(input_text, add_special_tokens=False)
                pos_logger.info("Prompt token IDs: %s", len(prompt_token_ids))


                response_token_ids = tokenizer.encode(log_content, add_special_tokens=False)
                pos_logger.info("Response token IDs: %s", len(response_token_ids))

                self.llmCallTokens += (len(response_token_ids) + len(prompt_token_ids))
                pos_logger.info("LLM call tokens: %s", self.llmCallTokens)
            """
            import tiktoken
            encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
            prompt_token_ids = encoder.encode(prompt)
            pos_logger.info("Prompt token IDs: %s", len(prompt_token_ids))
            response_token_ids = encoder.encode(log_content)
            pos_logger.info("Response token IDs: %s", len(response_token_ids))
            self.llmCallTokens += (len(response_token_ids) + len(prompt_token_ids))
            pos_logger.info("LLM call tokens: %s", self.llmCallTokens)

            try:
                import re
                import json

                last_json_str = self.extract_last_json(log_content)
                pos_logger.info("last_json_str: %s", last_json_str)
                if last_json_str:
                    parsed_response = json.loads(last_json_str)
                    template = parsed_response.get("template", "")
                    parsed = {"template": template}
                else:
                    pos_logger.error(f"Could not find JSON in response: {log_content}")
                    return False, ""
            except (json.JSONDecodeError, TypeError) as e:
                print(f"\nWarning: Could not parse response as JSON: {e}")
                parsed = {"template": ""}
            return parsed

        except Exception as e:
            pos_logger.error(f"Error calling LLM: {str(e)}")
            return '{"template": ""}'


    def extractTokensOfMsg1(self, msg: str) -> Tuple[List[str], List[str]]:
        """
        Analyze text from input file using spaCy and NLTK.
        First preprocess special tokens, then analyze with spaCy.

        Args:
            msg (str): The message to analyze

        Returns:
            Tuple[List[str], List[str]]: Final tokens and their POS tags
        """





        msg = msg.replace(",", ", ")

        msg = re.sub(r"\((?!\))", " (", msg)

        msg = re.sub(r"(?<!\()\)", ") ", msg)

        msg = re.sub(r"\[(?!\s)", "[ ", msg)

        msg = re.sub(r"(?!\s)\]", " ]", msg)

        initial_tokens = msg.split()
        pos_logger.debug("Initial tokenization: %s", initial_tokens)


        processed_tokens = []
        if len(initial_tokens) > 1:
            for token in initial_tokens:

                contains_special = False
                for special in ['<*>']:
                    if special in token:

                        if token == special:
                            processed_tokens.append(token)
                            contains_special = True
                            break

                        elif re.search(r'\.{2,}', token):

                            parts = re.split(r'(\.{2,})', token)

                            parts = [part for part in parts if part]
                            for i, part in enumerate(parts):

                                for sp in ['<*>']:
                                    if sp in part and part != sp and part != "'<*>'":
                                        parts[i] = sp

                            processed_tokens.extend(parts)
                            contains_special = True
                            break


                        elif '=' in token or ':' in token or '{' in token or '}' in token or '(' in token or ')' in token\
                            or '[' in token or ']' in token or ',' in token or '.' in token or '+' in token or ';' in token:

                            parts = self.split_string(token)

                            for i, part in enumerate(parts):

                                for sp in ['<*>']:
                                    if sp in part and part != sp and part != "'<*>'":
                                        parts[i] = sp

                            processed_tokens.extend(parts)
                            contains_special = True
                            break
                            r"""
                            elif re.search(r'\.{2,}', token):
                                # Split the token by ellipsis pattern, but keep the ellipsis parts
                                parts = re.split(r'(\.{2,})', token)
                                # Filter out empty strings from the result
                                parts = [part for part in parts if part]
                                for i, part in enumerate(parts):
                                    #for sp in special_words:
                                    for sp in ['<*>']:
                                        if sp in part and part != sp and part != "'<*>'":
                                            parts[i] = sp
                                            #pass
                                processed_tokens.extend(parts)
                                contains_special = True
                                break
                            """

                        else:
                            processed_tokens.append(special)

                            contains_special = True
                            break


                if not contains_special:
                    parts = self.split_string(token)
                    processed_tokens.extend(parts)
        else:

            processed_tokens = initial_tokens






        result = []
        for token in processed_tokens:
            if token == "<*>" and result and result[-1] == "<*>":
                continue
            result.append(token)


        processed_text = " ".join(result)


        processed_text = processed_text.replace("__DASH__", "-")


        result = processed_text.split()
        return result, []





        doc = nlp(processed_text)

        tokens = []
        tokenPosTag = []
        for token in doc:
            if token.text.strip():
                if token.pos_ == "NUM" and self.has_numbers(token.text):
                    tokens.append("<*>")
                else:
                    tokens.append(token.text)
            tokenPosTag.append(token.pos_)






        """
        #flair
        # 加载预训练的POS标签模型
        tagger = SequenceTagger.load("pos")
        sentence = Sentence(processed_text)
        # 预测词性
        tagger.predict(sentence)
        tokens = [token.text for token in sentence if token.text.strip()]
        tokenPosTag = [token.get_labels('pos')[0].value for token in sentence if token.text.strip()]
        print("Final POS tags: ", tokenPosTag)
        """
        """
        #nltk
        lowercaseTokens = [element.lower() for element in processed_tokens]
        posTagger = nltk.pos_tag(lowercaseTokens)
        tokenPosTag = [pos for _, pos in posTagger]
        tokens = processed_tokens
        """

        return tokens, tokenPosTag



















    def sampleLogsFromCluster(self, logContents):
        """
        Sample logs from a cluster of logs.
        """
        pos_logger.info(f"\n {'*'*50} \n")
        candidateLogs = []

        logTokens = [logString.split() for logString in logContents]

        logTokenLengths = [len(tokens) for tokens in logTokens]
        logTypeNum = len(set(logContents))
        pos_logger.debug(f"Logs token lengths: {set(logTokenLengths)}; first log: {logContents[0]}")
        if logTypeNum == 1:
            pos_logger.info("All logs are identical, using the first log as the candidate log.")
            candidateLogs.append(logContents[0])
            return candidateLogs

        if len(set(logTokenLengths)) != 1:
            pos_logger.info("Logs have different lengths, extracting template from candidates and using LLM.")
            for length in set(logTokenLengths):
                logIds = [i for i, l in enumerate(logTokenLengths) if l == length]
                candidateLogs.append(logContents[logIds[0]])
            return candidateLogs
        else:
            pos_logger.debug("All logs have the same length, using entropy to extract template.")


            positionTokensList = [list(col) for col in zip(*logTokens)]
            for i, positionTokens in enumerate(positionTokensList):
                if "<*>" in positionTokens and not any(token in string.punctuation for token in positionTokens):
                    positionTokensList[i] = ["<*>"] * len(positionTokens)


            entropyList = []
            typeNumList = []
            entropyNormList = []
            typeNumNormList = []
            max_types = 5
            for i, positionTokens in enumerate(positionTokensList):
                typeNum = len(set(positionTokens))
                typeNumList.append(typeNum)
                entropy = self.shannon_entropy(positionTokens)
                entropyMax = self.shannon_entropy([i for i in range(typeNum)])
                entropyList.append(entropy)
                entropyNormList.append(entropy / entropyMax if entropyMax > 0 else 0)
                typeNumNormList.append((typeNum-1) / (max_types - 1) if typeNum > 1 else 0)


            templateToken = []
            LLMisNeeded = False

            for pos, ent in enumerate(entropyList):
                pos_logger.info(f"Position {pos}: log: {logTokens[0][pos]} Entropy = {ent:.4f}, EntropyNorm = {entropyNormList[pos]:.4f}, TypeNum = {typeNumList[pos]}, TypeNumNorm = {typeNumNormList[pos]:.4f}")
                if entropyNormList[pos] * typeNumNormList[pos] >= 0.8:
                    templateToken.append(self.param_str)
                elif abs(ent) < 1e-9:
                    templateToken.append(positionTokensList[pos][0])
                elif any(not nlp.vocab.has_vector(token) and not self.is_snake_or_camel(token) for token in positionTokensList[pos]) and ent > 1.8:
                    templateToken.append(self.param_str)
                elif pos > 0 and nlp.vocab.has_vector(logTokens[0][pos]) and logTokens[0][pos-1] in {"="}:
                    templateToken.append(self.param_str)
                else:

                    LLMisNeeded = True


            template_tokens_no_punct = [token for token in templateToken if token not in string.punctuation and token not in ['+', '-', '*', '/', '=', '&', '^', '%', '$', '#', '@', ' ']]
            if len(set(template_tokens_no_punct)) == 1 and template_tokens_no_punct[0] == "<*>":
                LLMisNeeded = True
                pos_logger.info(f"TemplateToken is {templateToken}, too trivial, need LLM to extract template.")

            if not LLMisNeeded:

                templateStr = " ".join(templateToken)
                pos_logger.info("Template is extracted by entropy, merged template: " + templateStr)
                candidateLogs.append(templateStr)
                return candidateLogs
            else:

                pos_logger.info("Entropy is not enough to extract template, sampling candidates for LLM.")
                logContents = [" ".join(tokens) for tokens in logTokens]
                logCounts = Counter(logContents)
                num = 0
                for log, count in logCounts.items():
                    candidateLogs.append(log)
                    num += 1
                    if num >= 5:
                        break
                return candidateLogs


    def extractTemplatesByOneLLMCall(self, sampledLogGroupsForLLM):
        """
        Extract templates from a list of sampled log groups by calling LLM once.
        Input: sampledLogGroupsForLLM - list of log groups, each group is a list of log strings
        Output: list of template strings, one per group (in same order as input)
        """
        import json

        json_input = []
        for idx, log_group in sampledLogGroupsForLLM:


            logs_str = '\n'.join([f'`{log}`' for log in log_group])
            json_input.append({
                "Id": idx,
                "logs": logs_str
            })

        json_input_str = json.dumps(json_input, ensure_ascii=False, indent=2)


        instruction = "You are a log parse assistant to help extract log template from input logs."

        prompt = f"""
Your task is to extract log templates from multiple log groups. Each group contains similar logs that may or may not share the same template structure.

## Input Format:
You will receive a JSON array where each element contains:
- **Id**: The group identifier (integer index)
- **logs**: A string with multiple log lines, each wrapped in backticks (`) and separated by newlines

## Rules for Template Extraction:
- Replace variable parts (IDs, numbers, IPs, timestamps, user names, paths, interface name, domain name, etc.) with `<*>`.
- Keep constant tokens:
  - **Verbs, adjectives, adverbs, proper nouns, domain-specific terms** (e.g., `HTTPS`, `IPv4`, `BSSID`, `succeeded`, `failed`, `bytes`)
  - **Modifiers** representing fixed attributes (e.g., `Removable`, `Active`, `Accepted`, `Failed`, `Normal`) should be constant.
  - **Keys** in key:value or key=value pairs (abstract only the value as `<*>`) should be constant.
  - **Singular/plural variants** are distinct (e.g., "user" vs "users" — both stay constant)
  - **Function names or methods names or file names in programming languages** (e.g., getUserInfo, get_user_info, ext3_get_inode_loc) should be constant.
- **Abstract complete tokens only** (not partial fragments). For example, `ide0` and `ide1` should become `<*>`, not `ide<*>`.
- **Boolean values** (true, false) should be abstracted as `<*>`.
- **Process names or action names(android.intent.action.SCREEN_ON)** serve as the objects of operations should be abstracted as `<*>`.
- In "from X to Y" patterns, X and Y are variables.

## Template Extraction Process (for each group):

### Step 1: Token-by-Token Analysis
For each group, if only 1 log in group, consider it as a template with check whether any token is variable, if yes, replace the token with `<*>`, otherwise return the log as template.
For each group, Iterate over all token positions.
- For each position where the tokens **differ**, generate a separate JSON object (Position-by-Position Decision output) for this position, following the format below.
- **Skip positions** where all logs share the same token at this position.

Position-by-Position Decision output for **differing tokens** at this position:
{{{{
    "position": <index of the token position>,
    "tokens": [list of tokens at this position across all logs],
    "Explanation": "Your reasoning for labeling as constant or variable, based on the rules.",
    "result": "constant" or "variable"
}}}}

### Step 2: Template Decision
- If **any** position of differing tokens is labeled as `"constant"` (semantically different fixed values), the logs do **not** share the same template. Return **empty template** (`""`).
- If **all** positions of differing tokens are labeled as `"variable"`, the logs share the same template. Replace variable tokens with `<*>` and return the template string.

## Output Format:
Return a JSON array where each element contains:
- **Id**: Same as input Id (to match groups)
- **Reason**: The explanation for the template extraction based on Step 2: Template Decision rules using above Position-by-Position Decision output
- **template**: The extracted template string with variables replaced by `<*>`, or `""` if logs don't share a template

Examples:

**Example 1 - Same template:**
Input:
{{"Id": 0, "logs": "`User alice logged in from 192.168.1.1`\\n`User bob logged in from 10.0.0.5`"}}
Output:
{{"Id": 0,
  "Reason": "Position 1: alice and bob are different user names (variable). Position 5: 192.168.1.1 and 10.0.0.5 are different IPs (variable). All differing positions are labeled as variable, the logs share the same template.",
  "template": "User <*> logged in from <*>"}}

**Example 2 - Different templates (constant differs):**
Input:
{{"Id": 1, "logs": "`System boot completed`\\n`System shutdown completed`"}}
Output:
{{"Id": 1,
  "Reason": "Position 1: boot and shutdown are different behaviors (constant). Any position is labeled as constant, the logs do not share the same template.",
  "template": ""}}

## Input JSON:
{json_input_str}

Please generate output as a valid JSON array, parseable by json.loads. Output only the JSON array, no additional text or explanations.
"""


        pos_logger.info(f"LLM input messages: {json_input_str}")
        print("==== LLM input JSON ===")
        print(f"LLM input JSON: {json_input_str}")
        print("================================")


        raw_content = None
        isLLMCallResultFound = False
        ReuseLLMCallResult = True

        if ReuseLLMCallResult:
            candidate_paths = [
                os.path.join("evaluator", "LLMCalls_1.json"),
                "LLMCalls_1.json",
            ]
            for llm_calls_file in candidate_paths:
                if os.path.exists(llm_calls_file):
                    try:
                        with open(llm_calls_file, "r", encoding="utf-8") as f:
                            llm_calls = json.load(f)
                        for call in llm_calls:
                            if isinstance(call, dict) and "inputs" in call and "output" in call:

                                if call["inputs"] == json_input:
                                    pos_logger.info(f"Found matching LLM batch call result in cache: {llm_calls_file}")
                                    raw_content = json.dumps(call["output"], ensure_ascii=False)
                                    isLLMCallResultFound = True

                                    self.llmCallCnt += 1
                                    break

                    except Exception as e:
                        pos_logger.error(f"Error reading {llm_calls_file}: {e}")
                    break

        if isLLMCallResultFound:
            print("==== LLM batch JSON response (cache) ===")
            print(f"LLM batch response: {raw_content}")
            print("========================================")
        else:
            try:
                client = OpenAI(
                    api_key=self.LLM_api_key,
                    base_url=self.LLM_provider
                )

                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt}
                ]

                extra_body = {"enable_thinking": False, "stream": False}

                llm_response = client.chat.completions.create(
                    model=self.LLM_model,
                    messages=messages,
                    temperature=0,
                    extra_body=extra_body
                )

                if not llm_response:
                    pos_logger.error("No response received from LLM API")
                    return [None] * len(sampledLogGroupsForLLM)

                raw_content = llm_response.choices[0].message.content
                print("==== LLM batch JSON response ===")
                print(f"LLM batch response: {raw_content}")
                print("================================")

                self.llmCallCnt += 1


                import tiktoken
                encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
                prompt_token_ids = encoder.encode(prompt)
                response_token_ids = encoder.encode(raw_content)
                pos_logger.info("Response token IDs: %s", len(response_token_ids))
                self.llmCallTokens += (len(response_token_ids) + len(prompt_token_ids))
                pos_logger.info("LLM call tokens (cumulative): %s", self.llmCallTokens)

            except Exception as e:
                pos_logger.error(f"Failed to call LLM: {e}")
                return [None] * len(sampledLogGroupsForLLM)


        try:

            response_text = raw_content.strip()


            if response_text.startswith("```"):

                lines = response_text.split('\n')

                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = '\n'.join(lines)

            pos_logger.debug(f"Cleaned response text: {response_text}")
            parsed_response = json.loads(response_text)


            if isinstance(parsed_response, list):

                sorted_results = sorted(parsed_response, key=lambda x: x.get("Id", 0))
                templates = [item.get("template", "") for item in sorted_results]
                pos_logger.info(f"Extracted {len(templates)} templates from LLM batch call")
                return templates
            else:
                pos_logger.error(f"LLM response is not a list: {parsed_response}")
                return [None] * len(sampledLogGroupsForLLM)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            pos_logger.error(f"Failed to parse LLM batch response: {e}")
            pos_logger.error(f"Response was: {raw_content}")
            return [None] * len(sampledLogGroupsForLLM)

    def extractAllTemplatesByOneLLMCall(self, logClusters):
        """
        Extract all templates from a list of log clusters by calling LLM once.
        """
        logGroups = []
        sampledLogGroupsForLLM = []
        templateResults = []
        templateExtractionStartTime = time.perf_counter()
        for idx, (logStaticSeqString, cluster) in enumerate(logClusters):
            logContentsOfsameCluster = [cluster[i][1] for i in range(len(cluster))]
            logIndexesOfsameCluster = [cluster[i][0] for i in range(len(cluster))]
            logGroups.append((logStaticSeqString, logContentsOfsameCluster, logIndexesOfsameCluster))
            sampledLogsForLLM = self.sampleLogsFromCluster(logContentsOfsameCluster)
            if len(sampledLogsForLLM) == 1:
                pos_logger.info(f"Only 1 raw log or merged log in cluster {idx}, using the log as template.")
                templateStr = sampledLogsForLLM[0]
                templateResults.append({"Id": idx, "template": templateStr})
            else:
                sampledLogGroupsForLLM.append((idx, sampledLogsForLLM))

        if sampledLogGroupsForLLM:
            templateLLMResults = self.extractTemplatesByOneLLMCall(sampledLogGroupsForLLM)
            for i, templateLLMResult in enumerate(templateLLMResults):
                templateResults.append({"Id": sampledLogGroupsForLLM[i][0], "template": templateLLMResult})

        templateExtractionEndTime = time.perf_counter()
        self.templateExtractionTime += (templateExtractionEndTime - templateExtractionStartTime)

        for i in range(len(templateResults)):
            templatesList = []
            id = templateResults[i]["Id"]
            templateStr = templateResults[i]["template"]
            logStaticSeqString = logGroups[id][0]
            logContents = logGroups[id][1]
            logIndexes = logGroups[id][2]
            templateExtractionStartTime = time.perf_counter()

            if templateStr == None or templateStr == "":
                pos_logger.debug(f"Template extraction failed or returned empty for cluster {id}, using unique log method.")
                templatesList = self.getTemplatesByUniqueLog(logContents, logIndexes)
            else:
                (matchedLogContents, matchedLogIndexes), (unmatchedLogContents, unmatchedLogIndexes) = self.matchLogsWithTemplate(templateStr, logContents, logIndexes)
                if not unmatchedLogContents:
                    templatesList = [(templateStr, logIndexes)]
                else:
                    templatesList = self.getTemplatesByUniqueLog(unmatchedLogContents, unmatchedLogIndexes)
                    templatesList.extend([(templateStr, matchedLogIndexes)])
            templateExtractionEndTime = time.perf_counter()
            self.templateExtractionTime += (templateExtractionEndTime - templateExtractionStartTime)

            for templateStr, logIndexes in templatesList:


                pos_logger.debug(f"Checking templateStr: {templateStr} for merging with existing templates.")
                toMergeTemplateInfos = self.collectCanMergeTemplateInfos(templateStr)

                if toMergeTemplateInfos:
                    candidateLogs = [templateInfo.templateStr for templateInfo in toMergeTemplateInfos]
                    candidateLogs.append(templateStr)
                    candidateLogTokens = [log.split() for log in candidateLogs]
                    candidateLogTokensLengths = [len(tokens) for tokens in candidateLogTokens]
                    isLengthSame = len(set(candidateLogTokensLengths)) == 1

                    simpleMerge = False
                    if isLengthSame == True or simpleMerge == True:



                        newTemplateToken = []
                        for tokens in zip(*candidateLogTokens):

                            pos_logger.debug(f"Merging tokens: {tokens}")
                            if len(set(tokens)) == 1:
                                newTemplateToken.append(tokens[0])
                            else:
                                newTemplateToken.append("<*>")
                        pos_logger.debug(f"Tokens after merging: {newTemplateToken}")
                        newTemplateStr = " ".join(newTemplateToken)
                        pos_logger.debug(f"New merged templateStr: {newTemplateStr} is created from all existing templates.")
                    else:


                        newTemplateStr = self.extractTemplateByLLM(candidateLogs, 2)



                    mergedLogIndexes = []
                    for templateInfo in toMergeTemplateInfos:
                        pos_logger.debug(f"collecting existing template: {templateInfo.templateId} raw log indexes: {templateInfo.logIndexes} to new template.")
                        mergedLogIndexes.extend(templateInfo.logIndexes)
                    mergedLogIndexes.extend(logIndexes)

                    templateInfo = self.createNewTemplateInfo(logStaticSeqString, newTemplateStr, mergedLogIndexes)


                    pos_logger.info(f"New templateInfo created with ID: {templateInfo.templateId}, templateStr: {templateInfo.templateStr}, logIndexes: {templateInfo.logIndexes}")
                    self.idToTemplateInfos[templateInfo.templateId] = templateInfo
                    self.addTemplateToHashTables(templateInfo)

                    for templateInfo in toMergeTemplateInfos:
                        pos_logger.info(f"Removing merged template ID: {templateInfo.templateId} from idToTemplateInfos")
                        del self.idToTemplateInfos[templateInfo.templateId]
                        self.removeTemplateFromHashTables(templateInfo)
                else:
                    templateInfo = self.createNewTemplateInfo(logStaticSeqString, templateStr, logIndexes)
                    self.idToTemplateInfos[templateInfo.templateId] = templateInfo
                    self.addTemplateToHashTables(templateInfo)
                    pos_logger.info(f"New templateInfo created with ID: {templateInfo.templateId}, templateStr: {templateInfo.templateStr}, logIndexes: {templateInfo.logIndexes}")
            templatePostExtractionEndTime = time.perf_counter()
            self.templateSelfCorrectingTime += (templatePostExtractionEndTime - templateExtractionEndTime)

    def getAllTemplateInfo(self):
        return self.idToTemplateInfos

    def getAllLogTemplateMapping(self):
        logTemplateSummary = {}
        for id, info in self.idToTemplateInfos.items():
            for logIndex in info.logIndexes:

                logTemplateSummary[logIndex] = (id, info.templateStr)

        sorted_log_indices = sorted(logTemplateSummary.keys())
        template_ids = []
        template_strs = []

        for idx in sorted_log_indices:
            template_id, template_str = logTemplateSummary[idx]
            template_ids.append(template_id)
            template_strs.append(template_str)

        return template_ids, template_strs
