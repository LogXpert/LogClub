import datetime
import pandas as pd
import os, re
import json
import logging
import sys
import time
from os.path import dirname
import logging
import logging.config
import re
import math
from tabulate import tabulate
from tqdm import tqdm

pd.options.display.max_colwidth = None

if os.path.exists("./result.log"):
    os.rename("./result.log", "./result.log.prev")

result_logger = logging.getLogger("result_logger")

result_logger.setLevel(logging.DEBUG)
result_handler = logging.FileHandler("./result.log", mode="w")
result_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
result_logger.addHandler(result_handler)

result_logger.info("Starting evaluation at datetime: %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

BLUE = "\033[34m"
RESET = "\033[0m"
YELLOW = "\033[33m"
GREEN = "\033[32m"

CHUNK_SIZE = 10000

benchmark_settings = {
    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [],
        'filter': []
        },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \\[<ADDR>\\] <Content>',
        'regex': ["(\\w+-\\w+-\\w+-\\w+-\\w+)", r'HTTP\/\d+\.\d+', r'(?:\d{1,3}\.){3}\d{1,3},(?:\d{1,3}\.){3}\d{1,3}', ],
        'filter': [r'HTTP\/\d+\.\d+', ]
        },


    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [],
        'filter': []
        },

    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+'],
        'filter': []
        },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \\[<Process>\\] <Component>: <Content>',

        'regex': [],
        'filter': []
        },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'regex': [r'\b\d+\s*(B|KB|MB|GB)\b'],
        'filter': []
        },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \\[<Node>:<Component>@<Id>\\] - <Content>',
        'regex': [r'\b\d+ms\b', ],
        'filter': []
        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\\[<PID>\\])?: <Content>',
        'regex': [r'::ffff:(?:\d{1,3}\.){3}\d{1,3}', r'(\d+\.){3}\d+', r'\bLOCAL\(\d+\)', r'_([A-Z]\d+)', r'\b\d+(B|kB|mB)\b'],
        'filter': []
        },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [],
        'filter': []
        },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\\[<PID>\\])?: <Content>',
        'regex': [r'[a-z_]+:[a-z_]+:[a-z_]+'],
        'filter': []
        },

    'Andriod': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b',
                  r'-\<\*\>'],
        'filter': []
        },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\\|<Component>\\|<Pid>\\|<Content>',
        'regex': [],
        'filter': []
        },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\\[<Time>\\] \\[<Level>\\] <Content>',
        'regex': [r'\/(?:\w+\/){2,}\w+\.\w+$'],
        'filter': []
        },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\\[<Time>\\] <Program> - <Content>',
        'regex': [r'\(\d+(\.\d+)?\s(?:K|M|G|T)B\)', r'\(IPv[46]\)', r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*' ],

        'filter': []
        },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\\[<Pid>\\]: <Content>',

        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+', r'(\.\d+){2,}'],
        'filter': []
        },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\\[<PID>\\]( \\(<Address>\\))?: <Content>',

        'regex': [r'([\w-]+\.){2,}[\w-]+', r'([a-f0-9]{2}:){5}[a-f0-9]{2}'],
        'filter': []
        },

    'Thunderbird_epd': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\\[<PID>\\])?: <Content>',
        'regex': [],
        'filter': []
        },

    'Audit': {
        'log_file': 'Audit/Audit_2k.log',
        'log_format': "type=<Type> msg=audit\\(<Time>\\): <Content>",
        'regex': [],
        'filter': []
    },
}

class format_log:
    def __init__(self, log_format, indir='./'):
        self.path = indir
        self.logName = None
        self.df_log = None
        self.log_format = log_format

    def get_format_logs(self, logName):
        self.logName=logName
        self.load_data()
        return self.df_log

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(r' +', r'\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r', encoding='UTF-8') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf


    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)


def preprocess(line, rex, filter):
    for currentFil in filter:
        line = re.sub(currentFil, '', line)
    for currentRex in rex:
        line = re.sub(currentRex, '<*>', line)
    return line

def isTemplateSameWithGroundtruth(template, groundtruth):
    """
    Compare the template with the groundtruth template.
    :param template: The template generated by the miner.
    :param groundtruth: The groundtruth template.
    :return: True if they are the same, False otherwise.
    """
    groundtruth = re.sub(r'(<\*>[\s]*){2,}', '<*> ', groundtruth)
    normalized_groundtruth = re.sub(r'\s+', '', groundtruth).strip()
    normalized_template = re.sub(r'\s+', '', template).strip()

    return normalized_template == normalized_groundtruth

def parse_file_to_table(file_path):

    pattern = re.compile(
        r"Evaluation on Dataset: (\S+), Group Accuracy: ([\d.]+), PA Accuracy: ([\d.]+), FGA: ([\d.]+), FTA: ([\d.]+), "
        r"Duration: ([\d.]+) sec, "
        r"Total of (\d+) lines, rate ([\d.]+) lines/sec, (\d+) clusters, "
        r"Total LLM Call count: (\d+), "
        r"Total LLM Call Tokens: (\d+), "
        r"TemplateMatchingDuration: ([\d.]+) sec, "
        r"ClusterGroupingDuration: ([\d.]+) sec, "
        r"TemplateExtractionDuration: ([\d.]+) sec, "
        r"TemplateSelfCorrectingDuration: ([\d.]+) sec"
    )


    best_data_by_dataset = {}


    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                (
                    dataset, GA_accuracy, PA_accuracy, FGA, FTA, duration, line_number, rate,
                    clusters, llm_call_cnt, llm_call_tokens,
                    template_matching_duration,
                    cluster_grouping_duration,
                    template_extraction_duration,
                    template_self_correcting_duration
                ) = match.groups()

                data = [
                    dataset,
                    float(GA_accuracy),
                    float(PA_accuracy),
                    float(FGA),
                    float(FTA),
                    float(duration),
                    int(line_number),

                    int(clusters),
                    int(llm_call_cnt),
                    int(llm_call_tokens),
                    float(template_matching_duration),
                    float(cluster_grouping_duration),
                    float(template_extraction_duration),
                    float(template_self_correcting_duration)
                ]

                if dataset not in best_data_by_dataset or data[1] > best_data_by_dataset[dataset][1]:
                    best_data_by_dataset[dataset] = data


    best_data = list(best_data_by_dataset.values())




    df = pd.DataFrame(best_data, columns=[
        "Dataset", "GA", "PA", "FGA", "FTA", "Dura", "#Line",
        "#Tpl", "#Invoke", "#Tokens", "Match", "Group", "Extract", "Correct"
    ])


    if len(df) > 0:
        avg_row = ["Average"] + [round(df[col].mean(), 4) for col in df.columns[1:]]
        df.loc[len(df)] = avg_row

    return df


benchmark_result=[]


scope_dir = os.path.join(os.path.dirname(__file__), "../code")
sys.path.insert(0, scope_dir)

from templateMiner import TemplateMiner
from templateMinerConfig import TemplateMinerConfig






def main():
    import argparse


    parser = argparse.ArgumentParser(description='Log Template Mining Evaluator')
    parser.add_argument('--dataset', type=str, default='2K',
                        help='Dataset size: 2K (default) or Full or Full50K or Full50KRaw or LOGBASE')
    parser.add_argument('--type', type=str, default='corrected',
                        help='Log file type: corrected (default) or raw')
    parser.add_argument('--method', type=str, default='BertLog',
                        help='Template mining method: BertLog (default) or LogClub')
    args = parser.parse_args()
    args.dataset = args.dataset.upper()
    if args.dataset not in ['2K', 'FULL', 'FULL50K', 'FULL50KRAW', 'LOGBASE']:
        parser.error("Dataset must be either '2K', 'Full', 'Full50K', 'Full50KRAW', or 'LOGBASE'.")

    args.type = args.type.upper()
    if args.type not in ['CORRECTED', 'RAW']:
        parser.error("Type must be either 'CORRECTED' or 'RAW'.")

    global log_file_category, log_file_type
    if args.dataset == '2K':
        log_file_category = '2k'
    elif args.dataset == 'FULL':
        log_file_category = 'full'
    elif args.dataset == 'FULL50K':
        log_file_category = 'full50k'
    elif args.dataset == 'FULL50KRAW':
        log_file_category = 'full50kraw'
    elif args.dataset == 'LOGBASE':
        log_file_category = 'logbase'
    else:
        parser.error("Dataset must be either '2K', 'Full', 'Full50K', 'Full50KRAW', or 'LOGBASE'.")
    log_file_type = 'corrected' if args.type == 'CORRECTED' else 'raw'


    config = TemplateMinerConfig()
    args.method = args.method.lower()
    if args.method == 'logclub':
        config.load(f"{dirname(__file__)}/logClub.ini")
    else:
        config.load(f"{dirname(__file__)}/bertLog.ini")
    config.profiling_enabled = True





    FullDataSets = [
        "Proxifier",
        "Apache",
        "OpenSSH",
        "HDFS",
        "OpenStack",
        "HPC",
        "Zookeeper",
        "HealthApp",
        "Hadoop",
        "Spark",
        "BGL",
        "Linux",
        "Mac",
        "Thunderbird",
    ]
    DataSets2K = [
        "Proxifier",
        "Apache",
        "OpenSSH",
        "HDFS",
        "OpenStack",
        "HPC",
        "Zookeeper",
        "HealthApp",
        "Hadoop",
        "Spark",
        "BGL",
        "Linux",
        "Mac",
        "Thunderbird",
        "Windows",
        "Andriod",
    ]
    logBase = [
        "logstash",
        "airflow",
        "flyway",
        "jib",
        "teavm",
        "spring-boot-demo",
        "ailearning",
        "scikitlearn",
        "Peergos",
        "jeromq",
        "mockito",
        "javassist",
    ]

    logBase = [
        "ailearning",
        "scikitlearn",
        "teavm",
        "javassist",
        "spring-cloud-alibaba",
        "novel",
        "auto",
        "joda-time",
        "jeromq",
        "logback",
    ]


    selectiveDataSets = [











        "Linux",




    ]




    if log_file_category == 'logbase':
        datasets = logBase
    else:
        datasets = FullDataSets if log_file_category in ['full', 'full50k'] else DataSets2K




    for dataset in datasets:
        result_logger.info("Dataset: %s, type: %s, args.type: %s", dataset, args.dataset, args.type)
        if log_file_category == 'logbase':
            setting = {
                'log_file': dataset + '.GeneralAnnotation.csv',
                'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
                'regex': [],
                'filter': []
            }
        else:
            setting = benchmark_settings[dataset]

        df = pd.DataFrame()
        sentences = []
        if log_file_category == 'full':
            setting['log_file'] = setting['log_file'].replace("2k", "full")

            structured_log_file = os.path.join(dirname(__file__), 'datasets_full', setting['log_file'] + '_structured.csv')
        elif log_file_category == 'full50k':
            setting['log_file'] = setting['log_file'].replace("2k", "50K")
            structured_log_file = os.path.join(dirname(__file__), 'loghub-2.0-50K', setting['log_file'] + '_structured.csv')
        elif log_file_category == 'full50kraw':
            setting['log_file'] = setting['log_file'].replace("2k", "50K")
            structured_log_file = os.path.join(dirname(__file__), 'loghub-2.0-50K-raw', setting['log_file'] + '_structured.csv')
        elif log_file_category == 'logbase':
            structured_log_file = os.path.join(dirname(__file__), 'logBase-GeneralAnnotation', dataset + '.GeneralAnnotation.csv')
        else:
            if log_file_type == 'corrected':
                structured_log_file = os.path.join(dirname(__file__), 'loghub-2K', setting['log_file'] + '_structured_corrected.csv')
            else:
                structured_log_file = os.path.join(dirname(__file__), 'loghub-2K', setting['log_file'] + '_structured.csv')

        df = pd.read_csv(structured_log_file)
        if 0 and log_file_category == 'full':

            def sample_event(group, n):
                if len(group) > n:
                    return group.sample(n=n, random_state=42)
                else:
                    return group

            total_samples = 20000
            df_sample = pd.DataFrame()
            """
            event_counts = df['EventId'].value_counts()
            num_events = len(event_counts)
            avg_sample_per_event = int(round(total_samples / num_events))
            for event_id, group in df.groupby('EventId'):
                sampled = sample_event(group, avg_sample_per_event)
                df_sample = pd.concat([df_sample, sampled], ignore_index=True) """

            """
            # 按 EventId 分组
            for event_id, event_group in df.groupby('EventId'):
                # 在同一 EventId 组里，按不同日志内容分组
                # If there are more than 1000 logs with the same EventId, sample 1000
                if len(event_group) > 1000:
                    # First sample a smaller set from the large group
                    event_group = event_group.sample(n=1000, random_state=42)
                for log_content, log_group in event_group.groupby('Content'):
                    #print(f"EventId: {event_id}, Log Content: {log_content}, Count: {len(log_group)}")
                    n = len(log_group)
                    if n <= 5:
                        sampled = log_group
                    else:
                        sampled = log_group.sample(n=5, random_state=42)

                    # 合并到结果 DataFrame
                    df_sample = pd.concat([df_sample, sampled], ignore_index=True)
            """

            event_counts = df['EventId'].value_counts()


            proportions = event_counts / event_counts.sum()
            allocated = (proportions * total_samples).round().astype(int)


            sampled_logs = []

            for event_id, alloc in allocated.items():
                group = df[df['EventId'] == event_id]
                actual_count = len(group)

                if alloc <= actual_count:

                    sampled = group.sample(n=alloc, random_state=42)
                else:

                    sampled = group

                sampled_logs.append(sampled)


            df_sample = pd.concat(sampled_logs, ignore_index=True)

            df = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

        for _, row in df.iterrows():
            component = row['Component'].strip() if 'Component' in df.columns and not pd.isna(row['Component']) else "<*>"
            level = row['Level'].strip() if 'Level' in df.columns and not pd.isna(row['Level']) else "<*>"
            content = row['Content'].strip()

            combined = f"{component}¦¦¦{level}¦¦¦{content}"
            sentences.append(combined)

        df_groundtruth = df
        df_data = pd.DataFrame()

        retryMax = 1
        tryCnt = 0

        while(tryCnt < retryMax):


            templateMiner = TemplateMiner(config=config)

            batch_size = 10000
            chunk_size = CHUNK_SIZE
            line_count = len(sentences)
            logTemplateIds = []
            logTemplateStrs = []
            chunkLogs = []
            time_took = 0

            for index, line in tqdm(enumerate(sentences), total=len(sentences), unit="log"):
                logComponent = line.strip().split("¦¦¦")[0]
                logLevel = line.strip().split("¦¦¦")[1]
                logContent = line.strip().split("¦¦¦")[2]
                logContent = preprocess(logContent, setting['regex'], setting['filter'])
                logContent = templateMiner.masker.mask(logContent)
                combinedLog = f"{logComponent} ¦¦¦ {logLevel} ¦¦¦ {logContent}"
                chunkLogs.append(combinedLog)
                if len(chunkLogs) >= chunk_size or index == len(sentences) - 1:
                    base = index // chunk_size
                    start_time = time.time()
                    lastChunk = True if index == len(sentences) - 1 else False
                    templateMiner.parseChunkLogs(base, chunk_size, chunkLogs, lastChunk)
                    time_took += time.time() - start_time
                    chunkLogs = []
                else:
                    continue
            templateInfos = templateMiner.getAllTemplateInfo()
            for i, templateInfo in templateInfos.items():
                result_logger.info("ID: %s, Template: %s", i, templateInfo.templateStr)

            logTemplateIds, logTemplateStrs = templateMiner.getAllLogTemplateMapping()




            rate = line_count / time_took

            df_data['EventId'] = logTemplateIds
            df_data['EventTemplate'] = logTemplateStrs

            GA_cnt = 0
            PA_cnt = 0
            TP_cnt = 0
            TA_cnt = 0
            data = df_data['EventId']
            groundtruth = df_groundtruth['EventId']
            for parsed_eventId in data.value_counts().index:
                logIds = data[data == parsed_eventId].index
                result_logger.debug("\n********parsed template id: %s for log line IDs: %s", parsed_eventId, logIds)
                result_logger.debug("********The initial created template: %s", df_data['EventTemplate'][logIds[0]])
                result_logger.debug("********The final updated template: %s", df_data['EventTemplate'][logIds[-1]])
                groundtruth_eventIDs_Of_logLines = groundtruth[logIds].value_counts()
                result_logger.debug("====above log IDs maps to groundtruth template: %s ", groundtruth_eventIDs_Of_logLines)
                if groundtruth_eventIDs_Of_logLines.size == 1:
                    groundtruth_eventId = groundtruth_eventIDs_Of_logLines.index[0]
                    groundtruth_logLines_Of_eventId = groundtruth[groundtruth == groundtruth_eventId]
                    GroupMatch = False
                    if logIds.size == groundtruth_logLines_Of_eventId.size:
                        GA_cnt += logIds.size
                        TP_cnt += 1
                        GroupMatch = True
                        result_logger.debug("&&&&&& parsed logs as one group are equal to truth, GA_cnt: %d", GA_cnt)
                    else:
                        result_logger.debug("&&&&&& parsed logs as one group is smaller than truth logs")
                        result_logger.debug("parsed log group size: %s, truth log group size: %s", len(logIds), groundtruth_logLines_Of_eventId.size)

                        result_logger.debug("groundtruth template lD: %s", groundtruth_eventId)
                        result_logger.debug("groundtruth group's first log: %s", df_groundtruth['Content'][groundtruth_logLines_Of_eventId.index[0]])
                        result_logger.debug("groundtruth template: %s", df_groundtruth['EventTemplate'][groundtruth_logLines_Of_eventId.index[0]])
                        diff = groundtruth_logLines_Of_eventId.index.difference(logIds)
                        result_logger.debug("different lines: %s", diff)
                        result_logger.debug("parsed template for first missing line: %s", df_data['EventTemplate'][diff[0]])
                        result_logger.debug("parsed template for last missing line: %s", df_data['EventTemplate'][diff[-1]])
                    if isTemplateSameWithGroundtruth(df_data['EventTemplate'][logIds[-1]], df_groundtruth['EventTemplate'][groundtruth_logLines_Of_eventId.index[0]]):
                        PA_cnt += logIds.size
                        if GroupMatch:
                            TA_cnt += 1
                        result_logger.debug("&&&&&& parsed template is same with groundtruth template, PA_cnt: %d", PA_cnt)
                    else:
                        result_logger.debug("&&&&&& parsed template is different with groundtruth template")
                        result_logger.debug("groundtruth template: %s", df_groundtruth['EventTemplate'][groundtruth_logLines_Of_eventId.index[0]])
                        result_logger.debug("parsed template: %s", df_data['EventTemplate'][logIds[-1]])
                else:
                    result_logger.debug("&&&&&& parsed logs as one group but %s groups in truth", groundtruth_eventIDs_Of_logLines.size)
                    result_logger.debug("########groundtruth label: %s", df_groundtruth['EventTemplate'][logIds])
                    result_logger.debug("@@@@@parsed output: %s", df_data['EventTemplate'][logIds])

                    for groundtruth_eventId, groundtruth_logLines_num in groundtruth_eventIDs_Of_logLines.items():
                        result_logger.debug("groundtruth template ID: %s", groundtruth_eventId)

                        gt_indices = df_groundtruth[df_groundtruth['EventId'] == groundtruth_eventId].index
                        if len(gt_indices) > 0:
                            gt_template = df_groundtruth['EventTemplate'][gt_indices[0]]
                            if isTemplateSameWithGroundtruth(df_data['EventTemplate'][logIds[-1]], gt_template):
                                PA_cnt += groundtruth_logLines_num
                                result_logger.debug("&&&&&& parsed template is same with groundtruth template(%s), PA_cnt: %d", groundtruth_eventId, PA_cnt)
                            else:
                                result_logger.debug("&&&&&& parsed template is different with groundtruth template(%s)", groundtruth_eventId)
                                result_logger.debug("groundtruth template: %s", gt_template)
                                result_logger.debug("parsed template: %s", df_data['EventTemplate'][logIds[-1]])

            """
             for parsed_eventId in data.value_counts().index:
                logIds = data[data == parsed_eventId].index
                groundtruth_eventIDs_Of_logLines = groundtruth[logIds].value_counts()
                if groundtruth_eventIDs_Of_logLines.size == 1:
                    groundtruth_eventId = groundtruth_eventIDs_Of_logLines.index[0]
                    groundtruth_logLines_Of_eventId = groundtruth[groundtruth == groundtruth_eventId]
                    GroupMatch = False
                    if logIds.size == groundtruth_logLines_Of_eventId.size:
                        GA_cnt += logIds.size
                        TP_cnt += 1
                        GroupMatch = True
                    else:
                        diff = groundtruth_logLines_Of_eventId.index.difference(logIds)
                    if isTemplateSameWithGroundtruth(df_data['EventTemplate'][logIds[-1]], df_groundtruth['EventTemplate'][groundtruth_logLines_Of_eventId.index[0]]):
                        PA_cnt += logIds.size
                        if GroupMatch:
                            TA_cnt += 1
                else:
                    for groundtruth_eventId, groundtruth_logLines_num in groundtruth_eventIDs_Of_logLines.items():
                        gt_indices = df_groundtruth[df_groundtruth['EventId'] == groundtruth_eventId].index
                        if len(gt_indices) > 0:
                            gt_template = df_groundtruth['EventTemplate'][gt_indices[0]]
                            if isTemplateSameWithGroundtruth(df_data['EventTemplate'][logIds[-1]], gt_template):
                                PA_cnt += groundtruth_logLines_num
            """
            GA_accuracy = round(float(GA_cnt) / data.size, 4)
            PGA = round(float(TP_cnt) / len(data.value_counts()), 4)
            RGA = round(float(TP_cnt) / len(groundtruth.value_counts()), 4)
            FGA = round(2 * PGA * RGA / (PGA + RGA), 4) if (PGA + RGA) > 0 else 0.0


            PA_accuracy = round(float(PA_cnt) / data.size, 4)
            PTA = round(float(TA_cnt) / len(data.value_counts()), 4)
            RTA = round(float(TA_cnt) / len(groundtruth.value_counts()), 4)
            FTA = round(2 * PTA * RTA / (PTA + RTA), 4) if (PTA + RTA) > 0 else 0.0

            GA_accuracy = f"{GA_accuracy:.4f}"
            PA_accuracy = f"{PA_accuracy:.4f}"
            FGA = f"{FGA:.4f}"
            FTA = f"{FTA:.4f}"
            result_logger.info('\n=== Evaluation on %s ==='%dataset)
            result_logger.info(f"TP_cnt: {TP_cnt}")
            print(f"\n{'*'*100} \n")
            print('Dataset:', dataset, 'GA_accuracy:', GA_accuracy, 'PA_accuracy:', PA_accuracy, 'FGA:', FGA, 'FTA:', FTA)
            print(f"\n{'*'*100} \n")
            result_logger.info(
                f"Evaluation on Dataset: {dataset}, Group Accuracy: {GA_accuracy}, PA Accuracy: {PA_accuracy}, FGA: {FGA}, FTA: {FTA}, "
                f"Duration: {time_took:.2f} sec, "
                f"Total of {line_count} lines, rate {rate:.1f} lines/sec, {len(templateMiner.scope.clusters)} clusters, "
                f"Total LLM Call count: {templateMiner.scope.llmCallCnt}, "
                f"Total LLM Call Tokens: {templateMiner.scope.llmCallTokens}, "
                f"TemplateMatchingDuration: {templateMiner.scope.templateMatchingDuration:.2f} sec, "
                f"ClusterGroupingDuration: {templateMiner.scope.clusterGroupingDuration:.2f} sec, "
                f"TemplateExtractionDuration: {templateMiner.scope.templateExtractionDuration:.2f} sec, "
                f"TemplateSelfCorrectingDuration: {templateMiner.scope.templateSelfCorrectingDuration:.2f} sec"
            )

            if config.LLM_support:
                tryCnt += 1
            else:
                break

    file_path = "./result.log"

    config_info = ""
    with open("./result.log", "r") as f:
        for line in f:
            if "Starting evaluation at datetime:" in line or "Bi-Tree:" in line or "Dataset type:" in line:
                config_info += line.strip() + "\n"
    print("\n" + config_info)

    table = parse_file_to_table(file_path)

    print(tabulate(table, headers="keys", tablefmt="github"))


    result_logger.info("\nEvaluation Results Summary:")
    result_logger.info(config_info)
    result_logger.info("\n" + tabulate(table, headers="keys", tablefmt="github"))

    table.to_csv("output.csv", index=False)


if __name__ == "__main__":
    main()












