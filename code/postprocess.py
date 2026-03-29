import re

def post_process(response):

    response = response.replace('\n', '')
    first_backtick_index = response.find('`')
    last_backtick_index = response.rfind('`')
    if first_backtick_index == -1 or last_backtick_index == -1 or first_backtick_index == last_backtick_index:
        tmps = []
    else:
        tmps = response[first_backtick_index: last_backtick_index + 1].split('`')
    for tmp in tmps:
        if tmp.replace(' ','').replace('<*>','') == '':
            tmps.remove(tmp)
    tmp = ''
    if len(tmps) == 1:
        tmp = tmps[0]
    if len(tmps) > 1:
        tmp = max(tmps, key=len)

    template = re.sub(r'\{\{.*?\}\}', '<*>', tmp)
    template = re.sub(r'\$\{.*?\}', '<*>', template)
    template = correct_single_template(template)
    if template.replace('<*>', '').replace(' ','') == '':
        template = ''

    return template

def exclude_digits(string):
    '''
    exclude the digits-domain words from partial constant
    '''
    pattern = r'\d'
    digits = re.findall(pattern, string)
    if len(digits) == 0 or string[0].isalpha() or any(c.isupper() for c in string):
        return False
    elif len(digits) >= 4:
        return True
    else:
        return len(digits) / len(string) > 0.3



def correct_single_template(template, user_strings=None):
    """Apply all rules to process a template.

    DS (Double Space)
    BL (Boolean)
    US (User String)
    DG (Digit)
    PS (Path-like String)
    WV (Word concatenated with Variable)
    DV (Dot-separated Variables)
    CV (Consecutive Variables)
    KV (Consecutive Key Value pairs)

    """

    boolean = {'true', 'false'}
    default_strings = {'null', 'root'}
    path_delimiters = {
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'', r'\+',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({
        r'\.', r'\-', r'\@', r'\#', r'\$', r'\%', r'\&', r'\/'
    })

    if user_strings:
        default_strings = default_strings.union(user_strings)




    template = template.strip()
    template = re.sub(r'\s+', ' ', template)


    p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)
    new_p_tokens = []
    for p_token in p_tokens:


        if re.match(r'^(\/[^\/]+)+\/?$', p_token) or re.match(r'.*/.*\..*', p_token):


            p_token = '<*>'

        new_p_tokens.append(p_token)
    template = ''.join(new_p_tokens)

    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)
    new_tokens = []
    for token in tokens:

        for to_replace in boolean.union(default_strings):

            if token == to_replace:
                token = '<*>'









        if exclude_digits(token):
            token = '<*>'


        if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token) or re.match(r'^<\*>.*<\*>$', token):
            token = '<*>'

        new_tokens.append(token)


    template = ''.join(new_tokens)


    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break



    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break

    while "#<*>#" in template:
        template = template.replace("#<*>#", "<*>")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")

    while " #<*> " in template:
        template = template.replace(" #<*> ", " <*> ")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>#<*>" in template:
        template = template.replace("<*>#<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")

    while "<*>@<*>" in template:
        template = template.replace("<*>@<*>", "<*>")

    while "<*>.<*>" in template:
        template = template.replace("<*>.<*>", "<*>")

    while ' "<*>" ' in template:
        template = template.replace(' "<*>" ', ' <*> ')




    while "<*><*>" in template:
        template = template.replace("<*><*>", "<*>")

    symbols = [" : ", " / ", " # ", " @ ", " . ", " , ", " "]

    for sym in symbols:
        pattern = f"<*>{sym}<*>"
        while pattern in template:
            template = template.replace(pattern, "<*>")



    def normalize_log(template: str) -> str:

        pattern = r'(\b\w+)\s*=\s*([^=\s;]*)(?=\s+\w+\s*=|$)'

        pairs = re.findall(r'(\b\w+)\s*=\s*([^\s;]*)?', template)

        if len(pairs) > 1 and any(value == "<*>" for _, value in pairs):

            template = re.sub(pattern, lambda m: f"{m.group(1)} = <*>", template)
        return template

    template = normalize_log(template)


    return template

def correct_single_template_1(template, user_strings=None):
    """Apply all rules to process a template.

    DS (Double Space)
    BL (Boolean)
    US (User String)
    DG (Digit)
    PS (Path-like String)
    WV (Word concatenated with Variable)
    DV (Dot-separated Variables)
    CV (Consecutive Variables)
    KV (Consecutive Key Value pairs)

    """

    boolean = {'true', 'false'}
    default_strings = {'null', 'root'}
    path_delimiters = {
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'', r'\+',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({
        r'\.', r'\-', r'\@', r'\#', r'\$', r'\%', r'\&', r'\/'
    })

    if user_strings:
        default_strings = default_strings.union(user_strings)




    template = template.strip()
    template = re.sub(r'\s+', ' ', template)


    p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)
    new_p_tokens = []
    for p_token in p_tokens:


        if re.match(r'^(\/[^\/]+)+\/?$', p_token) or re.match(r'.*/.*\..*', p_token):


            p_token = '<*>'

        new_p_tokens.append(p_token)
    template = ''.join(new_p_tokens)

    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)
    new_tokens = []
    for token in tokens:

        for to_replace in boolean.union(default_strings):

            if token == to_replace:
                token = '<*>'








        if exclude_digits(token):
            token = '<*>'


        if re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token) or re.match(r'^<\*>.*<\*>$', token):
            token = '<*>'

        new_tokens.append(token)


    template = ''.join(new_tokens)


    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break



    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break

    while "#<*>#" in template:
        template = template.replace("#<*>#", "<*>")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")

    while " #<*> " in template:
        template = template.replace(" #<*> ", " <*> ")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>#<*>" in template:
        template = template.replace("<*>#<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")

    while "<*>@<*>" in template:
        template = template.replace("<*>@<*>", "<*>")

    while "<*>.<*>" in template:
        template = template.replace("<*>.<*>", "<*>")

    while ' "<*>" ' in template:
        template = template.replace(' "<*>" ', ' <*> ')




    while "<*><*>" in template:
        template = template.replace("<*><*>", "<*>")

    symbols = [" : ", " / ", " # ", " @ ", " . ", " , ", " "]








    def normalize_log(template: str) -> str:

        pattern = r'(\b\w+)\s*=\s*([^=\s;]*)(?=\s+\w+\s*=|$)'

        pairs = re.findall(r'(\b\w+)\s*=\s*([^\s;]*)?', template)

        if len(pairs) > 1 and any(value == "<*>" for _, value in pairs):

            template = re.sub(pattern, lambda m: f"{m.group(1)} = <*>", template)
        return template

    template = normalize_log(template)


    return template