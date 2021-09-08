import re
from typing import List, Tuple

from bs4 import BeautifulSoup

from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

punkt_param = PunktParameters()
abbreviation = ['u.s.a', 'fig', 'etc', 'eg', 'mr', 'mrs', 'e.g', 'no', 'vs', 'i.e']
punkt_param.abbrev_types = set(abbreviation)
tokenizer = PunktSentenceTokenizer(punkt_param)


def find_pairs(s: str, start: str, end: str) -> List[Tuple[int, int]]:
    string_indices = []
    start_loc = s.find(start)
    end_len = len(end)
    while start_loc != -1:
        end_loc = s.find(end, start_loc + 1) + end_len
        string_indices.append((start_loc, end_loc))
        start_loc = s.find(start, end_loc + 1)
    return string_indices


def adjust_pos(adjusted_start_pos, adjusted_end_pos, start_pos, sections, offset):
    for section_start, _ in sections:
        if start_pos > section_start:
            adjusted_start_pos -= offset
            adjusted_end_pos -= offset
    return adjusted_start_pos, adjusted_end_pos


class Annotation:
    def __init__(self, counter, annotation, start, end, text):
        self.counter = counter
        self.annotation = annotation
        self.start = start
        self.end = end
        self.text = text

    @property
    def type(self):
        return self.annotation

    def unwrap_textbound(self):
        return self.annotation, self.start, self.end, self.text

    def __str__(self):
        return f"T{self.counter}\t{self.annotation} {self.start} {self.end}\t{self.text}"


class StackoverflowInfoExtract:
    def __init__(self, annotation_folder):
        self.annotation_folder = annotation_folder

    def extract_xml_text(self, input_text):
        input_text_str = input_text.encode("utf-8").strip()
        extracted_text = ""

        soup = BeautifulSoup(input_text_str, "lxml")
        all_tags = soup.find_all(True)
        for para in soup.body:
            text_for_current_block = str(para)

            #
            temp_soup = BeautifulSoup(text_for_current_block, "lxml")
            list_of_tags = [tag.name for tag in temp_soup.find_all()]
            tag_len = len(list_of_tags)

            if set(list_of_tags) == set('html', 'body', 'pre', 'code'):
                code_string = temp_soup.pre.string
                temp_soup.pre.string = f"CODE_BLOCK: id_{self.code_file_number} (code omitted for annotation)\n"
            elif "code" in list_of_tags:
                all_inline_codes = temp_soup.find_all("code")

                for inline_code in all_inline_codes:
                    inline_code_string_raw = str(inline_code)

                    temp_code_soup = BeautifulSoup(inline_code_string_raw, "lxml")
                    inline_code_string_list_of_text = temp_code_soup.findAll(text=True)
                    inline_code_string_text = "".join(inline_code_string_list_of_text).strip()

                    try:
                        if "\n" in inline_code_string_text:
                            code_string = inline_code_string_text

                            inline_code.string = "CODE_BLOCK: id_" + str(
                                self.code_file_number) + " (code omitted for annotation)\n"



                        elif inline_code_string_text.count('.') >= 1:
                            inline_code.string = "--INLINE_CODE_BEGIN---" + inline_code_string_text.replace(".",
                                                                                                            "..").replace(
                                '\r', '').replace('\n', '') + "--INLINE_CODE_END---"

                        elif inline_code_string_text.count('?') >= 1:
                            inline_code.string = "--INLINE_CODE_BEGIN---" + inline_code_string_text.replace("?",
                                                                                                            "<?-?>").replace(
                                '\r', '').replace('\n', '') + "--INLINE_CODE_END---"
                        else:
                            inline_code.string = "--INLINE_CODE_BEGIN---" + inline_code_string_text.replace('\r',
                                                                                                            '').replace(
                                '\n', '') + "--INLINE_CODE_END---"
                    except Exception as e:

                        print("DEBUG----- inisde except for inline code -------- error", e)

                        continue

            if "blockquote" in list_of_tags:
                op_string = temp_soup.blockquote.string
                temp_soup.blockquote.string = "OP_BLOCK: (output omitted for annotation)\n"

            if "kbd" in list_of_tags:
                all_keyboard_ip = temp_soup.find_all("kbd")
                # print("DEBUG-keyboard-input", all_keyboard_ip)
                # print("DEBUG---inputxml: ",input_text_str)
                for keyboard_ip in all_keyboard_ip:
                    print("DEBUG-keyboard-input", keyboard_ip.string)
                    keyboard_ip.string = "--KEYBOARD_IP_BEGIN---" + keyboard_ip.string + "--KEYBOARD_IP_END---"

                # print(keyboard_ip.string)

            list_of_texts = temp_soup.findAll(text=True)
            text = "".join(list_of_texts)

            extracted_text += text
            extracted_text += "\n\n"

        # print("DEBUG--extracted-text-for-xml: \n",extracted_text)

        return extracted_text


def tokenize_and_annotate_post_body(body, post_id):
    tokenized_body = tokenizer.tokenize(body)

    cleaned_tokens = []
    for sentence in tokenized_body:
        if "--INLINE_CODE_BEGIN---" in sentence:
            sentence = sentence.replace("..", ".")
            sentence = sentence.replace("<?-?>", "?")
        sentence = re.sub(r"\n+", "\n", sentence)
        cleaned_tokens.append(sentence)

    tokenized_body_str = "\n".join(cleaned_tokens)

    inline_code_sections = find_pairs(tokenized_body_str, "--INLINE_CODE_BEGIN", "INLINE_CODE_END---")
    keyboard_ip_sections = find_pairs(tokenized_body_str, "--KEYBOARD_IP_BEGIN", "KEYBOARD_IP_END---")
    code_block_sections = find_pairs(tokenized_body_str, "CODE_BLOCK:", "(code omitted for annotation)")
    op_block_sections = find_pairs(tokenized_body_str, "OP_BLOCK:", "(output omitted for annotation)")

    question_url_text = f"Question_URL: https://stackoverflow.com/questions/{post_id}/"
    intro_text = ""
    op_string = tokenized_body_str \
        .replace("--INLINE_CODE_BEGIN---", "") \
        .replace("--INLINE_CODE_END---", "") \
        .replace("--KEYBOARD_IP_BEGIN---", "") \
        .replace("--KEYBOARD_IP_END---", "")

    annotations = []

    # -----------------creating automated annotations---------------------------
    tag_counter = 1
    init_offset = len(intro_text)

    len_keyboard_tag_str = len("--KEYBOARD_IP_BEGIN---" + "--KEYBOARD_IP_END---")

    # -----------------annotation for inline codes------------------------------
    inline_code_start = len("--INLINE_CODE_BEGIN---")
    inline_code_end = len("--INLINE_CODE_END---")
    inline_code_tag_offset = inline_code_start + inline_code_end

    for index, (start_pos, end_pos) in enumerate(inline_code_sections):
        raw_code_string = tokenized_body_str[start_pos:end_pos]
        code_string = raw_code_string[inline_code_start:-inline_code_end] \
            .replace('\r', '').replace('\n', '')
        # print(code_string)
        annotation_text = "Code_Block"

        # ---------adjust annotation location for keyboard ips------------------------
        adjusted_start, adjusted_end = adjust_pos(
            start_pos, end_pos, start_pos,
            keyboard_ip_sections, len_keyboard_tag_str
        )
        begin_loc = adjusted_start + init_offset - (index * inline_code_tag_offset)
        ending_loc = adjusted_end + init_offset - ((index + 1) * inline_code_tag_offset)
        annotation = Annotation(tag_counter, annotation_text, begin_loc, ending_loc, code_string)
        annotations.append(annotation)
        tag_counter += 1

    # -----------------annotation for output block------------------------------
    for start_pos, end_pos in op_block_sections:
        op_string = tokenized_body_str[start_pos:end_pos]
        # print(code_string)
        annotation_text = "Output_Block"

        # ---------adjust annotation location for inline code---------------
        adjusted_start, adjusted_end = adjust_pos(
            start_pos, end_pos, start_pos,
            inline_code_sections, inline_code_tag_offset
        )

        # ---------adjust annoatiotion location for keyboard ips------------------------
        adjusted_start, adjusted_end = adjust_pos(
            adjusted_start, adjusted_end, start_pos,
            keyboard_ip_sections, len_keyboard_tag_str
        )

        begin_loc = adjusted_start + init_offset
        ending_loc = adjusted_end + init_offset
        annotation = Annotation(tag_counter, annotation_text, begin_loc, ending_loc, op_string)
        annotations.append(annotation)
        tag_counter += 1

    # -----------------annotation for keyboard input------------------------------
    keyboard_tag_offset = len("--KEYBOARD_IP_BEGIN---") + len("--KEYBOARD_IP_END---")

    for index, (start_pos, end_pos) in enumerate(keyboard_ip_sections):
        keyboard_ip_string = tokenized_body_str[start_pos:end_pos] \
            .replace("--KEYBOARD_IP_BEGIN---", "") \
            .replace("--KEYBOARD_IP_END---", "")
        # print(code_string)
        annotation_text = "Keyboard_IP"

        # ---------adjust annotation location for inline code---------------
        adjusted_start, adjusted_end = adjust_pos(
            start_pos, end_pos, start_pos,
            inline_code_sections, inline_code_tag_offset
        )

        begin_loc = adjusted_start + init_offset - (index * keyboard_tag_offset)
        ending_loc = adjusted_end + init_offset - ((index + 1) * keyboard_tag_offset)
        annotation = Annotation(tag_counter, annotation_text, begin_loc, ending_loc, keyboard_ip_string)
        annotations.append(annotation)
        tag_counter += 1

    # -----------------annotation for code block------------------------------
    for start_pos, end_pos in code_block_sections:
        code_string = tokenized_body_str[start_pos:end_pos]
        annotation_text = "Code_Block"

        # ---------adjust annotation location for inline code---------------
        adjusted_start, adjusted_end = adjust_pos(
            start_pos, end_pos, start_pos,
            inline_code_sections, inline_code_tag_offset
        )

        # ---------adjust annotation location for keyboard ips------------------------
        adjusted_start, adjusted_end = adjust_pos(
            adjusted_start, adjusted_end, start_pos,
            keyboard_ip_sections, len_keyboard_tag_str
        )

        begin_loc = adjusted_start + init_offset
        ending_loc = adjusted_end + init_offset
        annotation = Annotation(tag_counter, annotation_text, begin_loc, ending_loc, code_string)
        annotations.append(annotation)
        tag_counter += 1

    return intro_text + op_string + "\n", annotations
