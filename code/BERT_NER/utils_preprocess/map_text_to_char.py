import itertools
from typing import List


def peek(it):
    first = next(it)
    return first, itertools.chain([first], it)


def take_while(it, pred):
    val, p_iter = peek(it)
    while pred(val):
        next(p_iter)
        try:
            val, p_iter = peek(p_iter)
        except StopIteration:
            break
    return val, p_iter


def map_text_to_char(main_sent: str, tokens: List[str], offset: int = 0):
    tokenized_sent = itertools.chain.from_iterable(tokens)
    # print("tokenized_sent: ", tokenized_sent)

    indices = []
    main_iter = enumerate(main_sent)
    token_iter = enumerate(tokenized_sent)

    while True:
        try:
            (ms_ind, ms_char), main_iter = peek(main_iter)
            (ts_ind, ts_char), token_iter = peek(token_iter)
        except StopIteration:
            break

        (ms_ind, ms_char), main_iter = take_while(main_iter, lambda ic: ic[1] != ts_char and ic[1] == " ")
        (ts_ind, ts_char), token_iter = take_while(token_iter, lambda ic: ic[1] != ms_char and ic[1] == " ")

        if ts_char != " ":
            indices.append((ts_char, ms_ind))

        next(main_iter)
        next(token_iter)

    token_ind = 0
    token_start_pos = []
    for t in tokens:
        # Adjust tokens to account for length in original text
        t1 = t.replace("-----", " ")
        if token_ind < len(indices):
            token_start_pos.append((t, indices[token_ind][1] + offset))
        token_ind += len(t1)

    return token_start_pos


if __name__ == '__main__':
    import stokenizer  # JT: Dec 6

    expected = [('NetBeans', 3), (':', 11), ('How', 13), ('to', 17), ('use', 20), ('.jar', 24), ('files', 29),
                ('in', 35), ('NetBeans', 38), ('want(With-----a-----maximum-----size-----of-----4608x3456)', 47),
                ('?', 85)]
    # text="TextView       has setText(String), but when looking on the Doc, I don't see one for GridLayout."
    text = "   NetBeans: How to use .jar files in NetBeans want(With-----a-----maximum-----size-----of-----4608x3456)?"
    print("main text: ", text)
    tokens = stokenizer.tokenize(text)
    print("split_token: ", tokens)
    token_W_pos = map_text_to_char(text, tokens, 0)
    print("token_W_pos: ", token_W_pos)
    assert expected == token_W_pos
