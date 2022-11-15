import json
import argparse
from tw_rouge import get_rouge


def main(args):
    refs, preds = {}, {}

    with open(args.reference) as file:
        for line in file:
            line = json.loads(line)
            refs[line['id']] = line['title'].strip() + '\n'

    with open(args.submission) as file:
        for line in file:
            line = json.loads(line)
            preds[line['id']] = line['title'].strip() + '\n'

    keys =  refs.keys()
    refs = [refs[key] for key in keys]
    preds = [preds[key] for key in keys]

    print(json.dumps(get_rouge(preds, refs), indent=2))

    """
        basically, get_rogue: list[str], list[str] -> dict(
            "rogue-1", "rogue-2", "rogue-l" -> dict(
                "f" (f1), "p" (precision), "r" (recall)
            )
        )
        the baseline only uses f1 (* 100) so...
            
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference')
    parser.add_argument('-s', '--submission')
    args = parser.parse_args()
    main(args)
