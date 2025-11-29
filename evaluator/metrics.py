# evaluator/metrics.py

def calculate_coverage(labels, check):
    def main(ttp):
        return ttp.split('.')[0] if '.' in ttp else ttp

    labels_set = set(labels)
    check_set = set(check)

    labels_main = {main(t) for t in labels}
    check_main = {main(t) for t in check}

    full = len(labels_set & check_set)

    semi = 0
    for t in labels:
        if main(t) in check_main and t not in check_set:
            semi += 1

    no = len(labels_set - check_set) - semi
    fp = len(check_set - labels_set)

    return {
        "full_coverage": full,
        "semi_coverage": semi,
        "no_coverage": no,
        "false_positive": fp
    }
